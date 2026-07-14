#!/usr/bin/env python3
"""顺序驱动 int4/int8 量化脚本，批量量化 /data/Qwen3 与 /data/Qwen3.5 下的模型。"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOTS = (Path("/data/Qwen3"), Path("/data/Qwen3.5"))
DEFAULT_INT4_SCRIPT = Path("/data/算法组/qwen_nf4/int4量化.py")
DEFAULT_INT8_SCRIPT = Path("/data/算法组/qwen_nf4/int8量化.py")
DEFAULT_LOG_DIR = Path("/data/算法组/qwen_nf4/logs")


@dataclass(frozen=True)
class Task:
    model_dir: Path
    quant_name: str
    output_name: str
    script: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动发现 Qwen3/Qwen3.5 本地模型，并顺序生成 w4a16、w8a16。"
    )
    parser.add_argument(
        "--quant",
        choices=("both", "int4", "int8"),
        default="both",
        help="执行哪些量化任务（默认：both）。",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="只处理指定的模型文件夹名；可重复使用，例如 --model Qwen3-0.6。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="删除已有目标目录并重新量化；默认跳过已完成的目录。",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任一任务失败后立即停止；默认继续处理后续模型。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的任务，不真正量化。",
    )
    parser.add_argument(
        "--int4-device-map",
        default="cpu",
        help='传给 int4 脚本的 device_map（默认："cpu"；GPU 自动分配可用 "auto"）。',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="int8 LLM.int8() 离群值阈值（默认：6.0）。",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="保存权重时单个分片的最大大小（默认：5GB）。",
    )
    parser.add_argument(
        "--int4-script",
        type=Path,
        default=DEFAULT_INT4_SCRIPT,
        help=f"int4 量化脚本路径（默认：{DEFAULT_INT4_SCRIPT}）。",
    )
    parser.add_argument(
        "--int8-script",
        type=Path,
        default=DEFAULT_INT8_SCRIPT,
        help=f"int8 量化脚本路径（默认：{DEFAULT_INT8_SCRIPT}）。",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"日志目录（默认：{DEFAULT_LOG_DIR}）。",
    )
    return parser.parse_args()


def is_hf_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def model_size_key(path: Path) -> tuple[float, str]:
    # 取文件夹名最后一个数字作为参数规模，例如 Qwen3.5-0.8B -> 0.8。
    numbers = re.findall(r"\d+(?:\.\d+)?", path.name)
    size = float(numbers[-1]) if numbers else float("inf")
    return size, path.name.lower()


def discover_models(selected_names: list[str]) -> list[Path]:
    selected = {name.lower() for name in selected_names}
    models: list[Path] = []

    for root in DEFAULT_ROOTS:
        if not root.is_dir():
            print(f"[警告] 模型根目录不存在，已跳过：{root}", file=sys.stderr)
            continue

        for child in root.iterdir():
            if not is_hf_model_dir(child):
                continue
            if selected and child.name.lower() not in selected:
                continue
            models.append(child.resolve())

    models.sort(key=model_size_key)

    if selected:
        found = {path.name.lower() for path in models}
        missing = sorted(selected - found)
        if missing:
            print(f"[警告] 未找到以下模型目录：{', '.join(missing)}", file=sys.stderr)

    return models


def output_is_complete(output_dir: Path) -> bool:
    if not output_dir.is_dir() or not (output_dir / "config.json").is_file():
        return False

    weight_patterns = (
        "*.safetensors",
        "*.bin",
    )
    return any(
        weight_file.is_file()
        for pattern in weight_patterns
        for weight_file in output_dir.glob(pattern)
    )


def build_tasks(models: list[Path], args: argparse.Namespace) -> list[Task]:
    quant_specs: list[tuple[str, str, Path]] = []

    if args.quant in ("both", "int4"):
        quant_specs.append(("int4", "w4a16", args.int4_script.resolve()))
    if args.quant in ("both", "int8"):
        quant_specs.append(("int8", "w8a16", args.int8_script.resolve()))

    return [
        Task(
            model_dir=model_dir,
            quant_name=quant_name,
            output_name=output_name,
            script=script,
        )
        for model_dir in models
        for quant_name, output_name, script in quant_specs
    ]


def make_command(task: Task, temp_output: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(task.script),
        "--model-path",
        str(task.model_dir),
        "--output-dir",
        str(temp_output),
        "--max-shard-size",
        args.max_shard_size,
    ]

    if task.quant_name == "int4":
        command.extend(["--device-map", args.int4_device_map])
    else:
        command.extend(["--threshold", str(args.threshold)])

    return command


def stream_process(command: list[str], log_file: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_file.open("w", encoding="utf-8", buffering=1) as log:
        log.write("$ " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)

        return process.wait()


def run_task(task: Task, args: argparse.Namespace) -> tuple[str, float]:
    target_output = task.model_dir / task.output_name
    temp_output = task.model_dir / f".{task.output_name}.tmp"
    log_file = args.log_dir / f"{task.model_dir.name}_{task.output_name}.log"

    if not task.script.is_file():
        print(f"[失败] 找不到量化脚本：{task.script}")
        return "failed", 0.0

    if output_is_complete(target_output) and not args.overwrite:
        print(f"[跳过] 已存在完整产物：{target_output}")
        return "skipped", 0.0

    if target_output.exists() and not args.overwrite:
        print(
            f"[失败] 目标目录已存在但不完整：{target_output}\n"
            "       确认不需要其中内容后，使用 --overwrite 重新生成。"
        )
        return "failed", 0.0

    command = make_command(task, temp_output, args)
    print("\n" + "=" * 88)
    print(f"[开始] {task.model_dir.name} -> {task.output_name}")
    print(f"[脚本] {task.script}")
    print(f"[日志] {log_file}")
    print("[命令] " + " ".join(command))
    print("=" * 88)

    if args.dry_run:
        return "dry-run", 0.0

    if temp_output.exists():
        shutil.rmtree(temp_output)

    if target_output.exists():
        shutil.rmtree(target_output)

    start = time.monotonic()
    return_code = stream_process(command, log_file)
    elapsed = time.monotonic() - start

    if return_code != 0:
        print(f"[失败] 返回码={return_code}，耗时={elapsed / 60:.1f} 分钟")
        if temp_output.exists():
            shutil.rmtree(temp_output)
        return "failed", elapsed

    if not output_is_complete(temp_output):
        print(f"[失败] 脚本正常退出，但临时输出不完整：{temp_output}")
        if temp_output.exists():
            shutil.rmtree(temp_output)
        return "failed", elapsed

    temp_output.rename(target_output)
    print(f"[完成] {target_output}，耗时={elapsed / 60:.1f} 分钟")
    return "success", elapsed


def main() -> None:
    args = parse_args()
    args.log_dir = args.log_dir.expanduser().resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models(args.model)
    if not models:
        raise SystemExit("没有发现可量化的模型目录；模型文件夹下必须存在 config.json。")

    tasks = build_tasks(models, args)

    print("发现模型：")
    for index, model in enumerate(models, start=1):
        print(f"  {index:>2}. {model}")

    print("\n任务列表：")
    for index, task in enumerate(tasks, start=1):
        print(f"  {index:>2}. {task.model_dir.name} -> {task.output_name}")

    counts = {"success": 0, "skipped": 0, "failed": 0, "dry-run": 0}
    total_elapsed = 0.0

    for task in tasks:
        status, elapsed = run_task(task, args)
        counts[status] += 1
        total_elapsed += elapsed

        if status == "failed" and args.stop_on_error:
            break

    print("\n" + "=" * 88)
    print("批量量化汇总")
    print(f"  成功：{counts['success']}")
    print(f"  跳过：{counts['skipped']}")
    print(f"  失败：{counts['failed']}")
    if counts["dry-run"]:
        print(f"  预演：{counts['dry-run']}")
    print(f"  总耗时：{total_elapsed / 60:.1f} 分钟")
    print(f"  日志目录：{args.log_dir}")
    print("=" * 88)

    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
