"""使用 bitsandbytes LLM.int8() 量化并保存 Hugging Face 模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_PATH = Path("/home/jlq/project/Qwen2.5-1.5B-Instruct")
DEFAULT_OUTPUT_DIR = Path("/home/jlq/project/Qwen2.5-1.5B-Instruct/w8a16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 bitsandbytes 将 Hugging Face 因果语言模型量化为 int8。"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"原始模型目录（默认：{DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"量化模型保存目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="LLM.int8() 离群值阈值（默认：6.0）",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="单个权重分片的最大大小（默认：5GB）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(f"找不到模型目录：{model_path}")
    if model_path == output_dir:
        raise ValueError("输出目录不能与原始模型目录相同，以免覆盖原始权重。")

    output_dir.mkdir(parents=True, exist_ok=True)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=args.threshold,
    )

    print(f"正在从 {model_path} 加载并量化模型……")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=False,
    )

    print(f"量化后模型内存占用：{model.get_memory_footprint() / 1024**3:.2f} GiB")
    print(f"正在保存到 {output_dir} ……")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)

    print("bitsandbytes int8 量化并保存完成。")


if __name__ == "__main__":
    main()