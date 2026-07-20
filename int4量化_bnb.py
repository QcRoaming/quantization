"""使用 bitsandbytes NF4（W4A16）量化并保存 Hugging Face 模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_PATH = Path("/home/jlq/project/Qwen2.5-1.5B-Instruct")
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_PATH / "w4a16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 bitsandbytes NF4 将 Hugging Face 因果语言模型量化为 W4A16。"
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
        "--device-map",
        default="cpu",
        help='模型加载位置（默认："cpu"；使用 GPU 自动分配时可传 "auto"）',
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="单个权重分片的最大大小（默认：5GB）",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="允许加载模型仓库中的自定义代码。",
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
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"正在从 {model_path} 加载并执行 NF4/W4A16 量化……", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"量化后模型内存占用：{model.get_memory_footprint() / 1024**3:.2f} GiB", flush=True)
    print(f"正在保存到 {output_dir} ……", flush=True)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)

    print("bitsandbytes NF4/W4A16 量化并保存完成。", flush=True)


if __name__ == "__main__":
    main()
