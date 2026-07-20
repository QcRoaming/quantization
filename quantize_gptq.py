import argparse
from pathlib import Path

from datasets import load_dataset
from gptqmodel import GPTQConfig, GPTQModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        required=True,
        help="原始模型本地目录",
    )
    parser.add_argument(
        "--bits",
        type=int,
        required=True,
        choices=[2, 3, 4, 8],
        help="GPTQ 权重量化位数",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="校准样本数量",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="量化校准 batch size",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"模型目录缺少 config.json：{model_path}"
        )

    # 自动生成：model_path/w4a16 或 model_path/w8a16
    output_path = model_path / f"w{args.bits}a16_gptq"

    if output_path.exists():
        raise FileExistsError(
            f"输出目录已经存在，请先检查或删除：{output_path}"
        )

    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(args.num_samples))["text"]

    quant_config = GPTQConfig(
        bits=args.bits,
        group_size=128,
        desc_act=False,
        sym=True,
        damp_percent=0.1,
    )

    model = GPTQModel.load(
        str(model_path),
        quant_config,
    )

    model.quantize(
        calibration_dataset,
        batch_size=args.batch_size,
    )

    model.save(str(output_path))

    print(f"原始模型：{model_path}")
    print(f"量化位数：{args.bits} bit")
    print(f"输出目录：{output_path}")


if __name__ == "__main__":
    main()
