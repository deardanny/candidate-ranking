"""Explicit preparation and training stages; training never opens test."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validate, clean and split the pinned raw file")
    prepare.add_argument("--raw", type=Path, default=Path("data/raw/fairjob.csv.gz"))
    prepare.add_argument("--output", type=Path, default=Path("data/processed/fairjob-v1"))
    baseline = commands.add_parser("train-baseline", help="Train on train; early-stop and report on validation")
    baseline.add_argument("--data", type=Path, default=Path("data/processed/fairjob-v1"))
    baseline.add_argument("--config", type=Path, default=Path("configs/lightgbm_baseline.json"))
    baseline.add_argument("--model-output", type=Path, default=Path("models/lightgbm-v1"))
    baseline.add_argument("--report-output", type=Path, default=Path("reports/week1"))
    neural = commands.add_parser("train-pytorch", help="Train embedding + MLP and compare validation results")
    neural.add_argument("--data", type=Path, default=Path("data/processed/fairjob-v1"))
    neural.add_argument("--config", type=Path, default=Path("configs/pytorch_baseline.json"))
    neural.add_argument("--model-output", type=Path, default=Path("models/pytorch-v1"))
    neural.add_argument("--report-output", type=Path, default=Path("reports/week2"))
    neural.add_argument("--baseline", type=Path, default=Path("models/lightgbm-v1"))
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare as prepare_data
        prepare_data(args.raw, args.output)
    elif args.command == "train-baseline":
        from .baseline import train_baseline
        train_baseline(args.data, args.config, args.model_output, args.report_output)
    else:
        from .torch_training import train_pytorch
        train_pytorch(args.data, args.config, args.model_output, args.report_output, args.baseline)


if __name__ == "__main__":
    main()
