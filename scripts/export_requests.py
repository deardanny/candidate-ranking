import argparse
from pathlib import Path
from fairjob_ranker.replay import export_requests

parser = argparse.ArgumentParser(description="Export label-free replay requests")
parser.add_argument("--split", choices=["validation", "test"], default="validation")
parser.add_argument("--data", type=Path, default=Path("data/processed/fairjob-v1"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--frozen", type=Path)
args = parser.parse_args()
print(export_requests(args.data, args.split, args.output, args.frozen))
