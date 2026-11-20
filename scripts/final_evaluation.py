import argparse
from pathlib import Path
from fairjob_ranker.final_evaluation import evaluate_frozen, freeze_experiment

parser = argparse.ArgumentParser(description="Freeze artifacts, then evaluate the held-out test once")
sub = parser.add_subparsers(dest="command", required=True)
freeze = sub.add_parser("freeze")
freeze.add_argument("--data", type=Path, default=Path("data/processed/fairjob-v1"))
freeze.add_argument("--model", type=Path, default=Path("models/pytorch-v1"))
freeze.add_argument("--baseline", type=Path, default=Path("models/lightgbm-v1"))
freeze.add_argument("--settings", type=Path, default=Path("configs/serving_final.json"))
freeze.add_argument("--output", type=Path, default=Path("reports/frozen_experiment.json"))
evaluate = sub.add_parser("evaluate")
evaluate.add_argument("--frozen", type=Path, default=Path("reports/frozen_experiment.json"))
evaluate.add_argument("--output", type=Path, default=Path("reports/final"))
args = parser.parse_args()
if args.command == "freeze":
    result = freeze_experiment(args.data, args.model, args.baseline, args.settings, args.output)
    print('Frozen before test evaluation:', args.output)
else:
    result = evaluate_frozen(args.frozen, args.output)
    print('Test evaluated:', result['test_rows'], 'rows; report:', args.output)
    for name, values in result['models'].items():
        print(name, values['all']['click']['logloss'], values['all']['ranking']['ndcg_at_5'])
