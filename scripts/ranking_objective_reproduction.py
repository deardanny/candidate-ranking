"""Run fixed top1 selection / LambdaRank comparisons on matched user partitions."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
from fairjob_ranker.diagnostics.ranking_objective_reproduction import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/ranking_objective_reproduction.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/ranking-objective-reproduction'))
    parser.add_argument('--models', type=Path, default=Path('models/ranking-objective-reproduction-v1'))
    parser.add_argument('--tracking-uri', default='sqlite:///'+str(Path('runs/v2/mlflow.db').resolve()))
    args = parser.parse_args()
    run(args.config, args.output, args.models, args.tracking_uri or None)
