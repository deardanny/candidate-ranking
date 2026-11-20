"""Run frozen-context, cross-fitted ad-correction experiments."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
from fairjob_ranker.diagnostics.context_residual import run_experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/processed/fairjob-v1'))
    parser.add_argument('--config', type=Path, default=Path('configs/context_residual.json'))
    parser.add_argument('--models', type=Path, default=Path('models/context-residual-v1'))
    parser.add_argument('--output', type=Path, default=Path('reports/context-residual'))
    parser.add_argument('--tracking-uri', default='sqlite:///'+str(Path('runs/v2/mlflow.db').resolve()))
    args = parser.parse_args()
    run_experiment(args.data, args.config, args.models, args.output, args.tracking_uri)
