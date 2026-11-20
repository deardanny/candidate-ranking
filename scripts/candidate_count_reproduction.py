"""Run count diagnostics and nested user-fold refits without touching original test."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
from fairjob_ranker.diagnostics.candidate_count_reproduction import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/candidate_count_reproduction.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/candidate-count-reproduction'))
    parser.add_argument('--models', type=Path, default=Path('models/candidate-count-reproduction-v1'))
    parser.add_argument('--tracking-uri', default='sqlite:///'+str(Path('runs/v2/mlflow.db').resolve()))
    args = parser.parse_args()
    run(args.config, args.output, args.models, args.tracking_uri or None)
