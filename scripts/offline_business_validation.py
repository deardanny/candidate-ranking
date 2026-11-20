"""Run the frozen-policy, observed-click utility audit."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
from fairjob_ranker.diagnostics.business_validation import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/offline_business_validation.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/offline-business-validation'))
    args = parser.parse_args()
    run(args.config, args.output)
