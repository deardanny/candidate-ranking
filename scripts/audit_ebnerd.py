"""Run local, aggregate-only EB-NeRD small audit; no training or test submission."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairjob_ranker.ebnerd.audit import run

if __name__ == '__main__':
    run()
