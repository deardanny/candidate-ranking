# Reproduction guide

## Environment and tests

Use Python 3.12. The dependency versions reflect the recorded local environment. A GPU is not required for the CPU experiments.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,week2,week3,v2]'
python3 scripts/run.py -m pytest -q
```

`scripts/run.py` uses the project virtual environment and, on macOS, makes scikit-learn's bundled OpenMP library visible to LightGBM in the child process. On Linux, install the platform OpenMP runtime if LightGBM reports a missing library. The repository's 131 tests use synthetic fixtures and do not require source datasets.

## Rebuild the published figures

```bash
.venv/bin/python -m pip install -r requirements-figures.txt
python3 scripts/run.py scripts/render_research_figures.py
```

The script reads the published `final_results.json` and serving `verification.json`; it does not train or select a model. Outputs include PNG, editable SVG and `figure-data.json` with the source report hashes. Arial and Times New Roman are used when available; Matplotlib can substitute available fonts on other systems.

## FairJob training

Review [data access](DATA_ACCESS.md), then run:

```bash
python3 scripts/run.py scripts/download_fairjob.py
python3 scripts/run.py -m fairjob_ranker prepare
python3 scripts/run.py -m fairjob_ranker train-baseline
python3 scripts/run.py -m fairjob_ranker train-pytorch
```

This creates the local splits, feature dictionaries and model artifacts that are intentionally absent from GitHub. Existing training outputs are protected against accidental overwrites; use a fresh output directory for a new run where supported. Full command flags are available through `--help`.

## EB-NeRD response experiment

Register with the publisher first. Obtain the official small and large bundles independently of this repository.

```bash
python3 scripts/run.py scripts/download_ebnerd.py --registered
mkdir -p data/processed/ebnerd-top1-feedback models/ebnerd-top1-feedback
mkdir -p reports/ebnerd-metadata-audit
mv reports/ebnerd-top1-feedback reports/published-ebnerd-top1-feedback
mkdir -p reports/ebnerd-top1-feedback
cp reports/published-ebnerd-top1-feedback/protocol.json reports/ebnerd-top1-feedback/protocol.json
python3 scripts/run.py scripts/prepare_metadata_audit.py
python3 scripts/run.py scripts/prepare_exposure_audit.py --delay-seconds 3600
python3 scripts/run.py scripts/prepare_feedback.py
python3 scripts/run.py scripts/train_feedback.py
python3 scripts/run.py scripts/prepare_fresh_users.py
python3 scripts/run.py scripts/score_fresh_users.py
python3 scripts/run.py scripts/evaluate_feedback.py
python3 scripts/run.py scripts/verify_feedback.py
```

The large archive must already be available under `data/raw/ebnerd-large/ebnerd_large.zip` before the fresh-user preparation step. Preparation retains full candidate sets. Scoring freezes predictions before label evaluation. Training labels and aggregate feature windows have distinct cutoffs.

Run this sequence in a fresh working copy. The `mv` command preserves the published aggregates before creating new run outputs, while `cp` retains the historical protocol. This is a reproduction of an existing experiment, not a newly preregistered evaluation. `prepare_metadata_audit.py` builds the article-free reference cohort and fixed-history features; it does not open the article table. Do not reuse the evaluated cohort to select new hyperparameters. Individual scripts guard against duplicate runs or resume only the outputs that they explicitly recognize.

The detailed [experiment protocol](../reports/ebnerd-top1-feedback/protocol.json) records selection rules, feature availability assumptions, evaluation boundaries and the success gate. The [research note](RESEARCH_NOTE.md) defines model-cold and zero-signal targets.

## Local serving

Serving commands require locally generated model bundles and a Redis installation. They are not a dataset-free demo.

```bash
python3 scripts/setup_local_redis.py
python3 scripts/run.py scripts/serve_local.py
```

The FairJob service exposes `/rank` and `/docs` on localhost. `compose.yaml` provides the original CPU API/Redis setup; `Dockerfile` describes the container environment. Model directories are mounted read-only.

The earlier EB-NeRD snapshot path uses `prepare_ebnerd.py`, `train_ebnerd.py`, `register_ebnerd.py` and `serve_ebnerd.py`. See the script arguments and configuration under `configs/`. Its recorded API report is `reports/ebnerd-serving/verification.json`.

The newest response model is not wired into that API. Do not interpret the snapshot API's latency as a measurement of the response model. A live receipt-time feature adapter, parity check and new load test are pending.

## Publication checks

Before release, the repository's synthetic test suite was rerun locally: **131 passed**, with two dependency deprecation/type-hint warnings. Figures were regenerated and checked against the frozen aggregate values. Public packaging excludes raw/derived user records, model bundles, local service state, neighboring projects and credentials.
