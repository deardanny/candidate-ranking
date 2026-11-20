"""Declare immutable artifacts before evaluating the held-out test split."""
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import numpy as np
import pandas as pd

from .evaluation import evaluate, paired_user_bootstrap
from .features import FeatureTransformer
from .io import read_json, sha256, write_json
from .replay import make_request
from .schema import FEATURES, VERSION
from .torch_model import load_bundle


def freeze_experiment(data_dir: Path, model_dir: Path, baseline_dir: Path, settings_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    bundle = load_bundle(model_dir)
    baseline = read_json(baseline_dir / "metadata.json")
    digest = sha256(data_dir / "manifest.json")
    if bundle.metadata["dataset_manifest_sha256"] != digest or baseline["dataset_manifest_sha256"] != digest:
        raise ValueError("Models must share the frozen dataset manifest.")
    files = {}
    for directory, names in [(model_dir, ["weights.pt", "features.json", "ad_catalog.parquet", "fallback.json", "metadata.json"]),
                             (baseline_dir, ["model.txt", "features.json", "metadata.json"])]:
        for name in names:
            path = directory / name
            files[str(path)] = sha256(path)
    for name in ["model.txt", "features.json"]:
        if files[str(baseline_dir / name)] != baseline["artifact_sha256"][name]:
            raise ValueError("Baseline artifact was modified.")
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "schema_version": VERSION,
              "dataset_manifest_sha256": digest, "data_dir": str(data_dir),
              "pytorch_dir": str(model_dir), "lightgbm_dir": str(baseline_dir), "artifact_sha256": files,
              "serving_settings_path": str(settings_path), "serving_settings_sha256": sha256(settings_path),
              "serving_settings": read_json(settings_path), "test_used_for_selection": False,
              "demo_serving_model": "pytorch", "validation_quality_winner": "lightgbm",
              "decision": "Keep the PyTorch implementation as the fixed engineering demo. LightGBM remains the stronger validation quality reference. Evaluate both without further tuning.",
              "test_policy": "One frozen comparison; do not tune data, model or serving settings on test results.",
              "source_code_sha256": {str(p): sha256(p) for p in sorted(Path('src').rglob('*.py'))}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def verify_frozen(path: Path) -> dict:
    frozen = read_json(path)
    for name, expected in frozen["artifact_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    if sha256(Path(frozen["serving_settings_path"])) != frozen["serving_settings_sha256"]:
        raise ValueError("Frozen serving settings changed.")
    if sha256(Path(frozen["data_dir"]) / "manifest.json") != frozen["dataset_manifest_sha256"]:
        raise ValueError("Frozen data manifest changed.")
    return frozen


def evaluate_frozen(frozen_path: Path, output: Path, bootstrap_repeats: int = 1000) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final evaluation {output}")
    frozen = verify_frozen(frozen_path)
    data_dir = Path(frozen["data_dir"])
    manifest = read_json(data_dir / "manifest.json")
    for name in ["train.parquet", "test.parquet"]:
        if sha256(data_dir / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    test = pd.read_parquet(data_dir / "test.parquet")
    training_products = set(pd.read_parquet(data_dir / "train.parquet", columns=["product_id"]).product_id)
    bundle = load_bundle(Path(frozen["pytorch_dir"]))
    baseline_dir = Path(frozen["lightgbm_dir"])
    baseline_transformer = FeatureTransformer.load(baseline_dir / "features.json")
    x = baseline_transformer.transform(test)
    baseline = lgb.Booster(model_file=str(baseline_dir / "model.txt"))
    predictions = {"train_ctr_constant": np.full(len(test), bundle.metadata["train_ctr"]),
                   "lightgbm": baseline.predict(x, num_threads=1), "pytorch": bundle.predict(test)}
    models, groups = {}, {}
    for name, scores in predictions.items():
        models[name], groups[name] = evaluate(test, scores, training_products)
    comparisons = {}
    for first, second in [("lightgbm", "train_ctr_constant"), ("pytorch", "train_ctr_constant"), ("pytorch", "lightgbm")]:
        comparison = paired_user_bootstrap(groups[first], groups[second], repeats=bootstrap_repeats)
        comparison["difference"] = f"{first} - {second}"
        comparisons[f"{first}_vs_{second}"] = comparison
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_split": "test",
              "test_evaluated": True, "frozen_experiment_sha256": sha256(frozen_path),
              "frozen_experiment": frozen, "models": models, "paired_bootstrap": comparisons,
              "scope": "Released observed candidates; no online CTR, causal or future-time performance claim.",
              "model_version": bundle.metadata["model_version"], "test_rows": len(test)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".final-", dir=output.parent) as temporary:
        staged = Path(temporary) / "evaluation"
        staged.mkdir()
        write_json(staged / "metrics.json", result)
        test[["impression_id", "user_id", "product_id", "click", "displayrandom"]].assign(
            **{f"{name}_score": scores for name, scores in predictions.items()}
        ).to_parquet(staged / "test_predictions.parquet", index=False)
        # Deterministic first eligible group, not selected for a favorable prediction.
        impression = int(groups["pytorch"].impression_id.min())
        selected = test.loc[test.impression_id.eq(impression)]
        example = {"selection_rule": "Smallest eligible test impression_id", "request": make_request(selected),
                   "rows": selected[["product_id", "rank", "click"]].assign(
                       **{f"{name}_score": np.asarray(scores)[test.impression_id.eq(impression)]
                          for name, scores in predictions.items()}).to_dict("records")}
        write_json(staged / "demo_example.json", example)
        lines = ["# 최종 고정 모델의 test 평가", "", "모델·설정을 먼저 고정한 뒤 test를 평가했다. 이후 이 결과로 모델을 다시 선택하거나 학습하지 않는다.", "",
                 "| 집단 | 모델 | LogLoss ↓ | AP ↑ | ROC-AUC ↑ | NDCG@5 ↑ | MRR ↑ | 평가 노출 |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for cohort in ["all", "randomized"]:
            for name, metrics in models.items():
                c, r = metrics[cohort]["click"], metrics[cohort]["ranking"]
                values = [c["logloss"], c["average_precision"], c["roc_auc"], r["ndcg_at_5"], r["mrr"]]
                lines.append(f"| {cohort} | {name} | " + " | ".join("N/A" if v is None else f"{v:.6f}" for v in values) + f" | {r['eligible_impressions']} |")
        lines += ["", "## 사용자 단위 paired bootstrap", ""]
        for pair, comparison in comparisons.items():
            for cohort in ["all", "randomized"]:
                item = comparison[cohort]
                lines.append(f"- {pair}, {cohort}: {item['impressions']}개 노출 / {item['users']}명 사용자")
                for metric in ["ndcg_at_5", "mrr"]:
                    v = item[metric]
                    if v:
                        lines.append(f"  - {metric}: {v['mean_difference']:.6f}, 95% CI [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]")
        lines += ["", "신규/기존 광고, calibration, Brier와 평가 coverage는 metrics.json에 있다.",
                  "공개된 후보의 관측 클릭에 대한 평가다. 온라인 CTR 상승·전체 경매·미래 시점 성능을 입증하지 않는다.", ""]
        (staged / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
        staged.rename(output)
    return result
