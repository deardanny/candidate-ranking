"""Train the fixed LightGBM baseline and report validation only."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from .evaluation import evaluate, paired_user_bootstrap
from .features import FeatureTransformer
from .io import read_json, sha256, write_json
from .schema import CATEGORICAL, FEATURES, VERSION


def load_split(data_dir: Path, name: str, manifest: dict) -> pd.DataFrame:
    path = data_dir / f"{name}.parquet"
    if sha256(path) != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"Processed file checksum mismatch: {path}")
    return pd.read_parquet(path)


def write_report(path: Path, results: dict) -> None:
    lines = [
        "# 1주차 LightGBM validation 결과", "",
        "Test는 아직 모델 평가에 사용하지 않았다. 아래 값은 validation 결과이며 온라인 CTR 향상이 아니다.", "",
        f"- 선택된 boosting round: {results['best_iteration']} / 최대 {results['config']['num_boost_round']}",
        f"- 학습 시간: {results['training_seconds']:.2f}초 (이 실행 환경에서의 측정값)",
        f"- 학습 CTR 상수: {results['train_ctr']:.8f}",
        "- AP는 average precision이며 사다리꼴 방식 PR-AUC와 구분한다.", "",
        "| 평가 집단 | 모델 | LogLoss ↓ | AP ↑ | ROC-AUC ↑ | Brier ↓ | NDCG@5 ↑ | MRR ↑ | 랭킹 평가 노출 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"
    for cohort in ["all", "randomized"]:
        for name, metrics in results["models"].items():
            c, r = metrics[cohort]["click"], metrics[cohort]["ranking"]
            values = [c["logloss"], c["average_precision"], c["roc_auc"], c["brier"], r["ndcg_at_5"], r["mrr"]]
            lines.append(f"| {cohort} | {name} | " + " | ".join(fmt(v) for v in values) + f" | {r['eligible_impressions']} |")
    lines += ["", "클릭 지표는 각 집단의 모든 행을 사용한다. 랭킹 지표는 후보 2개 이상이며 클릭/비클릭이 모두 있는 노출만 사용한다.",
              "상수 모델의 동점은 광고 ID 오름차순으로 처리한다. 기존 표시 위치는 어떤 모델에도 입력하지 않는다.", "",
              "## 사용자 단위 paired bootstrap", "",
              "각 값은 LightGBM − 상수 모델 차이와 95% 구간이다. Validation에서 모델을 선택했으므로 최종 일반화 검증을 대신하지 않는다.", ""]
    for cohort in ["all", "randomized"]:
        b = results["paired_bootstrap"][cohort]
        lines.append(f"- {cohort}: {b['impressions']}개 노출, {b['users']}명 사용자")
        for metric in ["ndcg_at_5", "mrr"]:
            item = b[metric]
            if item is not None:
                lo, hi = item["ci95"]
                lines.append(f"  - {metric}: {item['mean_difference']:.6f}, 95% CI [{lo:.6f}, {hi:.6f}]")
    lines += ["", "세부 calibration, 신규 광고 집단, 피처 OOV 비율과 실행 환경은 `metrics.json`에 있다.",
              "무작위 표시 집단도 이미 선택된 공개 후보에 대한 평가이며, 전체 경매나 실제 트래픽의 정책 가치를 측정하지 않는다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_baseline(data_dir: Path, config_path: Path, model_output: Path, report_output: Path) -> dict:
    if model_output.resolve() == report_output.resolve():
        raise ValueError("Model and report directories must differ.")
    for output in [model_output, report_output]:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory.")
    config = read_json(config_path)
    manifest = read_json(data_dir / "manifest.json")
    if manifest["version"] != VERSION:
        raise ValueError("Dataset version mismatch.")
    print("Reading train and validation only (test is held out)...", flush=True)
    train = load_split(data_dir, "train", manifest)
    validation = load_split(data_dir, "validation", manifest)
    for key in ["user_id", "impression_id"]:
        if set(train[key]) & set(validation[key]):
            raise ValueError(f"Train/validation overlap: {key}")
    transformer = FeatureTransformer.fit(train)
    x_train, x_valid = transformer.transform(train), transformer.transform(validation)
    train_ctr = float(train.click.mean())
    history = {}
    started = perf_counter()
    model = lgb.train(
        config["params"],
        lgb.Dataset(x_train, label=train.click, categorical_feature=CATEGORICAL),
        num_boost_round=config["num_boost_round"],
        valid_sets=[lgb.Dataset(x_valid, label=validation.click, categorical_feature=CATEGORICAL)],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(config["early_stopping_rounds"], first_metric_only=True),
                   lgb.record_evaluation(history), lgb.log_evaluation(50)],
    )
    training_seconds = perf_counter() - started
    prediction = model.predict(x_valid, num_iteration=model.best_iteration, num_threads=config["params"]["num_threads"])
    constant = np.full(len(validation), train_ctr)
    products = set(train.product_id)
    baseline_report, baseline_groups = evaluate(validation, constant, products)
    model_report, model_groups = evaluate(validation, prediction, products)
    bootstrap = paired_user_bootstrap(model_groups, baseline_groups, repeats=config["bootstrap_repeats"])
    project_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True)
    source_hashes = {str(p.relative_to(project_root)): sha256(p) for p in sorted((project_root / "src").rglob("*.py"))}
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_split": "validation",
        "test_evaluated": False, "config": config, "best_iteration": model.best_iteration,
        "training_seconds": training_seconds, "train_ctr": train_ctr,
        "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
        "source_sha256": manifest["source_sha256"], "split_version": VERSION,
        "split_sizes": manifest["splits"], "feature_order": FEATURES,
        "models": {"train_ctr_constant": baseline_report, "lightgbm": model_report},
        "paired_bootstrap": bootstrap,
        "validation_oov_rates": {c: float(x_valid[c].eq(0).mean()) for c in CATEGORICAL},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "packages": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow"]},
        },
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "source_code_sha256": source_hashes,
    }
    for output in [model_output, report_output]:
        output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".baseline-", dir=model_output.parent) as temporary:
        staged_model = Path(temporary) / "model"
        staged_model.mkdir()
        model.save_model(str(staged_model / "model.txt"), num_iteration=model.best_iteration)
        transformer.save(staged_model / "features.json")
        reloaded = lgb.Booster(model_file=str(staged_model / "model.txt"))
        reloaded_x = FeatureTransformer.load(staged_model / "features.json").transform(validation)
        np.testing.assert_array_equal(x_valid.to_numpy(), reloaded_x.to_numpy())
        reloaded_scores = reloaded.predict(reloaded_x, num_threads=config["params"]["num_threads"])
        np.testing.assert_allclose(prediction, reloaded_scores, atol=1e-12, rtol=1e-12)
        results["reload_max_abs_error"] = float(np.max(np.abs(prediction - reloaded_scores)))
        results["artifact_sha256"] = {name: sha256(staged_model / name) for name in ["model.txt", "features.json"]}
        write_json(staged_model / "metadata.json", results)
        write_json(staged_model / "training_history.json", history)
        validation[["user_id", "impression_id", "product_id", "click", "displayrandom"]].assign(
            lightgbm_score=prediction, constant_score=constant
        ).to_parquet(staged_model / "validation_predictions.parquet", index=False)
        # Complete the report before publishing either artifact directory.
        with TemporaryDirectory(prefix=".report-", dir=report_output.parent) as report_temporary:
            staged_report = Path(report_temporary) / "report"
            staged_report.mkdir()
            write_json(staged_report / "metrics.json", results)
            write_json(staged_report / "data_manifest.json", manifest)
            importance = pd.DataFrame({"feature": FEATURES,
                                       "gain": model.feature_importance(importance_type="gain"),
                                       "splits": model.feature_importance(importance_type="split")}).sort_values("gain", ascending=False)
            write_json(staged_report / "feature_importance.json", importance.to_dict("records"))
            eligible_ids = model_groups.sort_values("impression_id").impression_id.head(3)
            examples = validation.loc[validation.impression_id.isin(eligible_ids), ["impression_id", "product_id", "rank", "click"]].copy()
            examples["score"] = pd.Series(prediction, index=validation.index).loc[examples.index]
            examples = examples.sort_values(["impression_id", "score", "product_id"], ascending=[True, False, True])
            examples["new_position"] = examples.groupby("impression_id").cumcount() + 1
            write_json(staged_report / "ranking_examples.json", examples.to_dict("records"))
            write_report(staged_report / "RESULTS.md", results)
            staged_model.rename(model_output)
            staged_report.rename(report_output)
    print(f"Saved model: {model_output}; report: {report_output}", flush=True)
    print(f"Validation LogLoss: constant={baseline_report['all']['click']['logloss']:.6f}, LightGBM={model_report['all']['click']['logloss']:.6f}", flush=True)
    return results
