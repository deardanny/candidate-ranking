"""Fit embedding + MLP on train; select and compare models on validation only."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import random
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

from .evaluation import click_metrics, evaluate, paired_user_bootstrap
from .features import FeatureTransformer
from .io import read_json, sha256, write_json
from .schema import CATEGORICAL, FEATURES, VERSION
from .torch_model import EmbeddingMLP, load_bundle, predict_tensors, to_tensors


def checked_split(data_dir: Path, name: str, manifest: dict) -> pd.DataFrame:
    path = data_dir / f"{name}.parquet"
    if sha256(path) != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {path.name}")
    return pd.read_parquet(path)


def train_pytorch(data_dir: Path, config_path: Path, model_output: Path, report_output: Path,
                  baseline_dir: Path | None = None) -> dict:
    if model_output.resolve() == report_output.resolve():
        raise ValueError("Model and report directories must differ.")
    for output in [model_output, report_output]:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory.")
    config, manifest = read_json(config_path), read_json(data_dir / "manifest.json")
    if manifest["version"] != VERSION:
        raise ValueError("Dataset version mismatch.")
    torch.set_num_threads(config["num_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    print("Reading train and validation; test remains held out.", flush=True)
    train = checked_split(data_dir, "train", manifest)
    valid = checked_split(data_dir, "validation", manifest)
    for key in ["user_id", "impression_id"]:
        if set(train[key]) & set(valid[key]):
            raise ValueError(f"Train/validation overlap: {key}")
    transformer = FeatureTransformer.fit(train)
    x_train, x_valid = transformer.transform(train), transformer.transform(valid)
    ct, nt = to_tensors(x_train)
    cv, nv = to_tensors(x_valid)
    labels = torch.from_numpy(train.click.to_numpy(dtype=np.float32, copy=True))
    train_ctr = float(train.click.mean())
    if not 0 < train_ctr < 1:
        raise ValueError("Both label classes are required in train.")
    architecture = {
        "category_sizes": [len(transformer.vocabularies[c]) + 1 for c in CATEGORICAL],
        "embedding_dim": config["embedding_dim"], "hidden_dims": config["hidden_dims"],
        "dropout": config["dropout"],
    }
    model = EmbeddingMLP(**architecture)
    # Start near train CTR rather than 50% on this very imbalanced dataset.
    nn.init.normal_(model.network[-1].weight, std=0.01)
    nn.init.constant_(model.network[-1].bias, np.log(train_ctr / (1 - train_ctr)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    history, best_state, best_epoch = [], None, 0
    best_loss, stale = float("inf"), 0
    started = perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        epoch_start = perf_counter()
        model.train()
        order, total = torch.randperm(len(labels)), 0.0
        for start in range(0, len(order), config["batch_size"]):
            ids = order[start:start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ct[ids], nt[ids]), labels[ids])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        scores = predict_tensors(model, cv, nv)
        valid_loss = click_metrics(valid.click.to_numpy(), scores)["logloss"]
        history.append({"epoch": epoch, "train_logloss": total / len(train),
                        "validation_logloss": valid_loss, "seconds": perf_counter() - epoch_start})
        print(f"Epoch {epoch}: train={total / len(train):.6f}, validation={valid_loss:.6f}", flush=True)
        if valid_loss < best_loss:
            best_loss, best_epoch, stale = valid_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    if best_state is None:
        raise ValueError("No trained model was selected.")
    training_seconds = perf_counter() - started
    model.load_state_dict(best_state)
    scores = predict_tensors(model, cv, nv)
    products = set(train.product_id)
    model_metrics, model_groups = evaluate(valid, scores, products)
    constant_metrics, constant_groups = evaluate(valid, np.full(len(valid), train_ctr), products)
    bootstrap = paired_user_bootstrap(model_groups, constant_groups, repeats=config["bootstrap_repeats"])
    bootstrap["difference"] = "pytorch - train_ctr_constant"
    models = {"train_ctr_constant": constant_metrics, "pytorch": model_metrics}
    comparisons = {"vs_constant": bootstrap}
    if baseline_dir is not None:
        # Re-score the verified Week 1 model on exactly the same validation rows.
        import lightgbm as lgb
        baseline_meta = read_json(baseline_dir / "metadata.json")
        if baseline_meta["dataset_manifest_sha256"] != sha256(data_dir / "manifest.json"):
            raise ValueError("Baseline was trained with a different dataset manifest.")
        for name in ["model.txt", "features.json"]:
            if sha256(baseline_dir / name) != baseline_meta["artifact_sha256"][name]:
                raise ValueError("Baseline bundle checksum mismatch.")
        baseline_x = FeatureTransformer.load(baseline_dir / "features.json").transform(valid)
        baseline_scores = lgb.Booster(model_file=str(baseline_dir / "model.txt")).predict(baseline_x, num_threads=config["num_threads"])
        models["lightgbm"], baseline_groups = evaluate(valid, baseline_scores, products)
        comparison = paired_user_bootstrap(model_groups, baseline_groups, repeats=config["bootstrap_repeats"])
        comparison["difference"] = "pytorch - lightgbm"
        comparisons["vs_lightgbm"] = comparison
    counts = train.groupby("product_id").click.agg(["sum", "size"])
    alpha = config["fallback_alpha"]
    fallback = {"fitted_on": "train", "alpha": alpha, "train_ctr": train_ctr,
                "scores": {str(int(i)): float((r["sum"] + alpha * train_ctr) / (r["size"] + alpha))
                           for i, r in counts.iterrows()}}
    root = Path(__file__).resolve().parents[2]
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "schema_version": VERSION,
        "feature_order": FEATURES, "architecture": architecture, "config": config,
        "evaluation_split": "validation", "test_evaluated": False, "best_epoch": best_epoch,
        "training_seconds": training_seconds, "train_ctr": train_ctr, "models": models,
        "paired_bootstrap": comparisons, "training_history": history,
        "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
        "source_sha256": manifest["source_sha256"], "split_sizes": manifest["splits"],
        "validation_oov_rates": {c: float(x_valid[c].eq(0).mean()) for c in CATEGORICAL},
        "source_code_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").rglob("*.py"))},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {n: version(n) for n in ["torch", "numpy", "pandas", "lightgbm", "pyarrow"]}},
    }
    for output in [model_output, report_output]:
        output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".torch-", dir=model_output.parent) as temp:
        staged = Path(temp) / "model"
        staged.mkdir()
        torch.save(model.state_dict(), staged / "weights.pt")
        transformer.save(staged / "features.json")
        catalog_path = data_dir / "ad_catalog.parquet"
        if sha256(catalog_path) != manifest["files"][catalog_path.name]["sha256"]:
            raise ValueError("Catalog checksum mismatch.")
        shutil.copyfile(catalog_path, staged / "ad_catalog.parquet")
        write_json(staged / "fallback.json", fallback)
        result["artifact_sha256"] = {n: sha256(staged / n) for n in ["weights.pt", "features.json", "ad_catalog.parquet", "fallback.json"]}
        result["model_version"] = config["run_name"] + ":" + result["artifact_sha256"]["weights.pt"][:12]
        result["feature_version"] = result["artifact_sha256"]["features.json"]
        result["catalog_version"] = result["artifact_sha256"]["ad_catalog.parquet"]
        write_json(staged / "metadata.json", result)
        reloaded = load_bundle(staged)
        np.testing.assert_array_equal(x_valid.to_numpy(), reloaded.transformer.transform(valid).to_numpy())
        reloaded_scores = reloaded.predict(valid)
        np.testing.assert_allclose(scores, reloaded_scores, atol=1e-6, rtol=1e-5)
        result["reload_max_abs_error"] = float(np.max(np.abs(scores - reloaded_scores)))
        write_json(staged / "metadata.json", result)
        valid[["user_id", "impression_id", "product_id", "click", "displayrandom"]].assign(
            pytorch_score=scores).to_parquet(staged / "validation_predictions.parquet", index=False)
        with TemporaryDirectory(prefix=".report-", dir=report_output.parent) as report_temp:
            report = Path(report_temp) / "report"
            report.mkdir()
            write_json(report / "metrics.json", result)
            lines = ["# 2주차 모델 validation 결과", "", "Test 미평가. 공개된 후보에 대한 오프라인 평가이며 온라인 CTR 향상이 아니다.", "",
                     "| 모델 | LogLoss ↓ | AP ↑ | ROC-AUC ↑ | NDCG@5 ↑ | MRR ↑ |", "|---|---:|---:|---:|---:|---:|"]
            for name, metrics in models.items():
                c, r = metrics["all"]["click"], metrics["all"]["ranking"]
                values = [c["logloss"], c["average_precision"], c["roc_auc"], r["ndcg_at_5"], r["mrr"]]
                lines.append(f"| {name} | " + " | ".join("N/A" if v is None else f"{v:.6f}" for v in values) + " |")
            lines += ["", f"선택 epoch: {best_epoch}; 학습 시간: {training_seconds:.2f}초; CPU threads: {config['num_threads']}.",
                      f"클릭 평가 {len(valid):,}행; 랭킹 평가 {model_metrics['all']['ranking']['eligible_impressions']:,}개 노출.",
                      "무작위 표시 하위집단, 신규 광고, 보정 상태, bootstrap 구간은 metrics.json에 포함했다.", ""]
            (report / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
            staged.rename(model_output)
            report.rename(report_output)
    print(f"Saved {model_output}; best epoch={best_epoch}, validation LogLoss={best_loss:.6f}", flush=True)
    return result
