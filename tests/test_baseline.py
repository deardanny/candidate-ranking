import json

import numpy as np
import pandas as pd

from fairjob_ranker.baseline import train_baseline
from fairjob_ranker.io import sha256, write_json
from fairjob_ranker.schema import CATEGORICAL, NUMERICAL, VERSION


def test_training_pipeline_holds_out_test_and_reloads_artifacts(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    splits = {}
    for name, offset in [("train", 0), ("validation", 1000)]:
        n = 100
        y = np.tile([0, 1], n//2)
        frame = pd.DataFrame({**{c: y for c in CATEGORICAL}, **{c: np.ones(n) for c in NUMERICAL}})
        frame["user_id"] = np.repeat(np.arange(n//2) + offset, 2)
        frame["impression_id"] = frame.user_id
        frame["product_id"] = y + 1
        frame["rank"] = y + 1
        frame["click"] = y
        frame["displayrandom"] = 1
        frame.to_parquet(data / f"{name}.parquet", index=False)
        splits[name] = {"rows": len(frame)}
    # This intentionally invalid Parquet file makes any accidental test read fail.
    (data / "test.parquet").write_bytes(b"TEST MUST NOT BE OPENED BY BASELINE TRAINING")
    manifest = {
        "version": VERSION, "source_sha256": "synthetic-test-fixture", "splits": splits,
        "files": {p.name: {"sha256": sha256(p)} for p in data.glob("*.parquet")},
    }
    write_json(data / "manifest.json", manifest)
    config = tmp_path / "config.json"
    write_json(config, {
        "num_boost_round": 5, "early_stopping_rounds": 2, "bootstrap_repeats": 20,
        "params": {"objective": "binary", "metric": "binary_logloss", "num_threads": 1,
                   "verbosity": -1, "min_data_in_leaf": 5, "min_data_per_group": 1,
                   "num_leaves": 3, "seed": 42},
    })
    result = train_baseline(data, config, tmp_path / "model", tmp_path / "report")
    assert result["test_evaluated"] is False
    assert result["evaluation_split"] == "validation"
    assert result["reload_max_abs_error"] < 1e-12
    assert result["models"]["lightgbm"]["all"]["click"]["rows"] == 100
    assert result["models"]["lightgbm"]["all"]["ranking"]["eligible_impressions"] == 50
    saved = json.loads((tmp_path / "model/features.json").read_text())
    assert "click" not in saved["feature_order"] and "rank" not in saved["feature_order"]
    assert (tmp_path / "model/model.txt").is_file()
    assert (tmp_path / "report/RESULTS.md").is_file()
