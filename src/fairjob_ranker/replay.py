"""Label-free HTTP request snapshots, preserving released candidate groups."""
import json
from pathlib import Path

import pandas as pd

from .io import read_json, sha256, write_json
from .schema import CONTEXT, NUMERICAL, VERSION


def make_request(frame: pd.DataFrame) -> dict:
    first = frame.iloc[0]
    return {"request_id": f"impression-{int(first.impression_id)}", "schema_version": VERSION,
            "context": {"cat0": int(first.cat0), "cat1": int(first.cat1),
                        "numeric": {c: float(first[c]) for c in NUMERICAL}},
            "candidate_ids": [int(i) for i in frame.product_id], "top_k": len(frame)}


def export_requests(data_dir: Path, split: str, output: Path, frozen_path: Path | None = None) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("Replay uses validation or test only.")
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = read_json(data_dir / "manifest.json")
    if split == "test":
        if frozen_path is None:
            raise ValueError("Test replay requires a predeclared frozen experiment.")
        frozen = read_json(frozen_path)
        if frozen["dataset_manifest_sha256"] != sha256(data_dir / "manifest.json"):
            raise ValueError("Frozen dataset mismatch.")
    source = data_dir / f"{split}.parquet"
    if sha256(source) != manifest["files"][source.name]["sha256"]:
        raise ValueError("Replay source checksum mismatch.")
    frame = pd.read_parquet(source, columns=["impression_id", "product_id", *CONTEXT])
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = []
    with output.open("x") as stream:
        for _, group in frame.groupby("impression_id", sort=True):
            if group[CONTEXT].nunique().gt(1).any() or group.product_id.duplicated().any():
                raise ValueError("Invalid candidate group.")
            counts.append(len(group))
            stream.write(json.dumps(make_request(group), separators=(",", ":"), allow_nan=False) + "\n")
    result = {"schema_version": VERSION, "split": split, "source_sha256": sha256(source),
              "requests": len(counts), "candidates": sum(counts), "max_candidates": max(counts),
              "labels_included": False, "request_file_sha256": sha256(output),
              "frozen_experiment_sha256": sha256(frozen_path) if frozen_path else None}
    write_json(output.with_suffix(".manifest.json"), result)
    return result


def load_requests(path: Path) -> tuple[list[dict], dict]:
    manifest = read_json(path.with_suffix(".manifest.json"))
    if sha256(path) != manifest["request_file_sha256"]:
        raise ValueError("Replay corpus checksum mismatch.")
    requests = [json.loads(line) for line in path.read_text().splitlines()]
    if len(requests) != manifest["requests"] or not requests:
        raise ValueError("Replay corpus size mismatch.")
    return requests, manifest
