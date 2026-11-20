"""Preserve raw rows, quarantine ambiguous impressions, split whole users."""
from datetime import datetime, timezone
import hashlib
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .io import sha256, write_json
from .schema import AD_FEATURES, BINARY, CATEGORICAL, COLUMNS, CONTEXT, IDS, SPLITS, VERSION

RAW_SHA256 = "defeacb5233e26edd4679e2097d3582bc7858620ae301528675f78131a35c18c"


def validate_source(frame: pd.DataFrame) -> None:
    if set(frame.columns) != set(COLUMNS) or len(frame.columns) != len(COLUMNS):
        raise ValueError("Source must have exactly the 56 audited columns.")
    if frame.empty or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Source must be nonempty, without null or nonfinite values.")
    for column in IDS + CATEGORICAL + ["rank"]:
        values = frame[column].to_numpy()
        if not np.equal(values, np.floor(values)).all() or (values < 0).any():
            raise ValueError(f"Nonnegative integer required: {column}")
    if frame["rank"].lt(1).any():
        raise ValueError("Logged rank must be >= 1.")
    for column in BINARY:
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"Nonbinary values: {column}")
    grouping = frame.groupby("impression_id", sort=False)
    if grouping[["user_id", "displayrandom"] + CONTEXT].nunique().gt(1).any().any():
        raise ValueError("User, randomization flag and context must be constant per impression.")
    if frame.groupby("product_id", sort=False)[AD_FEATURES].nunique().gt(1).any().any():
        raise ValueError("Ad catalog features must be constant per product_id.")


def clean_impressions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop whole ambiguous groups, not arbitrarily one protected-attribute row."""
    bad_product = set(frame.loc[frame.duplicated(["impression_id", "product_id"], keep=False), "impression_id"])
    bad_rank = set(frame.loc[frame.duplicated(["impression_id", "rank"], keep=False), "impression_id"])
    excluded = pd.DataFrame({"impression_id": sorted(bad_product | bad_rank)})
    excluded["repeated_product"] = excluded.impression_id.isin(bad_product)
    excluded["repeated_rank"] = excluded.impression_id.isin(bad_rank)
    clean = frame.loc[~frame.impression_id.isin(excluded.impression_id)].copy()
    return clean, excluded


def split_for_user(user_id: int) -> str:
    digest = hashlib.sha256(f"{VERSION}:{int(user_id)}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def split_users(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    assignments = pd.DataFrame({"user_id": sorted(frame.user_id.unique())})
    assignments["split"] = assignments.user_id.map(split_for_user)
    labels = frame.user_id.map(assignments.set_index("user_id")["split"])
    splits = {
        name: frame.loc[labels.eq(name)].sort_values(["impression_id", "product_id"]).reset_index(drop=True)
        for name in SPLITS
    }
    for a, b in combinations(SPLITS, 2):
        for key in ["user_id", "impression_id"]:
            if set(splits[a][key]) & set(splits[b][key]):
                raise ValueError(f"Split leakage for {key}: {a}/{b}")
    if sum(len(part) for part in splits.values()) != len(frame):
        raise ValueError("Split lost rows.")
    return splits, assignments


def summarize(frame: pd.DataFrame) -> dict:
    groups = frame.groupby("impression_id").agg(
        n=("click", "size"), clicks=("click", "sum"), randomized=("displayrandom", "first")
    )
    eligible = groups.n.ge(2) & groups.clicks.gt(0) & groups.clicks.lt(groups.n)
    return {
        "rows": len(frame), "users": int(frame.user_id.nunique()),
        "impressions": len(groups), "products": int(frame.product_id.nunique()),
        "clicks": int(frame.click.sum()), "ctr": float(frame.click.mean()) if len(frame) else None,
        "rankable_impressions": int(eligible.sum()),
        "randomized_rankable_impressions": int((eligible & groups.randomized.eq(1)).sum()),
    }


def prepare(raw_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; use another --output for a new version.")
    if sha256(raw_path) != RAW_SHA256:
        raise ValueError("Raw SHA-256 differs from the pinned official FairJob release.")
    print("1/4 Reading and validating the full source...", flush=True)
    frame = pd.read_csv(raw_path)
    validate_source(frame)
    print("2/4 Quarantining whole ambiguous impressions...", flush=True)
    clean, excluded = clean_impressions(frame)
    for key in [["impression_id", "product_id"], ["impression_id", "rank"]]:
        if clean.duplicated(key).any():
            raise ValueError(f"Duplicate clean key: {key}")
    print("3/4 Assigning whole users to train/validation/test...", flush=True)
    splits, assignments = split_users(clean)
    catalog = clean[["product_id"] + AD_FEATURES].drop_duplicates().sort_values("product_id")
    if catalog.product_id.duplicated().any():
        raise ValueError("Catalog contains conflicting ad features.")
    manifest = {
        "version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": RAW_SHA256, "source_rows": len(frame),
        "source_url": "https://huggingface.co/datasets/criteo/FairJob",
        "license": "CC-BY-NC-SA-4.0",
        "policy": "Exclude complete impressions with repeated product_id or rank; keep all other exposures.",
        "excluded_impressions": len(excluded), "excluded_rows": len(frame) - len(clean),
        "split_policy": "SHA256('fairjob-v1:' + decimal user_id), first 8 bytes big-endian mod 100: <80 train, <90 validation, else test",
        "split_intersections": {"users": 0, "impressions": 0},
        "clean": summarize(clean), "splits": {name: summarize(part) for name, part in splits.items()},
        "catalog": {"products": len(catalog), "features": AD_FEATURES, "scope": "All clean static ad metadata, no labels; not used to fit transforms."},
        "code_sha256": sha256(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete directory; failed runs do not leave half a dataset.
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        for name, part in splits.items():
            part.to_parquet(staged / f"{name}.parquet", index=False, compression="zstd")
        excluded.to_parquet(staged / "excluded_impressions.parquet", index=False)
        assignments.to_parquet(staged / "user_splits.parquet", index=False)
        catalog.to_parquet(staged / "ad_catalog.parquet", index=False, compression="zstd")
        manifest["files"] = {
            path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(staged.glob("*.parquet"))
        }
        write_json(staged / "manifest.json", manifest)
        staged.rename(output)
    print(f"4/4 Saved {output}", flush=True)
    return manifest

