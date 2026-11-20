"""Audit the complete file; write aggregates and a few public example groups."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def distribution(values: pd.Series) -> dict:
    return {
        "count": int(len(values)), "min": float(values.min()),
        "mean": float(values.mean()), "p50": float(values.quantile(.5)),
        "p90": float(values.quantile(.9)), "p95": float(values.quantile(.95)),
        "p99": float(values.quantile(.99)), "max": float(values.max()),
    }


def main() -> None:
    print("Reading the complete gzip CSV...", flush=True)
    data = pd.read_csv(ROOT / "data/raw/fairjob.csv.gz")
    print(f"Loaded {data.shape}; auditing impression groups...", flush=True)
    cat = [c for c in data if c.startswith("cat")]
    num = [c for c in data if c.startswith("num")]
    grouped = data.groupby("impression_id", sort=False)
    groups = grouped.agg(
        n_rows=("click", "size"), n_ads=("product_id", "nunique"),
        clicks=("click", "sum"), user_id=("user_id", "first"),
        n_users=("user_id", "nunique"), randomized=("displayrandom", "first"),
        n_random_values=("displayrandom", "nunique"), n_positions=("rank", "nunique"),
        min_position=("rank", "min"), max_position=("rank", "max"),
    )
    groups["rankable"] = (groups.n_ads >= 2) & (groups.clicks > 0) & (groups.clicks < groups.n_rows)
    groups["rank_span"] = groups.max_position - groups.min_position + 1
    group_counts = {
        "total": len(groups), "single_candidate": int((groups.n_ads == 1).sum()),
        "zero_click": int((groups.clicks == 0).sum()),
        "one_click": int((groups.clicks == 1).sum()),
        "multiple_clicks": int((groups.clicks > 1).sum()),
        "with_positive_and_negative_candidates": int(groups.rankable.sum()),
        "randomized": int((groups.randomized == 1).sum()),
        "randomized_rankable": int(((groups.randomized == 1) & groups.rankable).sum()),
        "multiple_users": int((groups.n_users != 1).sum()),
        "mixed_randomization_flags": int((groups.n_random_values != 1).sum()),
        "duplicate_positions": int((groups.n_positions != groups.n_rows).sum()),
        "rank_gaps": int((groups.rank_span != groups.n_positions).sum()),
        "not_starting_at_one": int((groups.min_position != 1).sum()),
    }
    # A request feature must be constant within an impression. Record evidence,
    # without assigning business semantics to anonymous numeric columns.
    variation = {}
    for column in cat + num + ["protected_attribute", "senior"]:
        variation[column] = int(grouped[column].nunique(dropna=False).gt(1).sum())
    product_variation = {
        c: int(data.groupby("product_id", sort=False)[c].nunique(dropna=False).gt(1).sum())
        for c in cat + ["senior"]
    }
    user_variation = {
        c: int(data.groupby("user_id", sort=False)[c].nunique(dropna=False).gt(1).sum())
        for c in ["protected_attribute", "cat0", "cat1"] + num
    }
    duplicate_products = data.loc[data.duplicated(["impression_id", "product_id"], keep=False)]
    duplicate_positions = data.loc[data.duplicated(["impression_id", "rank"], keep=False)]
    duplicate_product_differences = duplicate_products.groupby(["impression_id", "product_id"]).nunique(dropna=False)
    ambiguous_ids = set(duplicate_products.impression_id) | set(duplicate_positions.impression_id)
    clean_mask = ~data.impression_id.isin(ambiguous_ids)
    clean_group_mask = ~groups.index.isin(ambiguous_ids)

    user_ids = data.user_id.unique()
    buckets = {
        int(uid): int.from_bytes(hashlib.sha256(f"fairjob-v1:{uid}".encode()).digest()[:8], "big") % 100
        for uid in user_ids
    }
    user_split = {uid: "train" if b < 80 else "validation" if b < 90 else "test" for uid, b in buckets.items()}
    group_split = groups.user_id.map(user_split)
    row_split = data.user_id.map(user_split)
    split_profile = {}
    training_products = set(data.loc[row_split == "train", "product_id"])
    clean_training_products = set(data.loc[(row_split == "train") & clean_mask, "product_id"])
    clean_split_profile = {}
    for split in ["train", "validation", "test"]:
        frame = data.loc[row_split == split]
        gs = groups.loc[group_split == split]
        split_profile[split] = {
            "rows": len(frame), "users": int(frame.user_id.nunique()), "impressions": len(gs),
            "clicks": int(frame.click.sum()), "ctr": float(frame.click.mean()),
            "rankable_impressions": int(gs.rankable.sum()),
            "randomized_impressions": int(gs.randomized.eq(1).sum()),
            "randomized_rows": int(frame.displayrandom.eq(1).sum()),
            "randomized_clicks": int(frame.loc[frame.displayrandom.eq(1), "click"].sum()),
            "randomized_rankable_impressions": int((gs.randomized.eq(1) & gs.rankable).sum()),
            "unseen_product_rows_vs_train": int((~frame.product_id.isin(training_products)).sum()),
        }
        clean_frame = data.loc[(row_split == split) & clean_mask]
        clean_gs = groups.loc[(group_split == split) & clean_group_mask]
        clean_split_profile[split] = {
            "rows": len(clean_frame), "users": int(clean_frame.user_id.nunique()), "impressions": len(clean_gs),
            "clicks": int(clean_frame.click.sum()), "ctr": float(clean_frame.click.mean()),
            "rankable_impressions": int(clean_gs.rankable.sum()),
            "randomized_rows": int(clean_frame.displayrandom.eq(1).sum()),
            "randomized_clicks": int(clean_frame.loc[clean_frame.displayrandom.eq(1), "click"].sum()),
            "randomized_rankable_impressions": int((clean_gs.randomized.eq(1) & clean_gs.rankable).sum()),
            "unseen_product_rows_vs_train": int((~clean_frame.product_id.isin(clean_training_products)).sum()),
        }

    clean = data.loc[clean_mask]
    clean_groups = groups.loc[clean_group_mask]
    key_diagnostics = {
        "duplicate_impression_product_groups": len(duplicate_product_differences),
        "affected_impressions": len(ambiguous_ids),
        "conflicting_click_groups": int(duplicate_product_differences.click.gt(1).sum()),
        "differing_columns": {k: int(v) for k,v in duplicate_product_differences.gt(1).sum().items() if v},
        "strict_view_policy": "Exclude whole impression groups with repeated product_id or repeated rank; preserve raw data. This is a proposed view, not a corrected source dataset.",
        "strict_view": {
            "rows": len(clean), "impressions": len(clean_groups), "users": int(clean.user_id.nunique()),
            "products": int(clean.product_id.nunique()), "clicks": int(clean.click.sum()),
            "ctr": float(clean.click.mean()), "removed_rows": int((~clean_mask).sum()),
            "rankable_impressions": int(clean_groups.rankable.sum()),
            "randomized_rankable_impressions": int((clean_groups.randomized.eq(1) & clean_groups.rankable).sum()),
            "duplicate_impression_product_rows": int(clean.duplicated(["impression_id", "product_id"]).sum()),
            "duplicate_impression_rank_rows": int(clean.duplicated(["impression_id", "rank"]).sum()),
        },
        "proposed_clean_user_disjoint_split": clean_split_profile,
    }

    profile = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(data), "columns": len(data.columns), "column_names": list(data),
        "users": int(data.user_id.nunique()), "impressions": len(groups),
        "products": int(data.product_id.nunique()), "clicks": int(data.click.sum()),
        "ctr": float(data.click.mean()), "dataframe_memory_bytes": int(data.memory_usage(deep=True).sum()),
        "feature_columns": {"categorical": cat, "numerical": num},
        "null_counts": {c: int(v) for c, v in data.isna().sum().items()},
        "nonfinite_numeric_counts": {c: int((~np.isfinite(data[c])).sum()) for c in num},
        "duplicate_complete_rows": int(data.duplicated().sum()),
        "duplicate_impression_product_rows": int(data.duplicated(["impression_id", "product_id"]).sum()),
        "duplicate_impression_rank_rows": int(data.duplicated(["impression_id", "rank"]).sum()),
        "binary_value_counts": {c: {str(k): int(v) for k,v in data[c].value_counts(dropna=False).items()} for c in ["click", "protected_attribute", "senior", "displayrandom"]},
        "candidate_counts": distribution(groups.n_ads),
        "candidate_count_histogram": {str(k): int(v) for k,v in groups.n_ads.value_counts().sort_index().items()},
        "impression_counts": group_counts,
        "varying_within_impression_counts": variation,
        "varying_within_product_counts": product_variation,
        "user_feature_variation_across_impressions": user_variation,
        "categorical_cardinalities": {c: int(data[c].nunique()) for c in cat},
        "numeric_summary": {c: {"min": float(data[c].min()), "max": float(data[c].max()), "mean": float(data[c].mean()), "std": float(data[c].std())} for c in num},
        "position_statistics": data.groupby("rank").click.agg(["size", "sum", "mean"]).reset_index().to_dict("records"),
        "split_algorithm": "SHA256('fairjob-v1:' + decimal user_id), first 8 bytes big-endian modulo 100; 0-79 train, 80-89 validation, 90-99 test",
        "proposed_user_disjoint_split": split_profile,
        "key_diagnostics": key_diagnostics,
        "split_materialized": False,
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "fairjob_profile.json").write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    (out / "key_diagnostics.json").write_text(json.dumps(key_diagnostics, indent=2) + "\n")
    group_columns = ["impression_id", "user_id", "product_id", "rank", "click", "displayrandom"]
    examples = []
    for label, mask in [
        ("clicked_randomized", groups.randomized.eq(1) & groups.rankable),
        ("clicked_nonrandomized", groups.randomized.eq(0) & groups.rankable),
        ("no_click", groups.clicks.eq(0) & groups.n_ads.ge(2)),
    ]:
        ids = groups.loc[mask].index[:1]
        frame = data.loc[data.impression_id.isin(ids), group_columns].sort_values("rank")
        examples.append({"type": label, "rows": frame.to_dict("records")})
    (out / "impression_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    print(json.dumps({k: profile[k] for k in ["rows", "columns", "users", "impressions", "products", "clicks", "ctr", "duplicate_complete_rows", "duplicate_impression_product_rows", "candidate_counts", "impression_counts", "varying_within_impression_counts", "varying_within_product_counts", "proposed_user_disjoint_split"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
