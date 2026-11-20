"""All-row click metrics and explicitly conditional impression-level metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def click_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("Labels and scores must be aligned 1-D arrays.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Binary labels and finite scores required.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Click scores must be in [0, 1].")
    n = len(labels)
    result = {"rows": n, "clicks": int(labels.sum())}
    if not n:
        return {**result, "ctr": None, "logloss": None, "average_precision": None,
                "roc_auc": None, "brier": None, "calibration": []}
    result.update({
        "ctr": float(labels.mean()), "mean_prediction": float(scores.mean()),
        "logloss": float(log_loss(labels, scores, labels=[0, 1])),
        # AP is a step-weighted precision/recall summary, not trapezoidal PR-AUC.
        "average_precision": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "brier": float(np.square(scores - labels).mean()),
    })
    # Quantile boundaries keep low-CTR calibration useful. Tied scores stay together.
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, 11)))
    bins = np.searchsorted(edges[1:-1], scores, side="right")
    result["calibration"] = [
        {"bin": int(b), "rows": int((bins == b).sum()),
         "clicks": int(labels[bins == b].sum()),
         "mean_prediction": float(scores[bins == b].mean()),
         "observed_ctr": float(labels[bins == b].mean())}
        for b in np.unique(bins)
    ]
    return result


def rank_groups(frame: pd.DataFrame, scores: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Score only groups with >=2 candidates and both positive/negative labels.

    Ties use product_id, never original display rank or input order.
    """
    columns = ["impression_id", "user_id", "displayrandom", "candidates", "clicks", "ndcg_at_5", "mrr"]
    if k != 5:
        raise ValueError("This version's metric contract fixes NDCG at k=5.")
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("One finite score per row is required.")
    if frame.duplicated(["impression_id", "product_id"]).any():
        raise ValueError("Ambiguous candidates must be cleaned before ranking evaluation.")
    grouping = frame.groupby("impression_id", sort=False)
    if grouping[["user_id", "displayrandom"]].nunique().gt(1).any().any():
        raise ValueError("Each impression must have a single user and randomization flag.")
    stats = grouping.agg(
        user_id=("user_id", "first"), displayrandom=("displayrandom", "first"),
        candidates=("click", "size"), clicks=("click", "sum")
    )
    stats = stats.loc[stats.candidates.ge(2) & stats.clicks.gt(0) & stats.clicks.lt(stats.candidates)].copy()
    if stats.empty:
        return pd.DataFrame(columns=columns)
    ordered = frame[["impression_id", "product_id", "click"]].assign(score=np.asarray(scores))
    ordered = ordered.loc[ordered.impression_id.isin(stats.index)].sort_values(
        ["impression_id", "score", "product_id"], ascending=[True, False, True]
    )
    ordered["position"] = ordered.groupby("impression_id").cumcount() + 1
    ordered["dcg"] = np.where(ordered.position.le(k), ordered.click / np.log2(ordered.position + 1), 0.0)
    dcg = ordered.groupby("impression_id").dcg.sum().reindex(stats.index)
    ideal = np.cumsum(1 / np.log2(np.arange(1, k + 1) + 1))
    stats["ndcg_at_5"] = dcg / ideal[np.minimum(stats.clicks.to_numpy(dtype=int), k) - 1]
    first_click = ordered.loc[ordered.click.eq(1)].groupby("impression_id").position.min()
    stats["mrr"] = 1.0 / first_click.reindex(stats.index)
    return stats.reset_index()[columns]


def ranking_summary(groups: pd.DataFrame, total_impressions: int) -> dict:
    return {
        "eligible_impressions": len(groups), "eligible_users": int(groups.user_id.nunique()),
        "total_impressions": int(total_impressions),
        "coverage": float(len(groups) / total_impressions) if total_impressions else None,
        "ndcg_at_5": float(groups.ndcg_at_5.mean()) if len(groups) else None,
        "mrr": float(groups.mrr.mean()) if len(groups) else None,
    }


def evaluate(frame: pd.DataFrame, scores: np.ndarray, training_products: set[int]) -> tuple[dict, pd.DataFrame]:
    scores = np.asarray(scores)
    groups = rank_groups(frame, scores)
    report = {}
    for name, mask in {
        "all": np.ones(len(frame), dtype=bool),
        "randomized": frame.displayrandom.eq(1).to_numpy(),
    }.items():
        subset = frame.loc[mask]
        gs = groups if name == "all" else groups.loc[groups.displayrandom.eq(1)]
        report[name] = {
            "click": click_metrics(subset.click.to_numpy(), scores[mask]),
            "ranking": ranking_summary(gs, subset.impression_id.nunique()),
        }
    known = frame.product_id.isin(training_products).to_numpy()
    report["product_cohorts"] = {
        name: click_metrics(frame.click.to_numpy()[mask], scores[mask])
        for name, mask in {"seen_in_train": known, "unseen_in_train": ~known}.items()
    }
    return report, groups


def paired_user_bootstrap(model: pd.DataFrame, baseline: pd.DataFrame, repeats: int = 1000, seed: int = 42) -> dict:
    """Resample whole eligible users; preserve each user's impression multiplicity."""
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive.")
    keys = ["impression_id", "user_id", "displayrandom"]
    paired = model.merge(baseline, on=keys, suffixes=("_model", "_baseline"), validate="one_to_one")
    if len(paired) != len(model) or len(paired) != len(baseline):
        raise ValueError("Models must be evaluated on the same eligible groups.")
    report = {"unit": "user", "repeats": repeats, "seed": seed, "difference": "lightgbm - train_ctr_constant"}
    for name, subset in {"all": paired, "randomized": paired.loc[paired.displayrandom.eq(1)]}.items():
        result = {"impressions": len(subset), "users": int(subset.user_id.nunique())}
        for metric in ["ndcg_at_5", "mrr"]:
            if subset.empty:
                result[metric] = None
                continue
            differences = subset[f"{metric}_model"] - subset[f"{metric}_baseline"]
            per_user = differences.groupby(subset.user_id).agg(["sum", "size"])
            rng = np.random.default_rng(seed)
            estimates = np.empty(repeats)
            sums, counts = per_user["sum"].to_numpy(), per_user["size"].to_numpy()
            for i in range(repeats):
                drawn = rng.integers(0, len(per_user), size=len(per_user))
                estimates[i] = sums[drawn].sum() / counts[drawn].sum()
            result[metric] = {
                "mean_difference": float(differences.mean()),
                "ci95": [float(v) for v in np.quantile(estimates, [.025, .975])],
            }
        report[name] = result
    return report

