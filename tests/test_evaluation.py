import numpy as np
import pandas as pd
import pytest

from fairjob_ranker.evaluation import click_metrics, paired_user_bootstrap, rank_groups, ranking_summary


def ranking_frame():
    return pd.DataFrame({
        "impression_id": [10, 10, 20, 20, 30], "user_id": [1, 1, 2, 2, 3],
        "product_id": [2, 1, 1, 2, 1], "click": [1, 0, 0, 0, 1],
        "displayrandom": [1, 1, 0, 0, 1], "rank": [1, 99, 1, 2, 1],
    })


def test_hand_calculated_ranking_and_input_order_invariance():
    frame = ranking_frame()
    groups = rank_groups(frame, np.full(len(frame), .1))
    assert groups.impression_id.tolist() == [10]
    assert groups.iloc[0].ndcg_at_5 == pytest.approx(1 / np.log2(3))
    assert groups.iloc[0].mrr == .5
    shuffled = frame.sample(frac=1, random_state=3)
    pd.testing.assert_frame_equal(groups, rank_groups(shuffled, np.full(len(frame), .1)))
    summary = ranking_summary(groups, frame.impression_id.nunique())
    assert summary["coverage"] == pytest.approx(1 / 3)


def test_multiclick_ideal_dcg():
    frame = pd.DataFrame({"impression_id": [1]*3, "user_id": [1]*3,
                          "displayrandom": [0]*3, "product_id": [1, 2, 3], "click": [1, 0, 1]})
    groups = rank_groups(frame, np.array([.9, .5, .1]))
    assert groups.iloc[0].ndcg_at_5 == pytest.approx((1 + .5) / (1 + 1/np.log2(3)))
    assert groups.iloc[0].mrr == 1


def test_click_metrics_hand_values_and_constant_calibration():
    result = click_metrics(np.array([0, 1]), np.array([.5, .5]))
    assert result["logloss"] == pytest.approx(np.log(2))
    assert result["average_precision"] == .5
    assert result["brier"] == .25
    assert result["roc_auc"] == .5
    assert len(result["calibration"]) == 1
    assert result["calibration"][0]["rows"] == 2
    negative_only = click_metrics(np.zeros(2), np.full(2, .1))
    assert negative_only["roc_auc"] is None and negative_only["average_precision"] is None


def test_bootstrap_identical_models_has_zero_difference():
    groups = rank_groups(ranking_frame(), np.full(5, .1))
    report = paired_user_bootstrap(groups, groups, repeats=100)
    assert report["all"]["ndcg_at_5"]["ci95"] == [0., 0.]
    assert report["randomized"]["mrr"]["mean_difference"] == 0.


def test_bootstrap_resamples_users_not_individual_impressions():
    base = pd.DataFrame({"impression_id": [1, 2, 3], "user_id": [10, 10, 20],
                         "displayrandom": [1, 1, 1], "ndcg_at_5": [0., 0., 0.], "mrr": [0., 0., 0.]})
    model = base.assign(ndcg_at_5=[1., 1., 0.], mrr=[1., 1., 0.])
    report = paired_user_bootstrap(model, base, repeats=100, seed=1)
    assert report["all"]["users"] == 2
    assert report["all"]["mrr"]["mean_difference"] == pytest.approx(2/3)
    assert report["all"]["mrr"]["ci95"] == [0., 1.]


def test_duplicate_candidates_rejected():
    frame = ranking_frame()
    frame.loc[0, "product_id"] = 1
    with pytest.raises(ValueError, match="Ambiguous"):
        rank_groups(frame, np.ones(5))
