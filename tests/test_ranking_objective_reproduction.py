from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from fairjob_ranker.diagnostics.ranking_objective_reproduction import (
    Top1Metric, balanced_query_weights, grouped_frame,
)


def example():
    return pd.DataFrame({'impression_id': [1]*3+[2]*2+[3]*2+[4]*2+[5],
        'user_id': [7]*5+[8]*2+[9]*2+[10], 'click': [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        'rank': [1, 2, 3, 1, 2, 1, 2, 1, 2, 1], 'product_id': np.arange(10)})


def test_top1_callback_matches_brute_force_ties_and_retains_zero_clicks():
    frame = example()
    scores = np.array([.7, .7, .1, -4, -4, 2, 1, 0, 0, 99.])
    values = []
    for _, part in frame.assign(score=scores).groupby('impression_id'):
        if len(part) < 2:
            continue
        tied = part.loc[part.score.eq(part.score.max()), 'click'].to_numpy()
        values.append(np.mean([p[0] for p in permutations(tied)]))
    assert Top1Metric(frame).value(scores) == pytest.approx(np.mean(values))
    assert Top1Metric(frame).value(np.ones(len(frame))) == pytest.approx((1/3+1/2+1/2+0)/4)
    changed = frame.copy()
    changed.loc[changed.impression_id.eq(3), 'click'] = 1
    assert Top1Metric(changed).value(scores) == pytest.approx(.5)
    with pytest.raises(ValueError, match='finite'):
        Top1Metric(frame).value(scores[:-1])


def test_weighting_equalizes_informative_query_mass_per_user_not_per_row():
    frame = example()
    weights, audit = balanced_query_weights(frame)
    # User7 has 2 informative queries; user8 has 1. Mean informative query weight is1.
    np.testing.assert_allclose(weights[:7], [.75]*5+[1.5]*2)
    np.testing.assert_allclose(weights[7:], 1.)
    assert audit['informative_queries'] == 3 and audit['informative_users'] == 2
    assert audit['per_user_query_weight_sum_min'] == pytest.approx(1.5)
    assert audit['per_user_query_weight_sum_max'] == pytest.approx(1.5)


def test_grouping_keeps_queries_contiguous_without_using_labels_or_positions():
    frame = example().iloc[::-1].reset_index(drop=True)
    sorted_frame, sizes = grouped_frame(frame, 42)
    changed = frame.copy()
    changed['click'], changed['rank'] = 1-changed.click, 100-changed['rank']
    other, _ = grouped_frame(changed, 42)
    np.testing.assert_array_equal(sorted_frame.product_id, other.product_id)
    np.testing.assert_array_equal(sizes, [3, 2, 2, 2, 1])
    assert len(sorted_frame) == len(frame)
