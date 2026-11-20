from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from fairjob_ranker.diagnostics.business_validation import (
    bootstrap_request_means, request_outcomes, topk_weights,
)


def test_topk_ties_match_exhaustive_permutations_across_selection_boundary():
    scores = np.array([.9, .6, .6, .6, .1])
    labels = np.array([0, 1, 0, 1, 1])
    possible = [np.r_[0, middle, 4] for middle in permutations([1, 2, 3])]
    for k in [1, 2, 3, 4, 7]:
        expected = np.mean([labels[order[:k]].sum() for order in possible])
        weights = topk_weights(scores, k)
        assert weights @ labels == pytest.approx(expected)
        assert weights.sum() == pytest.approx(min(k, len(scores)))
    with pytest.raises(ValueError):
        topk_weights([np.nan], 1)
    with pytest.raises(ValueError):
        topk_weights([.5], 0)


def test_zero_click_requests_stay_in_business_denominator_and_singletons_do_not():
    frame = pd.DataFrame({'impression_id': [1, 1, 2, 2, 3], 'user_id': [7, 7, 8, 8, 9],
                          'product_id': [10, 20, 30, 40, 50], 'displayrandom': [1]*5,
                          'click': [1, 0, 0, 0, 1], 'rank': [1, 2, 2, 4, 1]})
    outcomes = request_outcomes(frame, np.zeros(5))
    assert outcomes.impression_id.tolist() == [1, 2]
    assert outcomes.captured_at_1.mean() == .25
    assert outcomes.captured_at_3.sum() == 1
    assert outcomes.observed_rank_contiguous.tolist() == [True, False]
    # Exactly half a click in expectation, not the ID-tie rule's one click.
    assert outcomes.captured_at_1.tolist() == [.5, 0.]
    assert outcomes.id_ties_at_1.tolist() == [1., 0.]
    changed = frame.iloc[::-1].copy()
    changed.product_id = [90, 80, 70, 60, 50]
    again = request_outcomes(changed, np.zeros(5))
    np.testing.assert_array_equal(outcomes.captured_at_1, again.captured_at_1)


def test_cluster_resampling_is_paired_and_request_weighted():
    frame = pd.DataFrame({'user_id': [1, 1, 2], 'candidate': [1., 1., 0.],
                          'reference': [.5, .5, 0.]})
    point, draws = bootstrap_request_means(frame, ['candidate', 'reference'], 1000, 42)
    np.testing.assert_allclose(point, [2/3, 1/3])
    np.testing.assert_allclose(draws[:, 0], draws[:, 1]*2)
    # Two draws of user 1 keep both of their requests each time; drawing
    # each user once yields 2/3, not the user-macro average 1/2.
    assert set(draws[:, 0]) == {0., 2/3, 1.}


def test_all_click_groups_retained_and_oracle_is_an_upper_bound():
    frame = pd.DataFrame({'impression_id': [1]*3, 'user_id': [7]*3,
                          'product_id': [10, 20, 30], 'displayrandom': [0]*3,
                          'click': [1, 1, 1], 'rank': [1, 2, 3]})
    row = request_outcomes(frame, [.1, .9, .2]).iloc[0]
    assert row.captured_at_1 == row.oracle_at_1 == 1
    assert row.captured_at_3 == row.oracle_at_3 == 3
