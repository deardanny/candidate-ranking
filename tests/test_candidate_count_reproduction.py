import numpy as np
import pandas as pd
import pytest

from fairjob_ranker.diagnostics.business_validation import request_outcomes
from fairjob_ranker.diagnostics.candidate_count_reproduction import (
    split_frames, user_assignments, vector_outcomes,
)


def test_vectorized_topk_matches_independent_tie_enumeration_on_zero_and_all_click_groups():
    frame = pd.DataFrame({'impression_id': [1]*5+[2]*3+[3]*2+[4], 'user_id': [7]*5+[8]*3+[9]*2+[10],
                          'product_id': list(range(11)), 'rank': [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1],
                          'click': [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1], 'displayrandom': [1]*11})
    scores = np.array([.9, .6, .6, .6, .1, .8, .3, .2, .4, .4, .9])
    actual, _ = vector_outcomes(frame, scores)
    reference = request_outcomes(frame, scores)
    assert actual.impression_id.tolist() == [1, 2, 3]
    for k in [1, 3]:
        np.testing.assert_allclose(actual[f'capture{k}'], reference[f'captured_at_{k}'])
    assert actual.capture1.iloc[1] == 0 and actual.capture1.iloc[2] == 1


def test_nested_user_splits_cover_once_are_disjoint_and_ignore_labels_or_row_order():
    config = {'outer_folds': 3, 'outer_salt': 'outer-test', 'inner_salt': 'inner-test', 'inner_train_percent': 80}
    frame = pd.DataFrame({'user_id': np.repeat(np.arange(120), 4),
                          'impression_id': np.repeat(np.arange(240), 2), 'click': [0, 1]*240})
    assignments = user_assignments(frame, config)
    changed = frame.iloc[::-1].copy()
    changed.click = 1-changed.click
    pd.testing.assert_frame_equal(assignments, user_assignments(changed, config))
    held_ids = []
    for fold in range(3):
        fit, inner, held = split_frames(frame, assignments, fold)
        for key in ['user_id', 'impression_id']:
            sets = [set(p[key]) for p in [fit, inner, held]]
            assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
        held_ids.extend(held.impression_id.unique())
    assert sorted(held_ids) == sorted(frame.impression_id.unique())
    with pytest.raises(ValueError, match='exactly once'):
        split_frames(frame, assignments.iloc[:-1], 0)
