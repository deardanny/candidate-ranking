import numpy as np
import pytest
from fairjob_ranker.ebnerd.feedback import FeedbackState, hit1


def test_strict_maturity_boundary_and_zero_unknown():
    state = FeedbackState(delay=60)
    a = state.matrix(100, [1, 2, 3])
    state.observe(100, [1, 2], [1])
    np.testing.assert_array_equal(state.matrix(160, [1, 2, 3]), a)
    x = state.matrix(161, [1, 2, 3])
    assert x[0, 1] == np.float32(np.log(2))
    assert x[1, 1] == 0 and x[2, 0] == 0
    assert x[0, 2] > x[1, 2]


def test_current_future_labels_cannot_change_prefix():
    def replay(click):
        state = FeedbackState(delay=10)
        xs = []
        for t in [0, 0, 10, 11, 20]:
            xs.append(state.matrix(t, [1, 2]))
            state.observe(t, [1, 2], [click] if t >= 10 else [1])
        return xs
    a, b = replay(1), replay(2)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_window_counts_match_independent_event_filter():
    state = FeedbackState(delay=5)
    events = [(0, [1, 2], [1]), (3595, [2, 3], [3]), (3600, [1, 2], [2]), (86401, [2, 3], [2])]
    for t, ids, clicked in events:
        state.matrix(t, ids)
        state.observe(t, ids, clicked)
    state.matrix(86410, [1, 2, 3])
    for w in (0, 3600, 86400, 604800):
        chosen = [e for e in events if e[0] < 86405 and (not w or e[0] >= 86405-w)]
        for i in [1, 2, 3]:
            assert state.exposures[w][i] == sum(i in e[1] for e in chosen)
            assert state.clicks[w][i] == sum(i in e[2] for e in chosen)


def test_permutation_equivariance_and_time_guard():
    s = FeedbackState(delay=0)
    s.matrix(0, [1, 2]); s.observe(0, [1, 2], [1])
    x = s.matrix(1, [1, 2, 3])
    np.testing.assert_array_equal(s.matrix(1, [3, 1, 2]), x[[2, 0, 1]])
    with pytest.raises(ValueError): s.matrix(0, [1])


def test_uniform_ties_not_input_order_and_zero_positive():
    np.testing.assert_allclose(hit1(np.array([1, 0, 0, 0, 0]), np.ones(5), [3, 2]), [1/3, 0])


def test_repeated_click_is_one_binary_response():
    s = FeedbackState(delay=0)
    s.matrix(0, [1, 2]); s.observe(0, [1, 2], [1, 1])
    s.matrix(1, [1, 2])
    assert s.clicks[0][1] == 1 and s.totals[0] == [2, 1]


def test_cold_hit_competes_with_warm_positive():
    # Warm clicked item wins; a cold clicked item elsewhere does not earn credit.
    y=np.array([1,1,0]); cold=np.array([False,True,True]); score=np.array([3.,2.,1.])
    assert hit1(y,score,[3])[0]==1
    assert hit1(y*cold,score,[3])[0]==0
