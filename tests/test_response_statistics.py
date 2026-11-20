import numpy as np
import pandas as pd
import pytest

from fairjob_ranker.diagnostics.response_statistics import (
    AD_FEATURES, KEY_COLUMNS, ResponseStatistics, cross_fit_statistics, statistic_columns,
)


def example():
    frame = pd.DataFrame({'user_id': np.repeat(np.arange(80), 3),
        'impression_id': np.repeat(np.arange(80), 3), 'product_id': np.tile([10, 11, 12], 80),
        'click': np.tile([0, 1, 0, 1, 0, 1], 40), 'cat0': np.repeat(np.arange(80)%2, 3),
        'cat1': np.repeat(np.arange(80)%3, 3)})
    for i, ad in enumerate(AD_FEATURES):
        frame[ad] = (frame.product_id+i)%2
    return frame


def configuration():
    return {'encoding_folds': 5, 'encoding_salt': 'test-user-encoding', 'product_smoothing': 20., 'category_smoothing': 100.}


def test_flipping_all_labels_of_a_user_cannot_change_that_users_training_features():
    frame = example()
    original, assignments, audits = cross_fit_statistics(frame, configuration(), 0)
    changed = frame.copy()
    user = 7
    changed.loc[changed.user_id.eq(user), 'click'] = 1-changed.loc[changed.user_id.eq(user), 'click']
    other, _, _ = cross_fit_statistics(changed, configuration(), 0)
    fold = assignments.set_index('user_id').loc[user, 'encoding_fold']
    excluded_users = assignments.loc[assignments.encoding_fold.eq(fold), 'user_id']
    mask = frame.user_id.isin(excluded_users)
    np.testing.assert_array_equal(original.loc[mask], other.loc[mask])
    assert not np.array_equal(original.loc[~mask], other.loc[~mask])
    assert sum(a['target_rows'] for a in audits) == len(frame)


def test_unseen_product_uses_attribute_priors_and_unknown_attributes_fall_back_safely():
    encoder = ResponseStatistics.fit(example())
    query = example().iloc[:2][KEY_COLUMNS].copy()
    query['product_id'] = [900, 901]
    result = encoder.transform(query)
    for context in [None, 'cat0', 'cat1']:
        suffix = '' if context is None else '__'+context
        rate = 'rs__product_id'+suffix+'__p'
        parent = result[[f'rs__{a}{suffix}__p' for a in AD_FEATURES]].mean(axis=1)
        np.testing.assert_allclose(result[rate], parent, rtol=1e-6)
        assert result['rs__product_id'+suffix+'__log_n'].eq(0).all()
    unknown = query.copy()
    unknown.loc[:, KEY_COLUMNS] = 99999
    fallback = encoder.transform(unknown)
    np.testing.assert_allclose(fallback[[c for c in fallback if c.endswith('__p')]], encoder.prior, rtol=1e-6)
    assert fallback.filter(like='__log_n').eq(0).all().all()


def test_transform_rejects_labels_and_round_trip_preserves_exact_features(tmp_path):
    encoder = ResponseStatistics.fit(example())
    query = example().iloc[:5]
    with pytest.raises(ValueError, match='label-free'):
        encoder.transform(query[KEY_COLUMNS+['click']])
    expected = encoder.transform(query[KEY_COLUMNS])
    encoder.save(tmp_path/'stats')
    pd.testing.assert_frame_equal(expected, ResponseStatistics.load(tmp_path/'stats').transform(query[KEY_COLUMNS]))
    assert len(statistic_columns('ad_stats')) == 24
    assert len(statistic_columns('ad_context_stats')) == 76


def test_hierarchical_smoothing_matches_direct_counts():
    frame = example()
    encoder = ResponseStatistics.fit(frame)
    query = frame.iloc[:1][KEY_COLUMNS]
    result = encoder.transform(query).iloc[0]
    key = query.cat2.iloc[0]
    source = frame.loc[frame.cat2.eq(key)]
    expected = (source.click.sum()+100*(frame.click.sum()+.5)/(len(frame)+1))/(len(source)+100)
    assert result.rs__cat2__p == pytest.approx(expected, rel=1e-6)
    product = frame.loc[frame.product_id.eq(query.product_id.iloc[0])]
    parent = result[[f'rs__{a}__p' for a in AD_FEATURES]].mean()
    assert result.rs__product_id__p == pytest.approx((product.click.sum()+20*parent)/(len(product)+20), rel=1e-6)


def test_cold_capture_keeps_full_competition_and_cannot_credit_known_ad_clicks():
    from fairjob_ranker.diagnostics.response_statistics_experiment import selection_outcomes
    frame = pd.DataFrame({'impression_id': [1, 1, 2, 2, 2], 'user_id': [8, 8, 9, 9, 9],
        'product_id': [10, 11, 12, 13, 14], 'click': [1, 1, 1, 0, 1], 'rank': [1, 2, 1, 2, 3],
        'displayrandom': [1]*5, 'strict_unseen': [False, True, True, True, False],
        'fit_unseen': [False, True, True, True, False]})
    for variant in ['baseline', 'ad_stats', 'ad_context_stats']:
        frame[variant+'-s42'] = [2., 1., 3., 3., 1.]
    result = selection_outcomes(frame, {'variants': ['ad_stats', 'ad_context_stats'], 'seeds': [42]})
    np.testing.assert_allclose(result.baseline1, [1., .5])
    np.testing.assert_allclose(result.baseline_cold1, [0., .5])
    np.testing.assert_allclose(result.baseline_cold_selected1, [0., 1.])
    np.testing.assert_allclose(result.uniform_cold1, [.5, 1/3])
