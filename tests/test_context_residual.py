from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from fairjob_ranker.diagnostics.context_ablation import run_experiment as run_parent
from fairjob_ranker.diagnostics.context_residual import combine_scores, fit_correction, predict_bundle, run_experiment, user_folds
from fairjob_ranker.io import sha256, write_json
from fairjob_ranker.schema import AD_FEATURES, NUMERICAL


def test_probability_mean_preservation_is_not_raw_score_centering():
    p = np.array([.02, .02, .02, .8, .8, .07])
    raw = np.array([4., 0., -1., 1., -2., 9.])
    ids = np.array([10, 10, 10, 20, 20, 30])
    result = combine_scores(p, raw, ids, .5, True)
    for impression in np.unique(ids):
        mask = ids == impression
        assert result[mask].mean() == pytest.approx(p[mask][0], abs=1e-12)
        assert np.array_equal(np.argsort(result[mask]), np.argsort(raw[mask]))
    naive = expit(logit(p[:3])+.5*(raw[:3]-raw[:3].mean()))
    assert abs(naive.mean()-.02) > .001
    assert result[-1] == pytest.approx(.07)


def test_normalized_predictions_are_permutation_invariant_and_alpha_zero_is_exact():
    p, r, ids = np.array([.1, .1, .1, .2, .2]), np.array([1., 1., -2., 4., 8.]), np.array([1, 1, 1, 2, 2])
    assert np.array_equal(combine_scores(p, r, ids, 0., True), p)
    assert np.array_equal(combine_scores(p, r, ids, 0., False), p)
    expected = combine_scores(p, r, ids, .25, True)
    order = np.array([3, 1, 4, 0, 2])
    actual = combine_scores(p[order], r[order], ids[order], .25, True)
    np.testing.assert_allclose(actual, expected[order], atol=1e-14, rtol=0)
    assert expected[0] == expected[1]
    with pytest.raises(ValueError, match='constant'):
        combine_scores([.1, .2], [0., 1.], [1, 1], .5, True)


def test_group_fold_assignment_never_splits_a_user():
    frame = pd.DataFrame({'user_id': np.repeat(np.arange(100), 3)})
    folds = user_folds(frame, 3, 'test-fold')
    assert frame.assign(fold=folds).groupby('user_id').fold.nunique().max() == 1
    shuffled = frame.sample(frac=1, random_state=2)
    other = user_folds(shuffled, 3, 'test-fold')
    assert dict(zip(frame.user_id, folds)) == dict(zip(shuffled.user_id, other))


def test_lightgbm_init_score_must_be_restored_after_model_reload(tmp_path):
    n = 80
    x = pd.DataFrame({c: np.tile([1, 2], n//2) for c in AD_FEATURES})
    y = np.tile([0, 1], n//2)
    offset = np.full(n, -2.)
    config = {'params': {'objective': 'binary', 'metric': 'binary_logloss', 'num_threads': 1,
                         'verbosity': -1, 'min_data_in_leaf': 2, 'min_data_per_group': 1, 'num_leaves': 3},
              'num_boost_round': 4, 'early_stopping_rounds': 2}
    model, _, loss = fit_correction(x, y, offset, x, y, offset, config)
    path = tmp_path/'model.txt'
    model.save_model(str(path))
    saved = lgb.Booster(model_file=str(path))
    correction = saved.predict(x, raw_score=True, num_threads=1)
    combined = combine_scores(expit(offset), correction, np.repeat(np.arange(n//2), 2), 1., False)
    np.testing.assert_allclose(combined, expit(offset+correction), atol=1e-14)
    assert np.max(abs(saved.predict(x, num_threads=1)-combined)) > .1


def make_frame(offset, count):
    ads = np.tile([1, 2, 3], count)
    context = np.repeat(np.arange(count) % 4, 3)
    frame = pd.DataFrame({**{c: ads for c in AD_FEATURES}, 'cat0': context, 'cat1': context,
                          **{c: context.astype(float) for c in NUMERICAL}})
    frame['user_id'] = np.repeat(np.arange(count)+offset, 3)
    frame['impression_id'] = frame.user_id
    frame['product_id'] = ads
    frame['click'] = (ads == 1).astype(int)
    frame['displayrandom'] = np.repeat(np.arange(count) % 2, 3)
    frame['rank'] = ads
    return frame


def test_complete_residual_pipeline_freezes_selection_without_reading_test(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    for name, frame in [('train', make_frame(0, 60)), ('validation', make_frame(1000, 10))]:
        frame.to_parquet(data/f'{name}.parquet', index=False)
    (data/'test.parquet').write_bytes(b'NEVER READ THIS FILE')
    write_json(data/'manifest.json', {'files': {p.name: {'sha256': sha256(p)} for p in data.glob('*.parquet') if p.name != 'test.parquet'}})
    lgb_path = tmp_path/'lgb.json'
    write_json(lgb_path, {'params': {'objective': 'binary', 'metric': 'binary_logloss', 'num_threads': 1,
                                    'verbosity': -1, 'min_data_in_leaf': 2, 'min_data_per_group': 1, 'num_leaves': 3},
                         'num_boost_round': 3, 'early_stopping_rounds': 2})
    parent = json.loads(Path('configs/context_ablation.json').read_text())
    parent.update(lightgbm_config=str(lgb_path), seeds=[42], bootstrap_repeats=5)
    write_json(tmp_path/'parent.json', parent)
    run_parent(data, tmp_path/'parent.json', tmp_path/'parent_models', tmp_path/'parent_report')
    config = json.loads(Path('configs/context_residual.json').read_text())
    config.update(parent_experiment=str(tmp_path/'parent_report/metrics.json'), parent_models=str(tmp_path/'parent_models'),
                  seeds=[42], lightgbm_config=str(lgb_path), bootstrap_repeats=5, alpha_grid=[0., .1, 1.], example_impression=1000)
    write_json(tmp_path/'config.json', config)
    result = run_experiment(data, tmp_path/'config.json', tmp_path/'models', tmp_path/'report')
    assert not result['test_read'] and not result['existing_models_modified']
    assert all(a['overlap_users'] == 0 and not a['held_labels_used_for_fit_or_stopping'] for a in result['oof_audits'])
    selection = json.loads((tmp_path/'report/selection.json').read_text())
    assert selection['outer_read'] is False
    predictions = pd.read_parquet(tmp_path/'report/outer_predictions.parquet')
    frame = pd.read_parquet(data/'validation.parquet')
    for policy in config['policies']:
        np.testing.assert_allclose(predict_bundle(tmp_path/'models', frame, 42, policy), predictions[f'{policy}-s42'], atol=1e-14)
        if policy != 'ad_offset':
            assert result['mean_drift']['outer'][f'{policy}-s42']['max_absolute_impression_mean_change'] < 1e-12
