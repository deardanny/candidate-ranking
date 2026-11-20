from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import ndcg_score

from fairjob_ranker.diagnostics.context_ablation import (
    bootstrap_mean_by_user, diagnostic_groups, expected_ndcg, feature_audit,
    run_experiment, span_summary,
)
from fairjob_ranker.features import FeatureTransformer
from fairjob_ranker.io import sha256, write_json
from fairjob_ranker.schema import AD_FEATURES, CATEGORICAL, CONTEXT, NUMERICAL


def test_tie_expectation_matches_exhaustive_rankings_and_sklearn():
    # A four-way tie crosses the top-5 boundary; two tied items are clicked.
    y = np.array([1, 0, 1, 0, 0, 1, 0])
    p = np.array([.9, .8, .7, .4, .4, .4, .4])
    discount = 1 / np.log2(np.arange(5)+2)
    ideal = discount[:3].sum()
    brute = []
    for ordering in permutations(range(3, 7)):
        labels = y[np.r_[np.arange(3), ordering]]
        brute.append((labels[:5]*discount).sum()/ideal)
    actual = expected_ndcg(y, p)
    assert actual == pytest.approx(np.mean(brute))
    assert actual == pytest.approx(ndcg_score(y[None, :], p[None, :], k=5, ignore_ties=False))
    assert expected_ndcg(np.zeros(3), np.zeros(3)) == 0


def test_context_ties_have_no_ad_discrimination_despite_id_order():
    frame = pd.DataFrame({'impression_id': [1]*3, 'user_id': [10]*3, 'displayrandom': [1]*3,
                          'product_id': [1, 2, 3], 'click': [1, 0, 0]})
    scores = np.full(3, .2)
    a = diagnostic_groups(frame, scores).iloc[0]
    assert a.legacy_ndcg_id_ties == 1
    assert a.ndcg_tie_averaged == pytest.approx((1+1/np.log2(3)+.5)/3)
    assert a.pair_accuracy == .5 and a.pair_tie_rate == 1
    assert a.top1_click_tie_averaged == pytest.approx(1/3)
    changed = frame.iloc[::-1].copy()
    changed['product_id'] = [1, 2, 3]
    b = diagnostic_groups(changed, scores).iloc[0]
    assert a.ndcg_tie_averaged == b.ndcg_tie_averaged
    assert a.legacy_ndcg_id_ties != b.legacy_ndcg_id_ties
    assert span_summary(frame, scores, 1e-12)['all_scores_tied_fraction'] == 1


def test_pair_accuracy_credits_ties_and_normalizes_each_impression():
    frame = pd.DataFrame({'impression_id': [1]*3+[2]*2, 'user_id': [1]*3+[2]*2,
                          'displayrandom': [0]*5, 'product_id': [1, 2, 3, 1, 2],
                          'click': [1, 0, 0, 1, 0]})
    groups = diagnostic_groups(frame, np.array([.8, .8, .1, .1, .9]))
    assert groups.pair_accuracy.tolist() == [.75, 0.]
    assert groups.pair_accuracy.mean() == .375
    assert groups.positive_negative_pairs.tolist() == [2, 1]


def test_user_bootstrap_preserves_cluster_multiplicity():
    result = bootstrap_mean_by_user([1, 1, 2], [1, 1, 0], 200, 42)
    assert result['mean_difference'] == pytest.approx(2/3)
    assert result['ci95'] == [0., 1.]
    assert result['users'] == 2


def make_data(offset, groups):
    n = groups*3
    product = np.tile([1, 2, 3], groups)
    context = np.repeat(np.arange(groups) % 4, 3)
    frame = pd.DataFrame({**{c: product for c in AD_FEATURES},
                          'cat0': context, 'cat1': context,
                          **{c: context.astype(float) for c in NUMERICAL}})
    frame['user_id'] = np.repeat(np.arange(groups)+offset, 3)
    frame['impression_id'] = frame.user_id
    frame['product_id'] = product
    frame['click'] = (product == 1).astype(int)
    frame['displayrandom'] = np.repeat(np.arange(groups) % 2, 3)
    frame['rank'] = product
    return frame


def test_feature_contract_rejects_ad_like_context():
    frame = make_data(0, 20)
    transformer = FeatureTransformer.fit(frame)
    assert not any(feature_audit(frame, transformer.transform(frame))['context_variations_within_impression'].values())
    frame.loc[0, 'cat0'] = 19
    with pytest.raises(ValueError, match='feature roles'):
        feature_audit(frame, transformer.transform(frame))


def test_ablation_pipeline_never_opens_test_and_uses_explicit_subsets(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    for name, frame in [('train', make_data(0, 40)), ('validation', make_data(1000, 10))]:
        frame.to_parquet(data/f'{name}.parquet', index=False)
    (data/'test.parquet').write_bytes(b'NEVER OPEN THE TEST FILE')
    write_json(data/'manifest.json', {'files': {p.name: {'sha256': sha256(p)} for p in data.glob('*.parquet') if p.name != 'test.parquet'}})
    lgb_config = tmp_path/'lgb.json'
    write_json(lgb_config, {'params': {'objective': 'binary', 'metric': 'binary_logloss', 'num_threads': 1,
                                     'verbosity': -1, 'min_data_in_leaf': 1, 'min_data_per_group': 1, 'num_leaves': 3},
                            'num_boost_round': 3, 'early_stopping_rounds': 2})
    config = json.loads(Path('configs/context_ablation.json').read_text())
    config.update(lightgbm_config=str(lgb_config), seeds=[42], bootstrap_repeats=10)
    write_json(tmp_path/'config.json', config)
    result = run_experiment(data, tmp_path/'config.json', tmp_path/'models', tmp_path/'report')
    assert result['test_read'] is False and len(result['runs']) == 3
    for row in result['runs']:
        if row['variant'] == 'context_only':
            assert row['features'] == CONTEXT
            assert row['outer']['all']['ranking']['pair_accuracy'] == .5
            assert row['outer']['all']['score_discrimination']['all_scores_tied_fraction'] == 1
        if row['variant'] == 'ad_only':
            assert row['features'] == AD_FEATURES
        assert row['reload_max_abs_error'] == 0
        assert not set(row['features']) & {'user_id', 'product_id', 'click', 'rank', 'displayrandom'}
    assert not (tmp_path/'report/progress.json').exists()
    frozen = json.loads((tmp_path/'report/frozen_models.json').read_text())
    assert frozen['outer_loaded'] is False
