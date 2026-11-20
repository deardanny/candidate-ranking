"""Audit source-user exclusion, saved encoders, cold attribution and model replay."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
import lightgbm as lgb
import numpy as np
import pandas as pd

from fairjob_ranker.diagnostics.business_validation import request_outcomes, topk_weights
from fairjob_ranker.diagnostics.candidate_count_reproduction import META, split_frames, user_assignments
from fairjob_ranker.diagnostics.response_statistics import (
    KEY_COLUMNS, TABLE_KEYS, ResponseStatistics, cross_fit_statistics, statistic_columns,
)
from fairjob_ranker.diagnostics.response_statistics_experiment import cold_status
from fairjob_ranker.features import FeatureTransformer
from fairjob_ranker.io import read_json, sha256, write_json
from fairjob_ranker.schema import CATEGORICAL, FEATURES


def main():
    root = Path('reports/response-statistics')
    m, p = read_json(root/'metrics.json'), read_json(root/'protocol.json')
    cfg = m['config']
    parent = Path(cfg['parent_report'])
    parent_m = read_json(parent/'metrics.json')
    assert sha256(root/'protocol.json') == m['protocol_sha256']
    for key in ['input_sha256', 'existing_artifact_sha256']:
        for file, digest in p[key].items():
            assert sha256(Path(file)) == digest, file
    pool = pd.read_parquet(Path(cfg['data_dir'])/'train.parquet')
    assignments = pd.read_parquet(parent/'user_assignments.parquet')
    pd.testing.assert_frame_equal(assignments, user_assignments(pool, parent_m['config']))
    oof = pd.read_parquet(root/'oof_outcomes.parquet').set_index('impression_id')
    previous = pd.read_parquet(parent/'oof_outcomes.parquet').set_index('impression_id').loc[oof.index]
    for seed in cfg['seeds']:
        for k in [1, 3]:
            np.testing.assert_array_equal(oof[f'baseline_s{seed}_{k}'], previous[f'full_s{seed}_{k}'])
    seen, seen_rows, checks, stats_checks = set(), 0, [], []
    for fold in range(3):
        fit, inner, held = split_frames(pool, assignments, fold)
        assert not seen & set(held.user_id)
        seen.update(held.user_id)
        seen_rows += len(held)
        audit = m['split_audits'][fold]
        folder = Path(next(r['model'] for r in m['runs'] if r['fold'] == fold)).parent
        assert sha256(folder/'fit-response-oof.parquet') == audit['oof_stats_sha256']
        assert sha256(folder/'encoding-users.parquet') == audit['encoding_users_sha256']
        assert sha256(folder/'response-store/metadata.json') == audit['response_store_metadata_sha256']
        saved_fit = pd.read_parquet(folder/'fit-response-oof.parquet')
        recomputed, enc_users, enc_audits = cross_fit_statistics(fit, cfg, fold)
        pd.testing.assert_frame_equal(saved_fit, recomputed, check_exact=True)
        pd.testing.assert_frame_equal(enc_users, pd.read_parquet(folder/'encoding-users.parquet'))
        assert enc_audits == audit['encoding_audits']
        assert sum(a['target_rows'] for a in enc_audits) == len(fit)
        store = ResponseStatistics.load(folder/'response-store')
        assert store.rows == len(fit) and store.clicks == int(fit.click.sum())
        for keys in TABLE_KEYS:
            expected_n = fit[list(keys)].value_counts(sort=False).sort_index()
            expected_c = fit.groupby(list(keys)).click.sum().sort_index()
            actual = store.tables[keys].sort_index()
            # value_counts uses a one-level MultiIndex even for a single column.
            np.testing.assert_array_equal(actual.n.to_numpy(), expected_n.to_numpy())
            pd.testing.assert_series_equal(actual.c, expected_c, check_names=False)
        transformer = FeatureTransformer.load(Path(audit['transformer']))
        refit = FeatureTransformer.fit(fit)
        assert transformer.vocabularies == refit.vocabularies
        assert transformer.means == refit.means and transformer.scales == refit.scales
        prediction = pd.read_parquet(root/f'fold-{fold}-predictions.parquet')
        pd.testing.assert_frame_equal(held[META], prediction[META].reset_index(drop=True))
        status = cold_status(held, fit, inner)
        pd.testing.assert_frame_equal(status, prediction[['fit_unseen', 'strict_unseen']].reset_index(drop=True))
        cold_products = set(held.loc[status.strict_unseen, 'product_id'])
        assert not cold_products & (set(fit.product_id) | set(inner.product_id))
        stats = store.transform(held[KEY_COLUMNS])
        # All unseen IDs use exactly attribute-based fallback, not accidental ID lookup.
        for context in ['', '__cat0', '__cat1']:
            assert stats.loc[status.fit_unseen, f'rs__product_id{context}__log_n'].eq(0).all()
            parent_rate = stats.loc[status.fit_unseen, [f'rs__cat{i}{context}__p' for i in range(2, 13)]].mean(axis=1)
            np.testing.assert_allclose(stats.loc[status.fit_unseen, f'rs__product_id{context}__p'], parent_rate, rtol=2e-6)
        x = pd.concat([transformer.transform(held), stats], axis=1)
        request_info = prediction.assign(cold_click=prediction.click*prediction.strict_unseen).groupby('impression_id').agg(
            cold=('strict_unseen', 'sum'), cold_click=('cold_click', 'sum'), clicks=('click', 'sum'))
        sample_ids = sorted(set(request_info.loc[request_info.cold_click.gt(0)].head(60).index)
            | set(request_info.loc[request_info.cold.gt(0)].head(40).index) | set(request_info.loc[request_info.clicks.gt(0)].head(40).index))
        sample_mask = prediction.impression_id.isin(sample_ids)
        sample = prediction.loc[sample_mask].reset_index(drop=True)
        for row in [r for r in m['runs'] if r['fold'] == fold]:
            path = Path(row['model'])
            assert sha256(path) == row['model_sha256']
            model = lgb.Booster(model_file=str(path))
            columns = FEATURES+statistic_columns(row['variant'])
            assert model.feature_name() == columns
            name = f"{row['variant']}-s{row['seed']}"
            scores = model.predict(x[columns], num_threads=4)
            error = float(np.max(np.abs(scores-prediction[name])))
            assert error < 1e-12
            reference = request_outcomes(sample, sample[name])
            for k in [1, 3]:
                np.testing.assert_allclose(oof.loc[reference.impression_id, f"{row['variant']}_s{row['seed']}_{k}"],
                                           reference[f'captured_at_{k}'], atol=1e-12, rtol=0)
            for impression, part in sample.groupby('impression_id'):
                if len(part) < 2:
                    continue
                weights = topk_weights(part[name], 1)
                expected = float(weights @ (part.click*part.strict_unseen))
                actual = oof.loc[impression, f"{row['variant']}_s{row['seed']}_cold1"]
                assert abs(actual-expected) < 1e-12
                selected = float(weights @ part.strict_unseen)
                assert abs(oof.loc[impression, f"{row['variant']}_s{row['seed']}_cold_selected1"]-selected) < 1e-12
            history = read_json(path.with_name(path.stem+'-history.json'))['inner']['binary_logloss']
            assert np.argmin(history)+1 == row['best_iteration']
            checks.append({'fold': fold, 'variant': row['variant'], 'seed': row['seed'],
                'full_held_prediction_max_error': error, 'independent_requests': len(sample_ids)})
        stats_checks.append({'fold': fold, 'oof_rows_recomputed': len(saved_fit), 'user_exclusion_folds': len(enc_audits),
            'full_fit_tables_independently_counted': len(TABLE_KEYS), 'strict_unseen_products': len(cold_products)})
        # One deterministic end-to-end training replay proves the saved booster uses OOF X.
        if fold == 0:
            row = next(r for r in m['runs'] if r['fold'] == 0 and r['variant'] == 'ad_context_stats' and r['seed'] == 42)
            baseline = read_json(Path(cfg['baseline_config']))
            columns = FEATURES+statistic_columns('ad_context_stats')
            xf = pd.concat([transformer.transform(fit), saved_fit], axis=1)
            xi = pd.concat([transformer.transform(inner), store.transform(inner[KEY_COLUMNS])], axis=1)
            train = lgb.Dataset(xf[columns], label=fit.click, categorical_feature=CATEGORICAL)
            valid = lgb.Dataset(xi[columns], label=inner.click, categorical_feature=CATEGORICAL, reference=train)
            replay = lgb.train(baseline['params'], train, num_boost_round=baseline['num_boost_round'], valid_sets=[valid],
                valid_names=['inner'], callbacks=[lgb.early_stopping(baseline['early_stopping_rounds'], verbose=False)])
            original = lgb.Booster(model_file=row['model'])
            assert replay.best_iteration == row['best_iteration']
            replay_error = float(np.max(np.abs(replay.predict(xi[columns], num_threads=4)-original.predict(xi[columns], num_threads=4))))
            assert replay_error < 1e-12
        print(f'Verified fold{fold}: five out-of-user encodings, full-fit tables, six models and cold attribution.', flush=True)
    assert seen_rows == len(pool) and seen == set(pool.user_id)
    sizes = pool.groupby('impression_id').size()
    assert set(oof.index) == set(sizes[sizes >= 2].index)
    assert oof.groupby('user_id').outer_fold.nunique().eq(1).all()
    frozen = Path(m['runs'][0]['model']).parent.parent/'frozen_models.json'
    assert sha256(frozen) == m['frozen_models_sha256']
    tracking = {}
    if m['tracking_uri']:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=m['tracking_uri'])
        for row in m['runs']:
            record = client.get_run(row['mlflow_run_id'])
            assert record.info.status == 'FINISHED'
            assert record.data.params['encoding_split_unit'] == 'user'
            assert record.data.metrics['best_iteration'] == row['best_iteration']
        champion = client.get_model_version_by_alias('fairjob-v2', 'champion')
        assert str(champion.version) == '13'
        tracking = {'finished_runs': len(m['runs']), 'champion_version': str(champion.version)}
    write_json(root/'verification.json', {'status': 'passed', 'metrics_sha256': sha256(root/'metrics.json'),
        'original_test_read': False, 'original_validation_read': False, 'rows_evaluated_once': seen_rows,
        'users_evaluated_once': len(seen), 'statistics_checks': stats_checks, 'model_checks': checks,
        'one_deterministic_training_replay_max_error': replay_error, 'replay_is_verification_not_an_extra_selected_model': True,
        'prior_artifacts_unchanged': True, 'tracking': tracking})
    print('Verified all18 models,15 user-excluded encoding folds,114 full-fit count tables and one exact training replay.')


if __name__ == '__main__':
    main()
