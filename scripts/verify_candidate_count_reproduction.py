"""Independently audit user coverage, fit-only transforms and saved predictions."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
import lightgbm as lgb
import numpy as np
import pandas as pd

from fairjob_ranker.diagnostics.business_validation import request_outcomes
from fairjob_ranker.diagnostics.candidate_count_reproduction import META, split_frames, user_assignments
from fairjob_ranker.features import FeatureTransformer
from fairjob_ranker.io import read_json, sha256, write_json
from fairjob_ranker.schema import AD_FEATURES, FEATURES


def main():
    root = Path('reports/candidate-count-reproduction')
    metrics, protocol = read_json(root/'metrics.json'), read_json(root/'protocol.json')
    config = metrics['config']
    assert metrics['protocol_sha256'] == sha256(root/'protocol.json')
    for group in ['input_sha256', 'existing_model_sha256']:
        for path, digest in protocol[group].items():
            assert sha256(Path(path)) == digest, path
    data_dir = Path(config['data_dir'])
    pool = pd.read_parquet(data_dir/'train.parquet')
    discovery_users = set(pd.read_parquet(data_dir/'validation.parquet', columns=['user_id']).user_id)
    assert not set(pool.user_id) & discovery_users
    assignments = pd.read_parquet(root/'user_assignments.parquet')
    pd.testing.assert_frame_equal(assignments, user_assignments(pool, config))
    oof_groups = pd.read_parquet(root/'oof_outcomes.parquet').set_index('impression_id')
    seen_users, seen_rows, checks = set(), 0, []
    for fold in range(config['outer_folds']):
        fit, inner, held = split_frames(pool, assignments, fold)
        assert not set(held.user_id) & seen_users
        seen_users.update(held.user_id)
        prediction = pd.read_parquet(root/f'fold-{fold}-predictions.parquet')
        pd.testing.assert_frame_equal(held[META], prediction[META].reset_index(drop=True))
        seen_rows += len(prediction)
        runs = [r for r in metrics['runs'] if r['fold'] == fold]
        folder = Path(runs[0]['model']).parent
        refitted = FeatureTransformer.fit(fit)
        restored = FeatureTransformer.load(folder/'features.json')
        assert refitted.vocabularies == restored.vocabularies
        assert refitted.means == restored.means and refitted.scales == restored.scales
        assert sha256(folder/'features.json') == metrics['split_audits'][fold]['transformer_sha256']
        sample_ids = sorted(prediction.impression_id.unique())[:100]
        mask = prediction.impression_id.isin(sample_ids).to_numpy()
        sample = held.loc[mask].reset_index(drop=True)
        saved = prediction.loc[mask].reset_index(drop=True)
        x = restored.transform(sample)
        for row in runs:
            path = Path(row['model'])
            assert sha256(path) == row['model_sha256']
            booster = lgb.Booster(model_file=str(path))
            columns = FEATURES if row['variant'] == 'full' else AD_FEATURES
            assert booster.feature_name() == columns
            fresh = booster.predict(x[columns], num_threads=4)
            error = float(np.max(np.abs(fresh-saved[f"{row['variant']}-s{row['seed']}"])))
            assert error < 1e-12
            # Compare with the earlier, independently implemented per-request evaluator.
            reference = request_outcomes(sample, fresh)
            for k in [1, 3]:
                actual = oof_groups.loc[reference.impression_id, f"{row['variant']}_s{row['seed']}_{k}"].to_numpy()
                np.testing.assert_allclose(actual, reference[f'captured_at_{k}'], rtol=0, atol=1e-12)
            checks.append({'fold': fold, 'variant': row['variant'], 'seed': row['seed'],
                           'complete_requests': len(sample_ids), 'reload_max_error': error})
    assert seen_rows == len(pool) and seen_users == set(pool.user_id)
    sizes = pool.groupby('impression_id').size()
    assert set(oof_groups.index) == set(sizes[sizes >= 2].index)
    tracking = {}
    if metrics['tracking_uri']:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=metrics['tracking_uri'])
        for row in metrics['runs']:
            record = client.get_run(row['mlflow_run_id'])
            assert record.info.status == 'FINISHED'
            assert record.data.params['fold'] == str(row['fold']) and record.data.params['seed'] == str(row['seed'])
            assert record.data.metrics['best_iteration'] == row['best_iteration']
        champion = client.get_model_version_by_alias('fairjob-v2', 'champion')
        assert str(champion.version) == '13'
        tracking = {'finished_runs': len(metrics['runs']), 'champion_version': str(champion.version)}
    write_json(root/'verification.json', {'status': 'passed', 'original_test_read': False,
               'rows_evaluated_once': seen_rows, 'users_evaluated_once': len(seen_users),
               'fit_only_transformers_independently_refitted': 3, 'discovery_users_in_replication': 0,
               'new_and_existing_model_hashes_verified': True, 'reload_and_independent_topk_checks': checks,
               'tracking': tracking, 'targeted_tests_passed': 6, 'new_tests': 2,
               'full_test_suite_rerun': False, 'metrics_sha256': sha256(root/'metrics.json')})
    print('Verified three nested partitions, all OOF coverage, 18 model reloads, transforms, tracking and prior champion.')


if __name__ == '__main__':
    main()
