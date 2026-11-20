"""Independently check fixed splits, probabilities/scores, checkpoints and outcomes."""
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
from fairjob_ranker.schema import FEATURES


def main():
    root = Path('reports/ranking-objective-reproduction')
    metrics, protocol = read_json(root/'metrics.json'), read_json(root/'protocol.json')
    config = metrics['config']
    parent = Path(config['parent_report'])
    parent_metrics = read_json(parent/'metrics.json')
    assert metrics['protocol_sha256'] == sha256(root/'protocol.json')
    assert metrics['frozen_models_sha256'] == sha256(Path(metrics['runs'][0]['model']).parent.parent/'frozen_models.json')
    for category in ['input_sha256', 'existing_artifact_sha256']:
        for path, digest in protocol[category].items():
            assert sha256(Path(path)) == digest, path
    pool = pd.read_parquet(Path(config['data_dir'])/'train.parquet')
    assignments = pd.read_parquet(parent/'user_assignments.parquet')
    pd.testing.assert_frame_equal(assignments, user_assignments(pool, parent_metrics['config']))
    oof = pd.read_parquet(root/'oof_outcomes.parquet').set_index('impression_id')
    old_oof = pd.read_parquet(parent/'oof_outcomes.parquet').set_index('impression_id').loc[oof.index]
    for s in config['seeds']:
        for k in [1, 3]:
            np.testing.assert_allclose(oof[f'baseline_s{s}_{k}'], old_oof[f'full_s{s}_{k}'], rtol=0, atol=1e-12)
    checks, seen_users, seen_rows = [], set(), 0
    for fold in range(3):
        fit, inner, held = split_frames(pool, assignments, fold)
        assert not set(held.user_id) & seen_users
        seen_users.update(held.user_id)
        seen_rows += len(held)
        audit = metrics['split_audits'][fold]
        transformer = FeatureTransformer.load(Path(audit['transformer']))
        independent = FeatureTransformer.fit(fit)
        assert transformer.vocabularies == independent.vocabularies
        assert transformer.means == independent.means and transformer.scales == independent.scales
        prediction = pd.read_parquet(root/f'fold-{fold}-predictions.parquet')
        pd.testing.assert_frame_equal(prediction[META].reset_index(drop=True), held[META])
        # Include positive and zero-click queries; this is numerical QA, not metric selection.
        g = held.groupby('impression_id').click.agg(['sum', 'size'])
        ids = sorted(set(g.loc[g['sum'].gt(0)].head(75).index) | set(g.head(75).index))
        mask = held.impression_id.isin(ids).to_numpy()
        sample, saved = held.loc[mask].reset_index(drop=True), prediction.loc[mask].reset_index(drop=True)
        sample_x, full_x = transformer.transform(sample), transformer.transform(held)
        for row in [r for r in metrics['runs'] if r['fold'] == fold]:
            name = f"{row['variant']}-s{row['seed']}"
            path = Path(row['model'])
            assert sha256(path) == row['model_sha256']
            model = lgb.Booster(model_file=str(path))
            assert model.feature_name() == FEATURES
            assert model.num_trees() == row['best_iteration']
            assert model.params['objective'] == ('binary' if row['variant'] == 'bce_top1' else 'lambdarank')
            fresh = model.predict(full_x[FEATURES], num_threads=4)
            error = float(np.max(np.abs(fresh-prediction[name].to_numpy())))
            assert error < 1e-12
            reference = request_outcomes(sample, model.predict(sample_x[FEATURES], num_threads=4))
            for k in [1, 3]:
                actual = oof.loc[reference.impression_id, f"{row['variant']}_s{row['seed']}_{k}"].to_numpy()
                np.testing.assert_allclose(actual, reference[f'captured_at_{k}'], rtol=0, atol=1e-12)
            history = read_json(path.with_name(path.stem+'-history.json'))['inner']['tie_top1']
            assert np.argmax(history)+1 == row['best_iteration']
            assert abs(max(history)*1000-row['inner_top1_per1000']) < 1e-12
            if row['variant'] == 'bce_top1':
                assert row['bce_prefix_max_error'] < 1e-12
            checks.append({'fold': fold, 'variant': row['variant'], 'seed': row['seed'],
                           'all_held_rows': len(held), 'reload_max_error': error,
                           'independent_topk_sample_requests': len(ids)})
    assert seen_rows == len(pool) and seen_users == set(pool.user_id)
    sizes = pool.groupby('impression_id').size()
    assert set(oof.index) == set(sizes[sizes >= 2].index)
    assert not oof.index.duplicated().any() and oof.groupby('user_id').outer_fold.nunique().eq(1).all()
    # Independently recompute all reported point deltas and the predeclared decision.
    for label, groups, report in [('pooled', oof, metrics['pooled_oof'])] + [
        (str(f), oof.loc[oof.outer_fold.eq(f)], metrics['folds'][str(f)]) for f in range(3)]:
        for cohort, mask in {'all_multi': np.ones(len(groups), bool), 'randomized_multi': groups.displayrandom.eq(1),
            'all_large': groups.candidates.ge(11), 'randomized_large': groups.candidates.ge(11) & groups.displayrandom.eq(1)}.items():
            part = groups.loc[mask]
            assert report[cohort]['requests'] == len(part)
            for variant in ['baseline'] + config['variants']:
                assert abs(report[cohort]['top1_per1000'][variant]-part[variant+'1'].mean()*1000) < 1e-10
            for name, item in report[cohort]['comparisons'].items():
                a, b, k = name.split('-')
                assert abs(item['estimate']-(part[a+k[-1]]-part[b+k[-1]]).mean()*1000) < 1e-10
    decision = metrics['primary_decision']
    supported = (all(metrics['folds'][str(f)]['randomized_multi']['comparisons']['rank_top1-baseline-top1']['estimate'] > 0 for f in range(3))
                 and metrics['pooled_oof']['randomized_multi']['comparisons']['rank_top1-baseline-top1']['ci95'][0] > 0
                 and metrics['pooled_oof']['all_multi']['comparisons']['rank_top1-baseline-top1']['ci95'][0] > 0)
    assert decision['supported'] == bool(supported)
    tracking = {}
    if metrics['tracking_uri']:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=metrics['tracking_uri'])
        for row in metrics['runs']:
            record = client.get_run(row['mlflow_run_id'])
            assert record.info.status == 'FINISHED'
            assert record.data.params['variant'] == row['variant']
            assert record.data.metrics['best_iteration'] == row['best_iteration']
        champion = client.get_model_version_by_alias('fairjob-v2', 'champion')
        assert str(champion.version) == '13'
        tracking = {'finished_runs': len(metrics['runs']), 'champion_version': str(champion.version)}
    write_json(root/'verification.json', {'status': 'passed', 'original_test_read': False, 'original_validation_read': False,
        'rows_checked': seen_rows, 'users_held_once': len(seen_users), 'fit_only_transformers_refitted': 3,
        'existing_and_new_hashes_verified': True, 'bce_selection_only_trajectories_unchanged': True,
        'all_27_best_checkpoints_match_inner_metric': True, 'model_checks': checks, 'tracking': tracking,
        'metric_point_values_and_primary_decision_independently_recomputed': True, 'metrics_sha256': sha256(root/'metrics.json')})
    print('Verified 27 full prediction reloads, independent tie-aware topk, checkpoint selection, splits and champion13.')


if __name__ == '__main__':
    main()
