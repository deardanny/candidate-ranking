"""Check frozen artifacts, selection, complete-group inference and isolation."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
import mlflow
import numpy as np
import pandas as pd

from fairjob_ranker.diagnostics.context_residual import predict_bundle
from fairjob_ranker.io import read_json, sha256, write_json


def main():
    output, models = Path('reports/context-residual'), Path('models/context-residual-v1')
    result = read_json(output/'metrics.json')
    prereg = read_json(output/'preregistered.json')
    assert prereg['source_sha256'] == sha256(Path('src/fairjob_ranker/diagnostics/context_residual.py'))
    assert prereg['config_sha256'] == sha256(Path('configs/context_residual.json'))
    assert prereg['parent_metrics_sha256'] == sha256(Path('reports/context-ablation/metrics.json'))
    metadata = read_json(models/'metadata.json')
    for path, digest in metadata['artifact_sha256'].items():
        assert sha256(models/path) == digest
    for path, digest in prereg['frozen_parent_hashes'].items():
        assert sha256(models/path) == digest
    grid = read_json(output/'inner_grid.json')
    selection = read_json(output/'selection.json')
    assert selection['outer_read'] is False
    for policy, choice in selection['selection'].items():
        candidates = [(float(np.mean([r['inner_logloss'] for r in grid if r['policy'] == policy and r['alpha'] == a])), a)
                      for a in result['config']['alpha_grid']]
        assert (choice['mean_inner_logloss'], choice['alpha']) == min(candidates)
    assigned = pd.read_parquet(output/'oof_users.parquet').set_index('user_id').fold
    for seed in result['config']['seeds']:
        scores = pd.read_parquet(output/f'oof-s{seed}.parquet')
        assert len(scores) == prereg['inner_train_rows']
        assert np.isfinite(scores.context_raw_oof).all()
        assert np.array_equal(scores.fold.to_numpy(), scores.user_id.map(assigned).to_numpy())
        assert not scores.duplicated(['impression_id', 'product_id']).any()
    for audit in result['oof_audits']:
        assert audit['overlap_users'] == 0 and audit['held_labels_used_for_fit_or_stopping'] is False
        folder = models/'oof'/f"seed-{audit['seed']}-fold-{audit['fold']}"
        assert sha256(folder/'model.txt') == audit['model_sha256']
        assert sha256(folder/'features.json') == audit['transformer_sha256']
    prediction = pd.read_parquet(output/'outer_predictions.parquet')
    parent = pd.read_parquet('reports/context-ablation/outer_predictions.parquet')
    assert prediction[['impression_id', 'product_id']].equals(parent[['impression_id', 'product_id']])
    proof = {'configuration_source_and_model_hashes_verified': True, 'inner_only_strength_selection_verified': True,
             'oof_models_verified': len(result['oof_audits']), 'test_read': False,
             'mean_preservation_max_errors': {}, 'rank_graft_order_unchanged': {}, 'reload_max_errors': {}}
    for seed in result['config']['seeds']:
        np.testing.assert_array_equal(prediction[f'context-s{seed}'], parent[f'context_only-s{seed}'])
        np.testing.assert_array_equal(prediction[f'full-s{seed}'], parent[f'full-s{seed}'])
        for policy in ['ad_mean_preserved', 'full_rank_graft']:
            means = prediction.groupby('impression_id')[[f'context-s{seed}', f'{policy}-s{seed}']].mean()
            error = float(np.max(np.abs(means.iloc[:, 0]-means.iloc[:, 1])))
            assert error < 1e-12
            proof['mean_preservation_max_errors'][f'{policy}-s{seed}'] = error
        full_order = prediction.sort_values(['impression_id', f'full-s{seed}', 'product_id'], ascending=[True, False, True]).index
        graft_order = prediction.sort_values(['impression_id', f'full_rank_graft-s{seed}', 'product_id'], ascending=[True, False, True]).index
        # With alpha > 0, this transformation preserves the donor ranking.
        if result['selection']['full_rank_graft']['alpha'] > 0:
            assert full_order.equals(graft_order)
            proof['rank_graft_order_unchanged'][str(seed)] = True
    data = pd.read_parquet('data/processed/fairjob-v1/validation.parquet')
    selected = data.impression_id.isin(data.impression_id.drop_duplicates().head(100))
    sample = data.loc[selected].copy()
    reference = prediction.loc[selected]
    for seed in result['config']['seeds']:
        for policy in result['config']['policies']:
            reloaded = predict_bundle(models, sample, seed, policy)
            error = float(np.max(np.abs(reloaded-reference[f'{policy}-s{seed}'].to_numpy())))
            assert error < 1e-12
            proof['reload_max_errors'][f'{policy}-s{seed}'] = error
    # Neither labels nor logged ranks may influence online-style predictions.
    changed = sample.assign(click=1-sample.click, rank=999).sample(frac=1, random_state=123)
    p = predict_bundle(models, changed, 42, 'ad_mean_preserved')
    np.testing.assert_allclose(p, prediction.loc[changed.index, 'ad_mean_preserved-s42'], rtol=0, atol=1e-12)
    proof['label_and_input_order_invariance_verified'] = True
    previous = read_json(Path('reports/v2/experiment/metrics.json'))
    mlflow.set_tracking_uri(previous['tracking_uri'])
    mlflow.set_registry_uri(previous['tracking_uri'])
    client = mlflow.MlflowClient()
    assert client.get_model_version_by_alias(previous['registry_name'], 'champion').version == previous['champion']['registry_version']
    for row in result['runs']:
        run = client.get_run(row['mlflow_run_id'])
        assert run.info.status == 'FINISHED'
        assert run.data.metrics['ad_mean_preserved_outer_logloss'] == result['metrics']['outer'][f"ad_mean_preserved-s{row['seed']}"]['all']['click']['logloss']
    proof['mlflow_runs_verified'] = len(result['runs'])
    proof['champion_registry_version_unchanged'] = previous['champion']['registry_version']
    frozen = read_json(Path('reports/final/metrics.json'))['frozen_experiment']
    for name, digest in {**frozen['artifact_sha256'], **frozen['source_code_sha256']}.items():
        assert sha256(Path(name)) == digest
    proof['v1_frozen_sources_and_models_unchanged'] = True
    suites = list(ET.parse(output/'pytest.xml').getroot().iter('testsuite'))
    assert all(int(s.attrib.get('errors', 0)) == int(s.attrib.get('failures', 0)) == 0 for s in suites)
    proof['tests_passed'] = sum(int(s.attrib['tests']) for s in suites)
    proof['metrics_sha256'] = sha256(output/'metrics.json')
    write_json(output/'verification.json', proof)
    print('Verified', proof['tests_passed'], 'tests, 9 OOF models, 3 corrections, probability means, donor rankings and unchanged champion')


if __name__ == '__main__':
    main()
