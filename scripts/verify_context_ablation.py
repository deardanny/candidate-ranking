"""Independently check saved predictions, model provenance and registry isolation."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from fairjob_ranker.io import read_json, sha256, write_json


def main():
    output = Path('reports/context-ablation')
    results = read_json(output/'metrics.json')
    prereg = read_json(output/'preregistered.json')
    assert prereg['source_sha256'] == sha256(Path('src/fairjob_ranker/diagnostics/context_ablation.py'))
    assert prereg['config_sha256'] == sha256(Path('configs/context_ablation.json'))
    frozen = read_json(output/'frozen_models.json')
    for row in results['runs']:
        assert sha256(Path(row['model'])) == frozen['models'][row['name']] == row['model_sha256']
    predictions = pd.read_parquet(output/'outer_predictions.parquet')
    groups = pd.read_parquet(output/'outer_ranking.parquet')
    # Independent library implementation on every eligible impression.
    proof = {'model_hashes_verified': len(results['runs']), 'test_read': False,
             'source_and_configuration_match_preregistration': True, 'sklearn_crosscheck': {}}
    for model in ['context_only-s42', 'ad_only-s42', 'full-s42']:
        ranked = groups.loc[groups.model == model].set_index('impression_id')
        errors = []
        for impression, frame in predictions.loc[predictions.impression_id.isin(ranked.index)].groupby('impression_id'):
            value = ndcg_score(frame.click.to_numpy()[None, :], frame[model].to_numpy()[None, :], k=5, ignore_ties=False)
            errors.append(abs(value-ranked.loc[impression, 'ndcg_tie_averaged']))
        assert max(errors) < 1e-12
        proof['sklearn_crosscheck'][model] = {'impressions': len(errors), 'max_absolute_error': max(errors)}
    context_scores = predictions.groupby('impression_id')['context_only-s42'].nunique()
    assert context_scores.eq(1).all()
    proof['context_all_impressions_tied'] = True
    v1 = read_json(Path('reports/final/metrics.json'))['frozen_experiment']
    for name, digest in {**v1['artifact_sha256'], **v1['source_code_sha256']}.items():
        assert sha256(Path(name)) == digest
    proof['v1_frozen_sources_and_models_unchanged'] = True
    previous = read_json(Path('reports/v2/experiment/metrics.json'))
    mlflow.set_tracking_uri(previous['tracking_uri'])
    mlflow.set_registry_uri(previous['tracking_uri'])
    client = mlflow.MlflowClient()
    active = client.get_model_version_by_alias(previous['registry_name'], 'champion')
    assert active.version == previous['champion']['registry_version']
    assert active.tags['bundle_metadata_sha256'] == sha256(Path(previous['champion']['bundle'])/'metadata.json')
    proof['champion_registry_version_unchanged'] = active.version
    experiment = client.get_experiment_by_name(results['config']['experiment_name'])
    runs = {run.info.run_id: run for run in client.search_runs([experiment.experiment_id], max_results=100)}
    for row in results['runs']:
        assert runs[row['mlflow_run_id']].info.status == 'FINISHED'
        assert runs[row['mlflow_run_id']].data.metrics['outer_ndcg_tie_averaged'] == row['outer']['all']['ranking']['ndcg_tie_averaged']
    proof['mlflow_finished_runs_verified'] = len(results['runs'])
    suite = list(ET.parse(output/'pytest.xml').getroot().iter('testsuite'))
    assert all(int(s.attrib.get('errors', 0)) == int(s.attrib.get('failures', 0)) == 0 for s in suite)
    proof['tests_passed'] = sum(int(s.attrib['tests']) for s in suite)
    proof['prediction_sha256'] = sha256(output/'outer_predictions.parquet')
    proof['metrics_sha256'] = sha256(output/'metrics.json')
    write_json(output/'verification.json', proof)
    print(proof)


if __name__ == '__main__':
    main()
