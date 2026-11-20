"""Recheck HTTP promotion after moving registry I/O outside the request lock.

Reuses the identical-weight release fixture; never promotes the failed DCN.
"""
from datetime import datetime, timezone
import os
from pathlib import Path

os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
import httpx
import mlflow

from fairjob_ranker.io import read_json, sha256, write_json
from fairjob_ranker.replay import load_requests
from local_runtime import local_stack


def main():
    output = Path('reports/v2')
    fixture = output/'release_rehearsal'
    controller = read_json(fixture/'controller.json')
    experiment = read_json(output/'experiment/metrics.json')
    source = Path('src/fairjob_ranker/v2/deployment.py')
    measured_source = output/'load/source/deployment.py'
    load = read_json(output/'load/metrics.json')
    assert sha256(measured_source) == load['source_code_sha256'][str(source)]
    requests, _ = load_requests(Path('data/replay/validation.jsonl'))
    mlflow.set_tracking_uri(controller['tracking_uri'])
    mlflow.set_registry_uri(controller['tracking_uri'])
    registry = mlflow.MlflowClient()
    env = {'FAIRJOB_V2_EXPERIMENT': str(fixture/'controller.json'),
           'FAIRJOB_V2_SHADOW': '', 'FAIRJOB_MAX_INFLIGHT': '8',
           'FAIRJOB_ADMIN_TOKEN': 'local-control-verification-only'}
    proof = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'change': 'Registry I/O and model preparation run outside the active-request snapshot lock; control transitions remain serialized.',
             'load_scope': '54 load runs preceded this control-only change. Scoring and scheduling code are unchanged. No claim that those runs measured the new control lock.',
             'measured_source': str(measured_source),
             'measured_sha256': sha256(measured_source),
             'current_sha256': sha256(source),
             'regression_test': 'tests/test_v2_serving.py',
             'fixture_scope': 'Identical-weight package revision; actual DCN stays blocked'}
    with local_stack(Path('models/pytorch-v1'), 8032, 6398,
                     Path('runs/v2/control-verification/logs'),
                     environment_overrides=env, api_app='v2_server:app') as stack:
        with httpx.Client(base_url=stack.api_url, timeout=10, trust_env=False) as client:
            first = client.post('/rank', json=requests[0])
            first.raise_for_status()
            first = first.json()
            headers = {'Authorization': 'Bearer '+env['FAIRJOB_ADMIN_TOKEN']}
            response = client.post('/admin/promote', headers=headers, json={
                'bundle': controller['challenger']['bundle'],
                'quality_evidence': str(fixture/'quality.json'),
                'performance_evidence': str(fixture/'performance.json'),
                'warmup': requests[0]})
            response.raise_for_status()
            proof['promotion'] = response.json()
            try:
                promoted = client.post('/rank', json=requests[0]).json()
                assert promoted['model_version'] == controller['challenger']['model_version']
                assert promoted['rankings'] == first['rankings']
                assert registry.get_model_version_by_alias(controller['registry_name'], 'champion').version == controller['challenger']['registry_version']
            finally:
                response = client.post('/admin/rollback', headers=headers)
                response.raise_for_status()
                proof['rollback'] = response.json()
            reverted = client.post('/rank', json=requests[0]).json()
            assert reverted['model_version'] == first['model_version']
            assert reverted['rankings'] == first['rankings']
            assert registry.get_model_version_by_alias(controller['registry_name'], 'champion').version == controller['champion']['registry_version']
    assert registry.get_model_version_by_alias(experiment['registry_name'], 'champion').version == experiment['champion']['registry_version']
    proof['real_champion_unchanged'] = True
    proof['http_promotion_rollback_verified'] = True
    write_json(output/'control_hardening.json', proof)
    print('Updated control locks: real HTTP promotion/rollback verified; scoring unchanged')


if __name__ == '__main__':
    main()
