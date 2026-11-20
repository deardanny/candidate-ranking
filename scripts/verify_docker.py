"""Build/start an isolated Compose project; compare real container scores and failure behavior."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import numpy as np

from fairjob_ranker.io import write_json, sha256
from fairjob_ranker.replay import load_requests
from fairjob_ranker.torch_model import load_bundle
from fairjob_ranker.catalog import AdCatalog
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/week3/docker_verification.json'))
    parser.add_argument('--port', type=int, default=8003)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    env = os.environ.copy()
    env['FAIRJOB_HTTP_PORT'] = str(args.port)
    command = [sys.executable, 'scripts/docker_project.py']
    compose = [*command, 'compose', '-p', 'fairjob-verification']
    def run(argv, **kwargs):
        return subprocess.run(argv, env=env, check=True, text=True, **kwargs)
    existing = run([*compose, 'ps', '-q'], capture_output=True).stdout.strip()
    if existing:
        raise RuntimeError('Verification project already has containers; stop it explicitly before rerunning')
    requests, manifest = load_requests(Path('data/replay/validation.jsonl'))
    bundle = load_bundle(Path('models/pytorch-v1'))
    catalog = AdCatalog(bundle.directory / 'ad_catalog.parquet', bundle.metadata['catalog_version'])
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'corpus': manifest,
              'requests': 100, 'max_abs_error': 0., 'compose_sha256': sha256(Path('compose.yaml')),
              'dockerfile_sha256': sha256(Path('Dockerfile')), 'cpu_limit': 2, 'memory_limit_bytes': 2 * 2**30,
              'model_version': bundle.metadata['model_version']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / 'docker-build.log').open('w') as log:
        run([*compose, 'build'], stdout=log, stderr=subprocess.STDOUT)
    try:
        run([*compose, 'up', '-d', '--wait', '--wait-timeout', '90'])
        report['docker_version'] = json.loads(run([*command, 'version', '--format', '{{json .}}'], capture_output=True).stdout)
        container = run([*compose, 'ps', '-q', 'api'], capture_output=True).stdout.strip()
        inspect = json.loads(run([*command, 'inspect', container], capture_output=True).stdout)[0]
        report['image_id'] = inspect['Image']
        report['user'] = inspect['Config']['User']
        report['read_only_rootfs'] = inspect['HostConfig']['ReadonlyRootfs']
        report['container_architecture'] = run([*compose, 'exec', '-T', 'api', 'uname', '-m'], capture_output=True).stdout.strip()
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}', timeout=10, trust_env=False) as client:
            for request in requests[:100]:
                raw = pd.DataFrame([{**catalog.records[i], 'cat0': request['context']['cat0'],
                                     'cat1': request['context']['cat1'], **request['context']['numeric']}
                                    for i in request['candidate_ids']])
                expected = dict(zip(request['candidate_ids'], bundle.predict(raw)))
                response = client.post('/rank', json=request)
                response.raise_for_status()
                body = response.json()
                assert not body['degraded'], body
                actual = [r['score'] for r in body['rankings']]
                reference = [expected[r['product_id']] for r in body['rankings']]
                np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)
                report['max_abs_error'] = max(report['max_abs_error'], max(abs(a-b) for a,b in zip(actual,reference)))
            request = requests[0]
            before = client.post('/rank', json=request).json()
            run([*compose, 'stop', 'redis'])
            after = client.post('/rank', json=request).json()
            assert after['degraded'] and 'redis_unavailable' in after['fallback_reason']
            assert before['rankings'] == after['rankings']
            report['redis_down_same_scores'] = True
            run([*compose, 'start', 'redis'])
            run([*compose, 'exec', '-T', 'api', 'python', '-c',
                 'from fairjob_ranker.catalog import *; from fairjob_ranker.torch_model import load_bundle; from pathlib import Path; b=load_bundle(Path("/model")); c=redis_client("redis://redis:6379/0",2); print(seed_catalog(AdCatalog(Path("/model/ad_catalog.parquet"), b.metadata["catalog_version"]),c)); c.close()'])
            recovered = client.post('/rank', json=request).json()
            assert not recovered['degraded']
            assert before['rankings'] == recovered['rankings']
            report['redis_recovered_same_scores'] = True
            report['readiness'] = client.get('/readyz').json()
        write_json(args.output, report)
        print('PASS: real Linux containers, 100 requests, Redis stop/restart, max error', report['max_abs_error'])
    finally:
        run([*compose, 'down', '--remove-orphans'])


if __name__ == '__main__':
    main()
