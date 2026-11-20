"""Replay frozen test candidates and verify real API parity and failure semantics."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from fairjob_ranker.final_evaluation import verify_frozen
from fairjob_ranker.io import read_json, write_json, sha256
from fairjob_ranker.replay import load_requests
from fairjob_ranker.torch_model import load_bundle
from local_runtime import local_stack, stop_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, default=Path('reports/frozen_experiment.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/week3/final_serving_verification.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = verify_frozen(args.frozen)
    requests, manifest = load_requests(Path('data/replay/test.jsonl'))
    assert manifest['frozen_experiment_sha256'] == sha256(args.frozen)
    scores = pd.read_parquet('reports/final/test_predictions.parquet').set_index(['impression_id','product_id'])
    bundle = load_bundle(Path(frozen['pytorch_dir']))
    settings = frozen['serving_settings']
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'frozen_sha256': sha256(args.frozen),
              'test_requests': 100, 'max_abs_error': 0.0, 'input_order_invariant': True}
    with local_stack(Path(frozen['pytorch_dir']), 8020, 6395, Path('runs/final-parity'),
                     environment_overrides=settings['environment'], uvicorn_workers=settings['uvicorn_workers']) as stack:
        with httpx.Client(base_url=stack.api_url, timeout=5, trust_env=False) as client:
            for request in requests[:100]:
                response = client.post('/rank', json=request)
                response.raise_for_status()
                body = response.json()
                assert not body['degraded']
                impression = int(request['request_id'].split('-')[-1])
                reference = [scores.loc[(impression,r['product_id']), 'pytorch_score'] for r in body['rankings']]
                actual = [r['score'] for r in body['rankings']]
                np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)
                report['max_abs_error'] = max(report['max_abs_error'], max(abs(a-b) for a,b in zip(actual,reference)))
                reverse = client.post('/rank', json={**request,'candidate_ids':request['candidate_ids'][::-1]})
                reverse.raise_for_status()
                assert [r['product_id'] for r in body['rankings']] == [r['product_id'] for r in reverse.json()['rankings']]
            example = read_json(Path('reports/final/demo_example.json'))['request']
            before = client.post('/rank', json=example).json()
            stop_process(stack.redis_process)
            after = client.post('/rank', json=example).json()
            assert after['degraded'] and 'redis_unavailable' in after['fallback_reason']
            assert before['rankings'] == after['rankings']
            report.update(example_request=example, normal_response=before, redis_down_response=after,
                          redis_down_same_scores=True, readiness_after_redis_stop=client.get('/readyz').json())
    with local_stack(Path(frozen['pytorch_dir']), 8020, 6395, Path('runs/final-model-fault'),
                     environment_overrides={**settings['environment'],'FAIRJOB_FAILURE_MODE':'model_error'},
                     api_app='fault_server:app') as stack:
        response = httpx.post(stack.api_url+'/rank', json=requests[0], timeout=5, trust_env=False)
        response.raise_for_status()
        body = response.json()
        assert body['scoring_method'] == 'smoothed_ad_ctr' and 'model_inference_error' in body['fallback_reason']
        for row in body['rankings']:
            assert row['score'] == bundle.fallback['scores'].get(str(row['product_id']), bundle.fallback['train_ctr'])
        report['model_failure_response'] = body
        report['model_failure_exact_train_fallback'] = True
    write_json(args.output, report)
    print('PASS: frozen test parity, input permutations, Redis outage, exact model-failure fallback')


if __name__ == '__main__':
    main()
