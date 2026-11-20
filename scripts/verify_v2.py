"""Real HTTP parity, isolated shadow faults, and failed quality promotion."""
import asyncio
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

import httpx
import mlflow
import numpy as np
import pandas as pd

from fairjob_ranker.io import read_json,sha256,write_json
from fairjob_ranker.replay import load_requests
from fairjob_ranker.schema import FEATURES
from fairjob_ranker.v2.load import run_load
from fairjob_ranker.v2.models import load_bundle
from local_runtime import local_stack,stop_process


def main():
    output=Path('reports/v2/verification')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    experiment=read_json(Path('reports/v2/experiment/metrics.json'))
    requests,corpus=load_requests(Path('data/replay/validation.jsonl'))
    champion,candidate=experiment['champion'],experiment['challenger']
    bundle=load_bundle(champion['bundle'])
    catalog=pd.read_parquet(Path(champion['bundle'])/'ad_catalog.parquet').set_index('product_id')
    proof={'created_at_utc':datetime.now(timezone.utc).isoformat(),'corpus':corpus,'parity_requests':100}
    env={'FAIRJOB_SHADOW_RATE':'1','FAIRJOB_SHADOW_CAPACITY':'16','FAIRJOB_MAX_INFLIGHT':'8',
         'FAIRJOB_ADMIN_TOKEN':'local-verification-only','MLFLOW_DISABLE_TELEMETRY':'true','MLFLOW_DISABLE_AGENT_HINT':'1'}
    max_error=0.
    with local_stack(Path('models/pytorch-v1'),8031,6397,output/'logs'/'normal',environment_overrides=env,api_app='v2_server:app') as stack:
        with httpx.Client(base_url=stack.api_url,timeout=5,trust_env=False) as client:
            for request in requests[:100]:
                context={'cat0':request['context']['cat0'],'cat1':request['context']['cat1'],**request['context']['numeric']}
                raw=pd.DataFrame([{**catalog.loc[i].to_dict(),**context} for i in request['candidate_ids']])
                scores=bundle.predict(raw[FEATURES])
                response=client.post('/rank',json=request);response.raise_for_status(); normal=response.json()
                mapping={r['product_id']:r['score'] for r in normal['rankings']}
                actual=np.array([mapping[i] for i in request['candidate_ids']])
                np.testing.assert_allclose(actual,scores,atol=1e-6,rtol=1e-5)
                max_error=max(max_error,float(np.max(np.abs(actual-scores))))
            first=requests[0]
            normal=client.post('/rank',json=first).json()
            reversed_request={**first,'candidate_ids':list(reversed(first['candidate_ids']))}
            reverse=client.post('/rank',json=reversed_request).json()
            assert normal['rankings']==reverse['rankings']
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                shadow=client.get('/shadow/status').json()
                if shadow['compared']+shadow['error']==shadow['submitted']:break
                time.sleep(.1)
            assert shadow['compared']>0 and shadow['error']==0
            proof.update(max_abs_error=max_error,input_order_invariant=True,shadow=shadow)
            metric_text=client.get('/metrics').text
            assert 'fairjob_shadow_top1_total' in metric_text
            (output/'prometheus.txt').write_text(metric_text)
            stop_process(stack.redis_process)
            degraded=client.post('/rank',json=first).json()
            assert degraded['rankings']==normal['rankings'] and degraded['degraded']
            proof['redis_outage_same_scores']=True
    # Candidate serving evidence is generated independently of its failed quality gate.
    candidate_env={**env,'FAIRJOB_V2_ACTIVE':candidate['bundle'],'FAIRJOB_V2_SHADOW':'','FAIRJOB_SHADOW_RATE':'0'}
    with local_stack(Path('models/pytorch-v1'),8031,6397,output/'logs'/'candidate',environment_overrides=candidate_env,api_app='v2_server:app') as stack:
        asyncio.run(run_load(stack.api_url,requests,50,3,expected_version=candidate['model_version']))
        measured=asyncio.run(run_load(stack.api_url,requests,50,10,expected_version=candidate['model_version'],pids=[stack.api_process.pid,stack.redis_process.pid]))
        measured.pop('records')
        measured.update(candidate_metadata_sha256=sha256(Path(candidate['bundle'])/'metadata.json'),evaluation_split='validation')
        write_json(output/'candidate_performance.json',measured)
    with local_stack(Path('models/pytorch-v1'),8031,6397,output/'logs'/'blocked',environment_overrides=env,api_app='v2_server:app') as stack:
        with httpx.Client(base_url=stack.api_url,timeout=10,trust_env=False) as client:
            before=client.get('/readyz').json()
            response=client.post('/admin/promote',headers={'Authorization':'Bearer '+env['FAIRJOB_ADMIN_TOKEN']},json={
                'bundle':candidate['bundle'],'quality_evidence':'reports/v2/experiment/quality_gate.json',
                'performance_evidence':str(output/'candidate_performance.json'),'warmup':requests[0]})
            assert response.status_code==409 and 'quality_gate_failed' in response.text
            assert client.get('/readyz').json()['model_version']==before['model_version']
            proof['promotion_blocked']=response.json()
    # Slow or broken shadow must not turn an active request into an error.
    for label,extras in [('slow',{'TEST_SHADOW_DELAY':'.2'}),('error',{'TEST_SHADOW_ERROR':'1'})]:
        fault_env={**env,**extras,'FAIRJOB_SHADOW_CAPACITY':'1'}
        with local_stack(Path('models/pytorch-v1'),8031,6397,output/'logs'/label,environment_overrides=fault_env,api_app='v2_fault_server:app') as stack:
            with httpx.Client(base_url=stack.api_url,timeout=5,trust_env=False) as client:
                started=time.monotonic()
                for request in requests[:50]:
                    response=client.post('/rank',json=request);response.raise_for_status()
                    assert response.json()['model_version']==champion['model_version']
                elapsed=time.monotonic()-started
                time.sleep(.5)
                status=client.get('/shadow/status').json()
                if label=='slow':assert status['dropped']>0 and elapsed<5
                else:assert status['error']>0
                proof['shadow_'+label]={'active_successes':50,'elapsed_seconds':elapsed,'shadow':status}
    mlflow.set_tracking_uri(experiment['tracking_uri'])
    mlflow.set_registry_uri(experiment['tracking_uri'])
    registered=mlflow.pyfunc.load_model('models:/fairjob-v2@champion')
    sample=pd.read_parquet('data/processed/fairjob-v1/validation.parquet').head(10)[FEATURES]
    np.testing.assert_allclose(registered.predict(sample),bundle.predict(sample),atol=1e-6,rtol=1e-5)
    proof['mlflow_alias_load_matches_bundle']=True
    write_json(output/'evidence.json',proof)
    print('V2 HTTP, shadow faults, registry reload and blocked promotion verified',flush=True)


if __name__=='__main__':main()
