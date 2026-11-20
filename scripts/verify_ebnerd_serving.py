"""Real HTTP/Redis staging, scheduled load, gated promotion, rollback and faults."""
import asyncio
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import httpx
import numpy as np
import pandas as pd
import psutil
from ebnerd_runtime import stack,ROOT
from local_runtime import stop_process
from fairjob_ranker.ebnerd.bundle import Bundle
from fairjob_ranker.ebnerd.features import FeatureState
from fairjob_ranker.ebnerd.service import StateCache,Ranker,RankRequest
from fairjob_ranker.ebnerd.load import run_load
from fairjob_ranker.io import sha256


def make_request(row):
    return {'schema_version':'ebnerd-v1','request_id':'news-'+str(row.impression_id),'user_id':int(row.user_id),
        'request_time':str(row.impression_time),'device_type':int(row.device_type),
        'candidate_ids':list(map(int,row.article_ids_inview)),'top_k':len(row.article_ids_inview)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-load',action='store_true',help='Reuse this session\'s completed load runs after harness-only corrections')
    args=parser.parse_args()
    data=ROOT/'data/processed/ebnerd-v1'; output=ROOT/'reports/ebnerd-serving'; output.mkdir(parents=True,exist_ok=True)
    candidate=ROOT/'models/ebnerd-v1/final-personalized'; baseline=ROOT/'models/ebnerd-v1/final-metadata'
    final=json.loads((ROOT/'reports/ebnerd-experiment/final_metrics.json').read_text())
    if not final['quality_passed']: raise ValueError('Offline quality gate failed')
    state=FeatureState.load(data/'validation_state'); bundle=Bundle(candidate); ranker=Ranker(candidate,state)
    frame=pd.read_parquet(data/'validation_requests.parquet'); groups=np.load(data/'test_groups.npy'); offsets=np.r_[0,np.cumsum(groups)]
    saved=np.load(data/'test-personalized_scores.npy'); selected=np.random.default_rng(42).choice(len(frame),200,replace=False)
    selected=np.unique(np.r_[selected,int(np.argmax(groups))]); bodies=[make_request(frame.iloc[int(i)]) for i in selected]
    checks={}; largest_error=0.; load_results=[]
    with stack(candidate,baseline,registry=False) as runtime:
        with httpx.Client(base_url=runtime['api_url'],timeout=10,trust_env=False) as client:
            for index,body in zip(selected,bodies):
                response=client.post('/rank',json=body); response.raise_for_status(); result=response.json()
                assert not result['degraded'] and not result['skipped_candidate_ids']
                lookup=dict(zip(body['candidate_ids'],saved[offsets[index]:offsets[index+1]]))
                expected=sorted(lookup,key=lambda i:(-lookup[i],i))
                assert [r['article_id'] for r in result['rankings']]==expected
                largest_error=max(largest_error,max(abs(r['score']-lookup[r['article_id']]) for r in result['rankings']))
            checks['offline_http_parity']=largest_error<1e-12
            body=bodies[0]; a=client.post('/rank',json=body).json(); rev={**body,'candidate_ids':body['candidate_ids'][::-1]}
            checks['candidate_order_invariant']=a['rankings']==client.post('/rank',json=rev).json()['rankings']
            synthetic={**body,'request_id':'128-candidate-contract','candidate_ids':[i for i,a in state.articles.items() if a['published']<=state.available_from][:128],'top_k':128}
            checks['128_candidates_supported']=len(client.post('/rank',json=synthetic).json()['rankings'])==128
            checks['model_unseen_candidates_retained']=any(client.post('/rank',json=b).json()['model_unseen_candidates']>0 for b in bodies[:5])
            checks['future_history_rejected']=client.post('/rank',json={**body,'request_time':'2023-05-24 12:00:00'}).status_code==422
            checks['metrics_endpoint']=client.get('/metrics').status_code==200
            for rate,duration in [(50,60),(100,30)]:
                prior=output/f'load-{rate}.json'
                if args.reuse_load:
                    report=json.loads(prior.read_text())
                    if report['candidate_metadata_sha256']!=bundle.digest or report['state_version']!=state.version: raise ValueError('Load evidence mismatch')
                    load_results.append(report); print('reusing measured load',rate,flush=True); continue
                print('scheduled load',rate,duration,flush=True)
                report=asyncio.run(run_load(runtime['api_url'],bodies,rate,duration,expected_version=bundle.meta['model_version'],pids=[runtime['api_process'].pid]))
                records=report.pop('records'); (ROOT/'runs/ebnerd/serving'/f'load-{rate}-records.json').write_text(json.dumps(records))
                report.update(candidate_metadata_sha256=bundle.digest,state_version=state.version)
                load_results.append(report); (output/f'load-{rate}.json').write_text(json.dumps(report,indent=2)+'\n')
                print('load result',rate,report['on_time_success_rate'],report['latency_success_ms'],flush=True)
            shadow=client.get('/shadow/status').json(); checks['shadow_compared']=shadow['compared']>0
            checks['shadow_no_errors']=shadow['error']==0
    performance=load_results[0]
    quality={'candidate_metadata_sha256':bundle.digest,'champion_metadata_sha256':sha256(baseline/'metadata.json'),
        'state_version':state.version,'checks':{**final['quality_checks'],'offline_online_parity':checks['offline_http_parity']},
        'frozen_release_sha256':sha256(ROOT/'reports/ebnerd-experiment/frozen_release.json')}
    (output/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    control=[]
    with stack(baseline,candidate) as runtime:
        restarted=None
        try:
            with httpx.Client(base_url=runtime['api_url'],timeout=30,trust_env=False) as client:
                headers={'Authorization':'Bearer '+runtime['admin_token']}; body=bodies[0]
                promotion={'bundle':str(candidate),'quality':quality,'performance':performance,'warmup':body}
                checks['unauthorized_control_rejected']=client.post('/admin/rollback').status_code==403
                wrong={**promotion,'quality':{**quality,'state_version':'wrong'}}
                checks['mismatched_evidence_rejected']=client.post('/admin/promote',json=wrong,headers=headers).status_code==409
                response=client.post('/admin/promote',json=promotion,headers=headers)
                if response.status_code!=200: raise ValueError('Promotion gate rejected: '+response.text)
                control.append(response.json()); checks['promotion_succeeded']=True
                expected=ranker.rank(RankRequest.model_validate(body))['rankings']
                checks['promoted_scores_match']=client.post('/rank',json=body).json()['rankings']==expected
                response=client.post('/admin/rollback',headers=headers); response.raise_for_status(); control.append(response.json())
                checks['rollback_restores_baseline']=client.get('/readyz').json()['model_version']==Bundle(baseline).meta['model_version']
                response=client.post('/admin/promote',json=promotion,headers=headers); response.raise_for_status(); control.append(response.json())
                cache=StateCache(state,runtime['redis_url']); keys=cache.keys_values(body['candidate_ids'],body['user_id'])
                cache.client.set(keys[0][0],'corrupt')
                result=client.post('/rank',json=body).json()
                checks['corrupt_cache_fallback']=result['rankings']==expected and result['degraded'] and 'cache_invalid' in result['fallback_reason']
                cache.client.set(*keys[0]); cache.close()
                stop_process(runtime['redis_process'])
                result=client.post('/rank',json=body).json()
                checks['redis_outage_fallback']=result['rankings']==expected and result['degraded'] and 'redis_unavailable' in result['fallback_reason']
                with (ROOT/'runs/ebnerd/serving/redis-restart.log').open('a') as stream:
                    restarted=subprocess.Popen([str(ROOT/'.tools/redis-7.2.12/src/redis-server'),'--bind','127.0.0.1','--port','6381','--save','','--appendonly','no'],stdout=stream,stderr=subprocess.STDOUT)
                cache=StateCache(state,runtime['redis_url']); deadline=time.monotonic()+10
                while True:
                    try: cache.client.ping(); break
                    except Exception:
                        if time.monotonic()>deadline: raise
                        time.sleep(.1)
                cache.seed(); cache.close(); result=client.post('/rank',json=body).json()
                checks['redis_recovered']=not result['degraded'] and result['rankings']==expected
                children=psutil.Process(runtime['api_process'].pid).children()
                shadow_process=next((p for p in children if any('spawn_main' in c for c in p.cmdline())),None)
                if shadow_process:
                    # A grandchild may become a zombie until its API parent polls it.
                    # Kill then issue a sampled request, whose is_alive() reaps it.
                    shadow_process.kill(); time.sleep(.1)
                    sampled=next('fault-shadow-'+str(i) for i in range(100) if int.from_bytes(hashlib.sha256(('fault-shadow-'+str(i)).encode()).digest()[:8],'big')%5==0)
                    fault_body={**body,'request_id':sampled}
                    checks['shadow_process_failure_isolated']=client.post('/rank',json=fault_body).json()['rankings']==expected
                else: checks['shadow_process_failure_isolated']=False
                (ROOT/'runs/ebnerd/serving/example_response.json').write_text(json.dumps(result,indent=2)+'\n')
        finally: stop_process(restarted)
    registry=json.loads((ROOT/'models/ebnerd-v1/registry.json').read_text())
    import mlflow
    registered=mlflow.MlflowClient(tracking_uri=registry['tracking_uri']).get_model_version_by_alias('ebnerd-v1','champion')
    checks['registry_champion_matches']=registered.tags.get('bundle_metadata_sha256')==bundle.digest
    report={'passed':all(checks.values()),'checks':checks,'maximum_offline_http_score_error':largest_error,
        'parity_requests':len(bodies),'load_reports':load_results,'deployment_events':control,
        'registry_champion_version':registered.version,'candidate_metadata_sha256':bundle.digest,
        'state_version':state.version,'services_stopped':True}
    report['load_reused_after_harness_only_correction']=args.reuse_load
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']: raise SystemExit('Serving verification failed')


if __name__=='__main__': main()
