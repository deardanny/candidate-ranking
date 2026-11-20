"""Rehearse HTTP promote/rollback using identical weights in a new package revision.

The real DCN candidate stays blocked. This fixture tests deployment mechanics only.
"""
import asyncio
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil

os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY','true')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
import httpx
import mlflow
import numpy as np
import pandas as pd

from fairjob_ranker.evaluation import evaluate,paired_user_bootstrap
from fairjob_ranker.io import read_json,write_json,sha256
from fairjob_ranker.replay import load_requests
from fairjob_ranker.schema import FEATURES
from fairjob_ranker.v2.load import run_load
from fairjob_ranker.v2.models import load_bundle
from fairjob_ranker.v2.training import register_bundle
from local_runtime import local_stack


def main():
    output=Path('reports/v2/release_rehearsal')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    experiment=read_json(Path('reports/v2/experiment/metrics.json'))
    champion=experiment['champion']
    original=load_bundle(champion['bundle'])
    fixture_path=Path('models/v2-release-rehearsal')
    shutil.copytree(original.directory,fixture_path)
    meta=read_json(fixture_path/'metadata.json')
    meta['model_version'] += '-package-rehearsal'
    meta['purpose']='Identical-weight deployment fixture; no model improvement claim'
    write_json(fixture_path/'metadata.json',meta)
    fixture=load_bundle(fixture_path)
    assert fixture.metadata['artifact_sha256']==original.metadata['artifact_sha256']
    validation=pd.read_parquet('data/processed/fairjob-v1/validation.parquet')
    before=original.predict(validation);after=fixture.predict(validation)
    np.testing.assert_array_equal(before,after)
    _,a=evaluate(validation,before,set())
    _,b=evaluate(validation,after,set())
    comparison=paired_user_bootstrap(a,b,repeats=1000)
    comparison['difference']='identical_weight_package_revision - active_champion'
    quality={'candidate_metadata_sha256':sha256(fixture_path/'metadata.json'),
             'champion_metadata_sha256':sha256(original.directory/'metadata.json'),'passed':True,
             'checks':{k:True for k in ['inner_logloss_guard','outer_logloss_guard','outer_ndcg_point_estimate','outer_ndcg_noninferiority']},
             'scope':meta['purpose'],'identical_weight_hashes':True,'max_abs_prediction_error':0.,'comparison':comparison}
    write_json(output/'quality.json',quality)
    mlflow.set_tracking_uri(experiment['tracking_uri']);mlflow.set_registry_uri(experiment['tracking_uri'])
    mlflow.set_experiment('fairjob-v2-release-rehearsal')
    name='fairjob-v2-release-rehearsal'
    version=mlflow.register_model(champion['model_uri'],name)
    client=mlflow.MlflowClient()
    client.set_model_version_tag(name,version.version,'bundle_metadata_sha256',sha256(original.directory/'metadata.json'))
    client.set_registered_model_alias(name,'champion',version.version)
    with mlflow.start_run(run_name='identical-weights-package-revision') as run:
        registration=register_bundle(fixture,validation[FEATURES].head(2),run.info.run_id,name)
        mlflow.set_tag('purpose',meta['purpose'])
    controller={'registry_name':name,'tracking_uri':experiment['tracking_uri'],
                'champion':{**champion,'registry_version':version.version},
                'challenger':{'bundle':str(fixture_path),'model_version':fixture.metadata['model_version'],**registration},
                'runs':[{**champion,'registry_version':version.version},
                        {'bundle':str(fixture_path),**registration}]}
    write_json(output/'controller.json',controller)
    requests,_=load_requests(Path('data/replay/validation.jsonl'))
    env={'FAIRJOB_V2_EXPERIMENT':str(output/'controller.json'),'FAIRJOB_V2_SHADOW':'',
         'FAIRJOB_MAX_INFLIGHT':'8','FAIRJOB_ADMIN_TOKEN':'rehearsal-only',
         'MLFLOW_DISABLE_TELEMETRY':'true','MLFLOW_DISABLE_AGENT_HINT':'1'}
    # Measure this exact package before the promotion attempt.
    with local_stack(Path('models/pytorch-v1'),8032,6398,output/'logs'/'candidate',
                     environment_overrides={**env,'FAIRJOB_V2_ACTIVE':str(fixture_path)},api_app='v2_server:app') as stack:
        asyncio.run(run_load(stack.api_url,requests,50,3,expected_version=fixture.metadata['model_version']))
        performance=asyncio.run(run_load(stack.api_url,requests,50,10,expected_version=fixture.metadata['model_version']))
        performance.pop('records')
        performance.update(candidate_metadata_sha256=sha256(fixture_path/'metadata.json'),evaluation_split='validation')
        write_json(output/'performance.json',performance)
    proof={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':meta['purpose'],'quality':quality}
    with local_stack(Path('models/pytorch-v1'),8032,6398,output/'logs'/'release',environment_overrides=env,api_app='v2_server:app') as stack:
        with httpx.Client(base_url=stack.api_url,timeout=10,trust_env=False) as http:
            first=http.post('/rank',json=requests[0]).json()
            response=http.post('/admin/promote',headers={'Authorization':'Bearer '+env['FAIRJOB_ADMIN_TOKEN']},json={
                'bundle':str(fixture_path),'quality_evidence':str(output/'quality.json'),
                'performance_evidence':str(output/'performance.json'),'warmup':requests[0]})
            response.raise_for_status();proof['promotion']=response.json()
            promoted=http.post('/rank',json=requests[0]).json()
            assert promoted['model_version']==fixture.metadata['model_version'] and promoted['rankings']==first['rankings']
            assert client.get_model_version_by_alias(name,'champion').version==registration['registry_version']
            response=http.post('/admin/rollback',headers={'Authorization':'Bearer '+env['FAIRJOB_ADMIN_TOKEN']})
            response.raise_for_status();proof['rollback']=response.json()
            reverted=http.post('/rank',json=requests[0]).json()
            assert reverted['model_version']==first['model_version'] and reverted['rankings']==first['rankings']
            assert client.get_model_version_by_alias(name,'champion').version==version.version
            (output/'prometheus.txt').write_text(http.get('/metrics').text)
    proof['real_candidate_registry_champion_unchanged']=client.get_model_version_by_alias(experiment['registry_name'],'champion').version==champion['registry_version']
    assert proof['real_candidate_registry_champion_unchanged']
    write_json(output/'evidence.json',proof)
    print('Real HTTP package promotion and rollback verified; actual DCN candidate remains blocked')


if __name__=='__main__':main()
