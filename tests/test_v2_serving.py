import json
from pathlib import Path
import time
import threading

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
import torch

from fairjob_ranker.features import FeatureTransformer
from fairjob_ranker.io import sha256,write_json
from fairjob_ranker.schema import AD_FEATURES,CATEGORICAL,FEATURES,NUMERICAL,VERSION
from fairjob_ranker.serving import RankRequest
from fairjob_ranker.torch_model import EmbeddingMLP
from fairjob_ranker.v2.api import Settings,create_app
from fairjob_ranker.v2.deployment import Deployment,check_evidence
from fairjob_ranker.v2.models import load_bundle
from fairjob_ranker.v2.shadow import Shadow,compare
from fairjob_ranker.v2.training import save_bundle


@pytest.fixture(scope='module')
def bundles(tmp_path_factory):
    root=tmp_path_factory.mktemp('v2')
    frame=pd.DataFrame({**{c:[1,2] for c in CATEGORICAL},**{n:[0.,1.] for n in NUMERICAL}})
    transformer=FeatureTransformer.fit(frame)
    data=root/'data'; data.mkdir()
    frame[AD_FEATURES].assign(product_id=[1,2]).to_parquet(data/'ad_catalog.parquet',index=False)
    arch={'category_sizes':[3]*len(CATEGORICAL),'embedding_dim':2,'hidden_dims':[8],'dropout':0.}
    fallback={'fitted_on':'inner_train','train_ctr':.01,'scores':{'1':.02,'2':.01}}
    paths=[]
    for seed in [1,2]:
        torch.manual_seed(seed)
        model=EmbeddingMLP(**arch)
        path=root/f'model{seed}'
        save_bundle(path,model,'mlp',arch,transformer,fallback,data,{},f'test{seed}')
        paths.append(path)
    return paths


@pytest.fixture
def request_body():
    return {'request_id':'v2-test','schema_version':VERSION,
            'context':{'cat0':1,'cat1':1,'numeric':{n:0. for n in NUMERICAL}},'candidate_ids':[1,2],'top_k':2}


def passing_evidence(candidate,champion):
    quality={'candidate_metadata_sha256':sha256(candidate/'metadata.json'),
             'champion_metadata_sha256':sha256(champion/'metadata.json'),'passed':True,
             'checks':{k:True for k in ['inner_logloss_guard','outer_logloss_guard','outer_ndcg_point_estimate','outer_ndcg_noninferiority']}}
    performance={'candidate_metadata_sha256':sha256(candidate/'metadata.json'),'evaluation_split':'validation',
                 'scheduled':500,'duration_seconds':10,'on_time_success_rate':1.,'deadline_ms':50,
                 'invalid_responses':0,'degraded_successful':0,'generator_dropped':0}
    return quality,performance


def test_gate_blocks_wrong_artifact_nan_deadline_and_failed_quality(bundles):
    champion,candidate=bundles
    q,p=passing_evidence(candidate,champion)
    assert not check_evidence(candidate,champion,q,p)
    p['on_time_success_rate']=float('nan')
    assert 'deadline_gate_failed' in check_evidence(candidate,champion,q,p)
    p['candidate_metadata_sha256']='wrong'
    q['checks']['outer_logloss_guard']=False
    reasons=check_evidence(candidate,champion,q,p)
    assert 'performance_candidate_mismatch' in reasons and 'quality_gate_failed' in reasons


def test_atomic_swap_keeps_previous_snapshot_and_rollback_restores_scores(bundles,request_body,tmp_path):
    first,second=bundles
    deployed=Deployment(first)
    body=RankRequest.model_validate(request_body)
    old=deployed.snapshot()
    before=old.rank(body)['rankings']
    q,p=passing_evidence(second,first)
    write_json(tmp_path/'q.json',q); write_json(tmp_path/'p.json',p)
    try:
        deployed.promote(second,tmp_path/'q.json',tmp_path/'p.json',body)
        assert deployed.snapshot() is not old
        assert old.rank(body)['rankings']==before  # Admitted old requests can still finish.
        deployed.rollback()
        assert deployed.snapshot().rank(body)['rankings']==before
    finally:
        deployed.close()


def test_registry_update_failure_does_not_replace_active_model(bundles,request_body,tmp_path):
    first,second=bundles
    def broken(_): raise RuntimeError('registry unavailable')
    deployment=Deployment(first,alias_callback=broken)
    before=deployment.snapshot()
    q,p=passing_evidence(second,first)
    write_json(tmp_path/'q.json',q); write_json(tmp_path/'p.json',p)
    try:
        with pytest.raises(RuntimeError,match='registry'):
            deployment.promote(second,tmp_path/'q.json',tmp_path/'p.json',RankRequest.model_validate(request_body))
        assert deployment.snapshot() is before
    finally:
        deployment.close()


def test_slow_registry_update_does_not_block_active_snapshot(bundles,request_body,tmp_path):
    first,second=bundles
    entered,release,snapshot_ready=threading.Event(),threading.Event(),threading.Event()
    errors=[]
    def delayed_registry(_):
        entered.set()
        if not release.wait(5):raise TimeoutError('test release not signalled')
    deployment=Deployment(first,alias_callback=delayed_registry)
    original=deployment.snapshot()
    q,p=passing_evidence(second,first)
    write_json(tmp_path/'q.json',q);write_json(tmp_path/'p.json',p)
    def promote():
        try:deployment.promote(second,tmp_path/'q.json',tmp_path/'p.json',RankRequest.model_validate(request_body))
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=promote)
    snapshot_thread=None
    try:
        thread.start();assert entered.wait(5)
        captured=[]
        def snapshot():
            captured.append(deployment.snapshot());snapshot_ready.set()
        snapshot_thread=threading.Thread(target=snapshot);snapshot_thread.start()
        assert snapshot_ready.wait(.5),'Registry latency blocked the active request path'
        assert captured==[original]
    finally:
        release.set();thread.join(5)
        if snapshot_thread:snapshot_thread.join(5)
        deployment.close()
    assert not errors


def test_http_parity_prometheus_and_unauthorized_admin(bundles,request_body):
    app=create_app(Settings(active=bundles[0]))
    with TestClient(app) as client:
        response=client.post('/rank',json=request_body)
        assert response.status_code==200
        actual=response.json()
        bundle=load_bundle(bundles[0])
        raw=pd.DataFrame([{**{c:i for c in AD_FEATURES},'cat0':1,'cat1':1,**request_body['context']['numeric']} for i in [1,2]])
        scores=bundle.predict(raw)
        by_id={r['product_id']:r['score'] for r in actual['rankings']}
        np.testing.assert_allclose([by_id[1],by_id[2]],scores,atol=1e-6)
        request_body['candidate_ids'].reverse()
        assert client.post('/rank',json=request_body).json()['rankings']==actual['rankings']
        assert client.post('/admin/rollback').status_code==403
        text=client.get('/metrics').text
        assert 'fairjob_http_requests_total{status="200"} 2.0' in text
        assert 'fairjob_inflight 0.0' in text


def test_slow_shadow_queue_drops_without_waiting_in_caller(bundles,request_body):
    from fairjob_ranker.v2.service import Ranker
    active=Ranker(bundles[0])
    body=active.rank(RankRequest.model_validate(request_body))
    shadow=Shadow(bundles[1],capacity=1,fault_delay=.2)
    try:
        start=time.monotonic()
        accepted=[shadow.submit(request_body,body) for _ in range(100)]
        assert time.monotonic()-start < .15
        assert not all(accepted)
        deadline=time.monotonic()+10
        results=[]
        while not results and time.monotonic()<deadline:
            results.extend(shadow.drain()); time.sleep(.02)
        assert results and results[0]['status']=='compared'
    finally:
        shadow.close(); active.close()


def test_shadow_contract_mismatch_rejected():
    a={'schema_version':'a','catalog_version':'1'}
    b={'schema_version':'b','catalog_version':'1'}
    with pytest.raises(ValueError,match='contract'):
        compare(a,b)
