import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fairjob_ranker.io import sha256
from fairjob_ranker.ebnerd.features import FeatureState,FEATURES
from fairjob_ranker.ebnerd.service import RankRequest,Ranker,StateCache,Deployment,Settings,create_app,QUALITY_KEYS


@pytest.fixture
def bundles(tmp_path):
    a=pd.DataFrame({'article_id':range(1,101),'published_time':pd.to_datetime(['2023-05-01']*100),
        'last_modified_time':pd.to_datetime(['2023-06-29']*100),'category':[i%3 for i in range(100)],'premium':[False]*100,
        'topics':[['sport']]*100,'ner_clusters':[['team']]*100})
    h=pd.DataFrame({'user_id':[10],'article_id_fixed':[[1,2]],
        'impression_time_fixed':[np.array(['2023-05-03','2023-05-04'],dtype='datetime64[us]')]})
    statepath=tmp_path/'state'; FeatureState.save(statepath,a,h,'2023-05-18'); state=FeatureState.load(statepath)
    x=state.matrix(10,'2023-05-26',2,list(range(1,101)))
    for name in ['first','second']:
        directory=tmp_path/name; directory.mkdir()
        model=lgb.train({'objective':'binary','min_data_in_leaf':1,'min_data_in_bin':1,'num_leaves':3,'num_threads':1,'verbosity':-1},
            lgb.Dataset(x,label=np.arange(100)%2,feature_name=FEATURES),num_boost_round=3)
        model.save_model(str(directory/'model.txt')); (directory/'features.json').write_text(json.dumps(FEATURES))
        (directory/'seen_items.json').write_text('[1,2]')
        meta={'schema_version':'ebnerd-v1','score_type':'ranking_score','variant':'personalized','model_version':name,
            'content_snapshot_assumption':True,'publication_stability_assumption':True,
            'sha256':{n:sha256(directory/n) for n in ['model.txt','features.json','seen_items.json']}}
        (directory/'metadata.json').write_text(json.dumps(meta))
    return tmp_path,state


def body():
    return {'request_id':'test','user_id':10,'request_time':'2023-05-26 08:00:00','device_type':2,
            'candidate_ids':list(range(1,101)),'top_k':100}


def evidence(candidate,active,state):
    common={'candidate_metadata_sha256':sha256(candidate/'metadata.json'),'state_version':state.version}
    q={**common,'champion_metadata_sha256':active.bundle.digest,'checks':{k:True for k in QUALITY_KEYS}}
    p={**common,'on_time_success_rate':1.,'scheduled':500,'duration_seconds':10,'deadline_ms':50,
       'invalid_responses':0,'degraded_successful':0,'generator_dropped':0}
    return q,p


def test_100_candidates_unseen_ids_and_http_offline_parity(bundles):
    root,state=bundles; ranker=Ranker(root/'first',state); request=RankRequest(**body())
    expected=ranker.rank(request)
    assert len(expected['rankings'])==100 and expected['model_unseen_candidates']==98
    with TestClient(create_app(Settings(root/'first',root/'state'))) as client:
        result=client.post('/rank',json=body()); assert result.status_code==200
        assert result.json()['rankings']==expected['rankings']
        reverse=body(); reverse['candidate_ids'].reverse()
        assert client.post('/rank',json=reverse).json()['rankings']==expected['rankings']
        future=body(); future['request_time']='2023-05-17'
        assert client.post('/rank',json=future).status_code==422
        assert client.post('/admin/rollback').status_code==403
        assert 'ebnerd_rank_requests' in client.get('/metrics').text


def test_redis_corruption_and_inference_fault_use_explicit_fallback(bundles):
    root,state=bundles; cache=StateCache(state)
    class BadCache:
        def mget(self,keys): return ['wrong']*len(keys)
    cache.client=BadCache(); ranker=Ranker(root/'first',state,cache)
    normal=Ranker(root/'first',state).rank(RankRequest(**body()))
    degraded=ranker.rank(RankRequest(**body()))
    assert degraded['rankings']==normal['rankings'] and degraded['fallback_reason']=='cache_invalid'
    ranker.bundle.predict=lambda _: (_ for _ in ()).throw(RuntimeError('fault'))
    result=ranker.rank(RankRequest(**body()))
    assert result['scoring_method']=='fixed_past_click_popularity' and result['degraded']


def test_deployment_rejects_wrong_evidence_and_rolls_back(bundles):
    root,state=bundles; d=Deployment(root/'first',state,StateCache(state)); q,p=evidence(root/'second',d.active,state)
    bad={**q,'state_version':'future-state'}
    with pytest.raises(ValueError,match='state_mismatch'): d.promote(root/'second',bad,p,RankRequest(**body()))
    assert d.snapshot().bundle.meta['model_version']=='first'
    d.promote(root/'second',q,p,RankRequest(**body()))
    assert d.snapshot().bundle.meta['model_version']=='second'
    d.rollback(); assert d.snapshot().bundle.meta['model_version']=='first'


def test_registry_failure_cannot_partially_swap_the_active_model(bundles):
    root,state=bundles
    def fail(_): raise RuntimeError('Registry unavailable')
    d=Deployment(root/'first',state,StateCache(state),fail); q,p=evidence(root/'second',d.active,state)
    with pytest.raises(RuntimeError): d.promote(root/'second',q,p,RankRequest(**body()))
    assert d.snapshot().bundle.meta['model_version']=='first' and d.previous is None
