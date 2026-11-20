from itertools import permutations
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import ndcg_score,roc_auc_score
from fairjob_ranker.ebnerd.features import FeatureState,FEATURES,BEHAVIOR,FRESH,META,USER,PERSONAL
from fairjob_ranker.ebnerd.evaluation import outcomes,paired_interval


def fixture_state():
    a=pd.DataFrame({'article_id':[1,2,3], 'published_time':pd.to_datetime(['2023-05-01','2023-05-02','2023-05-25']),
        'last_modified_time':pd.to_datetime(['2023-06-29']*3),'category':[1,2,1],'premium':[False]*3,
        'topics':[['sport'],['news'],['sport']],'ner_clusters':[['team'],['city'],['team']]})
    h=pd.DataFrame({'user_id':[10,20], 'article_id_fixed':[[1,1],[2,2]],
        'impression_time_fixed':[np.array(['2023-05-03','2023-05-04'],dtype='datetime64[us]')]*2})
    return a,h,FeatureState(a,h,'2023-05-18')


def test_feature_matrix_respects_asof_and_new_items_without_fitting_catalog_ids():
    a,h,state=fixture_state()
    x=state.matrix(10,'2023-05-26',2,[1,3])
    assert x[1,FEATURES.index('history_item_unknown')]==1
    assert x[1,FEATURES.index('category_affinity')]==1
    with pytest.raises(ValueError,match='Future history'): state.matrix(10,'2023-05-17',2,[1])
    with pytest.raises(ValueError,match='Unpublished'): state.matrix(10,'2023-05-20',2,[3])
    h.at[0,'impression_time_fixed']=np.array(['2023-05-03','2023-05-18'],dtype='datetime64[us]')
    with pytest.raises(ValueError,match='strictly before'): FeatureState(a,h,'2023-05-18')


def test_permuting_candidates_preserves_scores_features_and_history_control_is_isolated():
    _,_,s=fixture_state(); x=s.matrix(10,'2023-05-26',2,[1,2,3]); rev=s.matrix(10,'2023-05-26',2,[3,2,1])
    np.testing.assert_array_equal(x,rev[::-1])
    shuffled=s.matrix(10,'2023-05-26',2,[1,2,3],history_user_id=20)
    fixed=[FEATURES.index(c) for c in BEHAVIOR+FRESH+META]
    np.testing.assert_array_equal(x[:,fixed],shuffled[:,fixed])
    assert not np.array_equal(x,shuffled)


def test_state_roundtrip_and_tamper_detection(tmp_path):
    a,h,s=fixture_state(); FeatureState.save(tmp_path/'state',a,h,'2023-05-18')
    loaded=FeatureState.load(tmp_path/'state')
    np.testing.assert_array_equal(s.matrix(10,'2023-05-26',2,[1,3]),loaded.matrix(10,'2023-05-26',2,[1,3]))
    with (tmp_path/'state/articles.parquet').open('ab') as stream: stream.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'): FeatureState.load(tmp_path/'state')


def test_ranking_metrics_match_independent_libraries_and_exhaustive_tie_permutations():
    y=np.array([1,0,1,0]); scores=np.ones(4)
    result=outcomes(y,scores,[4],np.arange(4),{2}).iloc[0]
    assert result.ndcg5==pytest.approx(ndcg_score([y],[scores],k=5))
    assert result.auc==pytest.approx(roc_auc_score(y,scores))
    rr=[1/(np.flatnonzero(y[list(p)])[0]+1) for p in permutations(range(4))]
    assert result.mrr==pytest.approx(np.mean(rr))
    assert result.cold_hit1==pytest.approx(.25)


def test_bootstrap_pairs_identical_users_and_preserves_direction():
    users=[1,1,2,3,3,3]
    zero=paired_interval(users,np.zeros(6),100)
    assert zero['estimate']==zero['lower']==zero['upper']==0
    shift=paired_interval(users,np.ones(6)*.2,100)
    assert shift['lower']==pytest.approx(.2)
