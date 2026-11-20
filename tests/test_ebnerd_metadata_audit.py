import numpy as np
import pandas as pd
import pytest
from fairjob_ranker.ebnerd.metadata_audit import EventOnlyState,structurally_valid,build_event_matrix,permute_metadata,STRICT_FEATURES,mask_columns
from fairjob_ranker.ebnerd.features import FEATURES,META,PERSONAL


def history():
    return pd.DataFrame({'user_id':[1],'article_id_fixed':[[10,20]],
        'impression_time_fixed':[np.array(['2023-05-01','2023-05-02'],dtype='datetime64[us]')]})


def requests():
    return pd.DataFrame({'impression_id':[11,12],'user_id':[1,1],'impression_time':pd.to_datetime(['2023-05-18','2023-05-19']),
        'device_type':[2,2],'article_ids_inview':[[10,30],[20,40]],'article_ids_clicked':[[10],[40]]})


def test_event_only_pipeline_does_not_read_catalog_or_labels_for_features(monkeypatch):
    def deny_read(*args,**kwargs): raise AssertionError('Feature builder opened an external table')
    monkeypatch.setattr(pd,'read_parquet',deny_read)
    state=EventOnlyState(history(),'2023-05-18'); frame=requests()
    x,y,ids,groups=build_event_matrix(structurally_valid(frame),state)
    flipped=frame.copy(); flipped['article_ids_clicked']=[[30],[20]]
    other,other_y,_,_=build_event_matrix(structurally_valid(flipped),state)
    np.testing.assert_array_equal(x,other); assert not np.array_equal(y,other_y)
    assert 30 in ids and 40 in ids  # IDs absent from all prior history still accepted.
    assert x.shape==(4,len(STRICT_FEATURES))


def test_strict_history_rejects_equal_future_and_missing_time():
    h=history(); h.at[0,'impression_time_fixed']=np.array(['2023-05-01','2023-05-18'],dtype='datetime64[us]')
    with pytest.raises(ValueError,match='Future'): EventOnlyState(h,'2023-05-18')
    h.at[0,'impression_time_fixed']=np.array(['2023-05-01','NaT'],dtype='datetime64[us]')
    with pytest.raises(ValueError,match='Invalid'): EventOnlyState(h,'2023-05-18')


def test_metadata_permutation_preserves_ids_timestamps_and_within_day_distribution():
    a=pd.DataFrame({'article_id':np.arange(20),'published_time':pd.to_datetime(['2023-05-01']*10+['2023-05-02']*10),
        'last_modified_time':pd.to_datetime(['2023-06-29']*20),'category':np.arange(20),
        'premium':[True,False]*10,'topics':[[str(i)] for i in range(20)],'ner_clusters':[['e'+str(i)] for i in range(20)]})
    changed,info=permute_metadata(a,42)
    pd.testing.assert_frame_equal(a[['article_id','published_time','last_modified_time']],changed[['article_id','published_time','last_modified_time']])
    assert info['remapped_ids']>0
    for day,group in a.groupby('published_time'):
        assert sorted(group.category)==sorted(changed.loc[changed.published_time.eq(day),'category'])
    for _,row in changed.iterrows(): assert row.topics==[str(row.category)]  # fields move coherently


def test_missing_value_intervention_changes_exactly_requested_columns():
    x=np.ones((2,len(FEATURES)),dtype=np.float32); changed=mask_columns(x,META+PERSONAL)
    selected=[FEATURES.index(c) for c in META+PERSONAL]
    assert np.isnan(changed[:,selected]).all()
    keep=[i for i in range(len(FEATURES)) if i not in selected]
    np.testing.assert_array_equal(x[:,keep],changed[:,keep]); assert np.isfinite(x).all()
