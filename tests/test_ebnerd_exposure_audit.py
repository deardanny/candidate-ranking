import numpy as np
import pytest
from fairjob_ranker.ebnerd.exposure_audit import ExposureState

def test_availability_is_strict_and_current_request_never_counts():
    state=ExposureState(60)
    x=state.matrix(1000,1,[10,20]);assert (x[:,0]==0).all()
    state.observe(1000,1,[10,20])
    np.testing.assert_array_equal(state.matrix(1000,1,[10,20]),x)
    np.testing.assert_array_equal(state.matrix(1060,1,[10,20]),x)
    after=state.matrix(1061,1,[10,30]);assert after[0,0]==np.float32(np.log(2))
    assert after[0,-1]==np.float32(np.log(2)) and after[1,0]==0
    assert state.matrix(1061,2,[10])[0,-1]==0

def test_history_windows_expire_without_erasing_total_or_first_seen():
    state=ExposureState(0);state.matrix(1000,1,[10]);state.observe(1000,1,[10])
    assert state.matrix(4600,1,[10])[0,1]>0
    v=state.matrix(4601,1,[10])[0];assert v[1]==0 and v[2]>0 and v[0]>0
    v=state.matrix(87401,1,[10])[0];assert v[1]==v[2]==0 and v[0]>0 and v[3]>0
    with pytest.raises(ValueError,match='Chronological'):state.matrix(1000,1,[10])

def test_same_timestamp_order_does_not_change_features():
    def run(order):
        state=ExposureState(0);out={}
        for user,ids in order:
            out[user]=state.matrix(1000,user,ids);state.observe(1000,user,ids)
        return out,state.matrix(1001,1,[10,20])
    a,ap=run([(1,[10]),(2,[10,20])]);b,bp=run([(2,[10,20]),(1,[10])])
    for u in a:np.testing.assert_array_equal(a[u],b[u])
    np.testing.assert_array_equal(ap,bp)
