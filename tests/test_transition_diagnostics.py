import numpy as np
import pytest
from trafficflow.transition_diagnostics import transition_diagnostics


def test_persistent_wave_and_isolated_return_have_distinct_label_diagnostics():
    y=np.full((7,3,2),1000.);y[...,0]=90.
    y[:,0,0]=[90,90,45,45,45,90,90];y[:,1,0]=[90,90,90,45,45,45,90]
    y[:,2,0]=[90,90,30,90,90,90,90];y[2,2,1]=1500.
    d=transition_diagnostics(y,np.ones((7,3),bool),[[],[0],[]],[[1],[],[]])
    assert d['drop20'][2,0] and d['speed_persistent10'][2,0] and not d['speed_reversal5'][2,0]
    assert d['neighbor_same_sign10'][2,0] and d['neighbor_available'][2,0]
    assert d['drop20'][2,2] and d['speed_reversal5'][2,2] and not d['speed_persistent10'][2,2]
    assert d['flow150'][2,2] and d['flow_reversal50'][2,2] and not d['flow_persistent75'][2,2]
    assert not d['neighbor_available'][:,2].any()
    assert d['recover20'][6,1] and not d['future2_eligible'][6,1]


def test_missing_labels_and_day_edges_cannot_create_confirmed_events():
    y=np.zeros((5,1,2));y[...,0]=np.array([90,0,20,90,0])[:,None]
    valid=np.ones((5,1),bool);valid[1]=False
    d=transition_diagnostics(y,valid,[[]],[[]])
    assert not d['pair_valid'][0].any() and not d['drop20'][1].any() and not d['recover20'][2].any()
    assert not np.isfinite(d['delta'][1:3]).any()
    assert d['drop20'][-1,0] and not d['future1_eligible'][-1,0] and not d['speed_reversal5'][-1,0]
    with pytest.raises(ValueError,match='topology'):transition_diagnostics(y,valid,[[0]],[[]])
