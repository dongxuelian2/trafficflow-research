from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from trafficflow.data import STATE_KEY,VALUES
from trafficflow.state import StateModel,Profile
from trafficflow.gap_gated_state import missing_run_steps,GapGatedState


class ConstantState(StateModel):
    def __init__(self,value):
        self.value=value;self.calls=0;self.profile=Profile(pd.DataFrame({'v':[1]}),pd.DataFrame({'v':[1]}));self.feature_version=3
    def predict(self,features,template):
        self.calls+=1;result=template[STATE_KEY].copy();result[VALUES]=np.full((len(template),2),self.value);return result


def test_gap_lengths_include_censored_edges_without_crossing_observed_cells():
    raw=np.ones((50,3,2));raw[:36,0]=np.nan;raw[10:45,1]=np.nan;raw[20:,2]=np.nan
    times=pd.date_range('2030-12-11',periods=50,freq='5min',tz='UTC')
    result=missing_run_steps(raw,times)
    np.testing.assert_array_equal(result[:,0],np.r_[np.full(36,36),np.zeros(14)])
    np.testing.assert_array_equal(result[:,1],np.r_[np.zeros(10),np.full(35,35),np.zeros(5)])
    np.testing.assert_array_equal(result[:,2],np.r_[np.zeros(20),np.full(30,30)])
    with pytest.raises(ValueError,match='Consecutive'):missing_run_steps(raw,times[::-1])


def test_gate_reads_availability_only_preserves_keys_and_skips_unused_expert(tmp_path):
    raw=np.ones((50,2,2));raw[5:41,0]=np.nan;raw[10:25,1]=np.nan
    features=SimpleNamespace(raw=raw,times=pd.date_range('2030-12-11',periods=50,freq='5min',tz='UTC'),
        lookup=lambda frame:frame.cell.to_numpy(int))
    target=pd.DataFrame({'panel':['p']*4,'timestamp':['t']*4,'station_id':['a','b','c','d'],'link_id':['x']*4,
        'mask_regime':['R1','R2','R3','R1'],'cell':[0,20,21,90]})
    a,b=ConstantState(1),ConstantState(2);model=GapGatedState(a,b)
    p=model.predict(features,target)
    np.testing.assert_array_equal(p.speed_kmh,[1,2,1,1]);assert a.calls==1 and b.calls==1
    changed=raw.copy();changed[np.isfinite(changed)]=999999.;features.raw=changed
    np.testing.assert_array_equal(model.predict(features,target)[VALUES],p[VALUES])
    single=target.iloc[[0]];before=b.calls;model.predict(features,single);assert b.calls==before
    single=target.iloc[[1]];before=a.calls;model.predict(features,single);assert a.calls==before
    path=tmp_path/'gate.pkl';model.save(path)
    loaded=StateModel.load(path);np.testing.assert_array_equal(loaded.predict(features,target)[VALUES],p[VALUES])
    with pytest.raises(ValueError,match='threshold'):GapGatedState(a,b,0)
    bad=ConstantState(2);bad.feature_version=2
    with pytest.raises(ValueError,match='identical'):GapGatedState(a,bad)
