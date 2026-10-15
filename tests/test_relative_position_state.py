from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
torch=pytest.importorskip('torch')
from trafficflow.relative_position_state import relative_distance,make_relative_network,RelativePositionStateModel
from trafficflow.recurrent_state import make_network
from trafficflow.state import StateModel,Profile
from trafficflow.data import VALUES


def test_relative_distance_has_analytic_units_and_permutation_invariance():
    p=np.array([0.,1.,3.]);wanted=np.array([[0,2/3,2],[2/3,0,4/3],[2,4/3,0]],np.float32)
    np.testing.assert_array_equal(relative_distance(p),wanted)
    np.testing.assert_array_equal(relative_distance(p*10+17),wanted)
    order=[2,0,1];np.testing.assert_array_equal(relative_distance(p[order]),wanted[np.ix_(order,order)])
    np.testing.assert_array_equal(relative_distance([5,5]),np.zeros((2,2)))
    assert relative_distance([0,1,2,100])[0,-1]==20
    with pytest.raises(ValueError,match='positions'):relative_distance([0,np.nan])


def test_zero_bias_matches_nonzero_head_control_and_distance_can_receive_gradient():
    torch.set_num_threads(1);torch.manual_seed(19);old=make_network('tcn_gru',8)
    torch.manual_seed(19);new=make_relative_network([0,1,4],8)
    for key,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[key])
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())==2
    old.output.weight.data.fill_(.01);new.output.weight.data.copy_(old.output.weight)
    x=torch.randn(1,12,8,3);a=torch.zeros(3,3);fd=torch.ones(3,3)
    assert torch.equal(old(x,a,a,fd),new(x,a,a,fd))
    new(x,a,a,fd).square().mean().backward()
    assert any(layer.distance_strength.grad.abs()>0 for layer in new.spatial)
    assert all(torch.isfinite(layer.distance_strength.grad) for layer in new.spatial)


def test_position_wrapper_preserves_units_keys_and_roundtrip_cache(tmp_path):
    times=pd.date_range('2030-07-01',periods=4,freq='5min',tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ').tolist();links=['a','b']
    grid=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id']);frame=grid.to_frame(index=False)
    frame['speed_kmh']=90.;frame['flow_vph']=1500.;frame['pct_observed']=100.;frame['occupancy']=.1
    net=pd.DataFrame({'lanes':[2.,3.],'free_speed_kmh':[100.,100.],'capacity_vph':[4000.,6000.],'k_jam':[200.,300.]},index=links)
    base=np.broadcast_to([90.,600.],(4,2,2)).copy();f=SimpleNamespace(links=links,times=times,frame=frame,net=net,version=2,raw=base,prior=base,base=base)
    f.lookup=lambda target:grid.get_indexer(pd.MultiIndex.from_frame(target[['timestamp','link_id']]))
    target=frame[['timestamp','link_id']].copy();target['panel']='p';target['station_id']=target.link_id;target['mask_regime']='R3'
    baseline=StateModel(Profile(pd.DataFrame(),pd.DataFrame()),feature_version=2);state=make_relative_network([0,1],8).state_dict()
    model=RelativePositionStateModel(baseline,state,pd.DataFrame(),.75,links,[0,1],hidden=8)
    expected=baseline.predict(f,target);actual=model.predict(f,target);pd.testing.assert_frame_equal(actual,expected)
    network=model._network;path=tmp_path/'relative.pkl';model.save(path);assert model._network is network
    loaded=StateModel.load(path);assert loaded._network is None;pd.testing.assert_frame_equal(loaded.predict(f,target),actual)
    np.testing.assert_array_equal(actual[VALUES].to_numpy()[:2],[[90.,1200.],[90.,1800.]])
    f.links=['b','a']
    with pytest.raises(ValueError,match='link order'):loaded.predict(f,target)
