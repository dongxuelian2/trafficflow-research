import numpy as np
import pandas as pd
import pytest
from trafficflow.state import Profile,StateFeatures
from trafficflow.outage_stress import MEASUREMENTS
from trafficflow.state_augmentation import rebuild_after_outage,draw_training_plan


def test_rebuilt_features_cannot_recover_removed_measurements_or_old_base():
    times=pd.date_range('2030-08-01',periods=20,freq='5min',tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    frame=pd.DataFrame([(t,link,float(i)) for t in times for i,link in enumerate(['a','b','c'])],
        columns=['timestamp','link_id','milepost'])
    frame['speed_kmh']=np.tile([35.,105.,45.],20)
    frame['flow_vph']=np.tile([1200.,2800.,1400.],20)
    frame['occupancy']=.15;frame['density_occ_linear_vehpkm']=50.;frame['pct_observed']=100.
    frame['is_observed']=1;frame['is_score_eligible']=1;frame['is_imputed']=0;frame['is_missing']=0
    net=pd.DataFrame({'lanes':[2.,2.,2.],'free_speed_kmh':[110.]*3,'capacity_vph':[4000.]*3,
        'length_km':[.5]*3,'k_jam':[300.]*3},index=['a','b','c'])
    topology=pd.DataFrame({'incoming_link_ids':['','a','b'],'outgoing_link_ids':['b','c','']},index=net.index)
    prior_frame=frame.copy();prior_frame['speed_kmh']=70.
    profile=Profile.fit([prior_frame]);selected=(frame.link_id=='b').to_numpy()
    original=StateFeatures(frame,net,topology,profile,version=3)
    hidden=rebuild_after_outage(frame,selected,net,topology,profile)
    assert not np.array_equal(original.base,hidden.base)
    poison=frame.copy();poison.loc[selected,MEASUREMENTS]=999999.
    again=rebuild_after_outage(poison,selected,net,topology,profile)
    pd.testing.assert_frame_equal(hidden.features,again.features)
    np.testing.assert_array_equal(hidden.base,again.base)
    assert np.isnan(hidden.raw[:,1]).all()
    np.testing.assert_array_equal(frame.speed_kmh,np.tile([35.,105.,45.],20))


def test_augmentation_plan_preserves_original_patch_rng_and_target_exposure():
    regimes=[[('a',288),('b',200)],[('c',288)],[('d',288),('e',240)]]
    plan=draw_training_plan(17,regimes,40)
    disabled=draw_training_plan(17,regimes,40,probability=0.)
    always=draw_training_plan(17,regimes,40,probability=1.)
    np.testing.assert_array_equal(plan[...,:3],disabled[...,:3])
    np.testing.assert_array_equal(plan[...,:3],always[...,:3])
    assert np.all(disabled[...,3]==-1) and np.all(always[...,3]>=0)
    rng=np.random.default_rng(17)
    expected=[]
    for _ in range(160):
        group=int(rng.integers(3));index=int(rng.integers(len(regimes[group])))
        expected.append([group,index,int(rng.integers(regimes[group][index][1]))])
    np.testing.assert_array_equal(plan[...,:3].reshape(-1,3),expected)
    with pytest.raises(ValueError):draw_training_plan(17,regimes,2,probability=1.1)
