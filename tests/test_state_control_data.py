import numpy as np
import pandas as pd
import pytest
from trafficflow.state_control_data import context_exposure,nominal_teacher_weights,current_weighted_mse,causal_state_input


def test_nominal_weights_reproduce_sum_of_window_mean_losses():
    schedule={'warm':np.array([[[0,12],[0,13]],[[0,12],[0,12]]]),'event':np.array([[[0,13]],[[0,13]]])};coords,counts,mass=context_exposure(schedule)
    np.testing.assert_array_equal(coords,[[0,12],[0,13]]);np.testing.assert_array_equal(counts,[[3,0],[1,2]])
    domain=np.array([[[1,1],[1,0]],[[1,0],[1,1]]],bool);error=np.array([[[2.,4.],[5.,999.]],[[3.,999.],[6.,8.]]]);raw,normalized=nominal_teacher_weights(mass,domain)
    scalar=3*(3+5)/2+1*(3+7)/2+2*(3+7)
    assert np.sum(raw*error)==pytest.approx(scalar)
    for field in range(2):assert normalized[:,field][domain[:,field]].mean()==pytest.approx(1.)


def test_causal_state_input_threshold_permutation_and_future_rejection():
    origin=pd.Timestamp('2030-08-01T01:00:00Z');times=pd.date_range(origin-pd.Timedelta(minutes=60),origin,freq='5min');f=pd.DataFrame({'timestamp':times,'link_id':'a','speed_kmh':67.86,'flow_vph':600.,'occupancy':.1,'pct_observed':100.});net=pd.DataFrame({'lanes':[2],'v_cut':[67.86]},index=['a']);x=causal_state_input(f,['a'],net,origin)
    assert x.shape==(13,1,9) and x[:,:,8].all();np.testing.assert_array_equal(causal_state_input(f.sample(frac=1,random_state=7),['a'],net,origin),x)
    np.testing.assert_array_equal(causal_state_input(f,['a'],net,origin.tz_convert('Asia/Shanghai')),x)
    with pytest.raises(ValueError):causal_state_input(pd.concat([f,f.iloc[[0]].assign(timestamp=origin+pd.Timedelta(minutes=5))]),['a'],net,origin)


def test_current_mse_does_not_fill_unobserved_components():
    a={'truth_current':np.zeros((1,2,2)),'current_domain':np.array([[[1,0],[1,0]]],bool)};p=np.array([[[1.,999.],[2.,999.]]]);assert current_weighted_mse(a,p)==pytest.approx(.54+.46*4)
    a['current_domain'][:,1]=False;assert current_weighted_mse(a,p) is None


def test_paired_state_interval_retains_known_constant_gain_and_checks_population():
    from trafficflow.state_control_eval import paired_state_bootstrap
    dates=pd.date_range('2030-10-01',periods=15).strftime('%Y-%m-%d').tolist();rows=[]
    for date in dates:rows.append({'date':date,'models':{name:{g:{'n':1,'speed_normalized_SSE':error**2,'flow_normalized_SSE':error**2} for g in ['R1','R2','R3']} for name,error in [('a',.5),('b',.25)]}})
    cases=[{'dates':dates,'records':rows}];r=paired_state_bootstrap(cases,'a','b');assert r['gain']==pytest.approx(.25) and r['ci95']==pytest.approx([.25,.25])
    rows[0]['models']['b']['R1']['n']=2
    with pytest.raises(ValueError):paired_state_bootstrap(cases,'a','b')
