import numpy as np
import pandas as pd
import pytest
from trafficflow.onset_cardinality_queue import onset_summaries,fit_count_state,predict_set_sizes,onset_threshold_decisions


def sample():
    x=pd.DataFrame({'current':np.tile([.4,.8,1.],6),'cut_ratio':.6,'trend_forecast':np.tile([.5,.9,1.1],6),'slope':np.tile([-.01,0,.01],6),
        'flow_capacity_ratio':np.tile([.5,1.,np.nan],6),'speed_std':np.tile([.1,.2,.3],6),'coverage':np.tile([1,.5,0],6),'age':np.tile([0,5,65],6),'sin_tod':0.,'cos_tod':1.,'weekday':2.})
    return x,np.full(18,'w'),np.repeat([5,10,15,20,25,30],3),np.tile(['c','a','b'],6)


def test_summary_analytic_and_permutation():
    x,w,t,l=sample();keys,a=onset_summaries(x,w,t,l)
    assert keys.tolist()==['w'] and a.shape==(1,24)
    np.testing.assert_allclose(a[0,[0,1,5,14,15,16,19,20]],[3,-.2,1/3,.75,1,1/3,.5,70/3])
    p=np.random.default_rng(46).permutation(18);_,b=onset_summaries(x.iloc[p].reset_index(drop=True),w[p],t[p],l[p]);np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):onset_summaries(x.iloc[:-1],w[:-1],t[:-1],l[:-1])


def test_fit_only_count_correction_and_bounds():
    x=np.ones((8,24));x[:,0]=10;y=np.full(8,3.);m=np.full(8,5.);n=np.full(8,10.)
    observed=fit_count_state(x,y,m,n);adjusted=fit_count_state(x,y,m,n,adjust=True)
    assert predict_set_sizes(observed,x,n).tolist()==[3]*8
    assert predict_set_sizes(adjusted,x,n).tolist()==[6]*8
    changed=x.copy();changed[:,1]=1e200
    assert ((predict_set_sizes(adjusted,changed,n)>=1)&(predict_set_sizes(adjusted,changed,n)<=n)).all()
    assert observed['n_fit_windows']==8 and not observed['coverage_adjusted']


def test_threshold_full_domain_force_and_future_mask_absent():
    _,w,t,l=sample();p=np.zeros(18);p[-3:]=[.7,.3,.9]
    out=onset_threshold_decisions(p,w,t,l,.5,False);assert out.sum()==2 and not out[t<30].any()
    forced=onset_threshold_decisions(p,w,t,l,.99,True);assert forced.sum()==1 and l[forced==1].tolist()==['b']
    with pytest.raises(TypeError):onset_threshold_decisions(p,w,t,l,.5,False,future_eligible=np.ones(18,bool))
