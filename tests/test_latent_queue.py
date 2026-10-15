import numpy as np
import pytest
from trafficflow.latent_queue import causal_queue_features,gaussian_queue_targets,latent_queue_decisions,window_ious


def test_gaussian_teacher_analytic_and_zero_noise():
    p=gaussian_queue_targets([62,66,70],66,4)
    np.testing.assert_allclose(p,[.841344746,.5,.158655254],atol=1e-7)
    np.testing.assert_array_equal(gaussian_queue_targets([62,66,70],66,0),[1,1,0])
    with pytest.raises(ValueError):gaussian_queue_targets([66],66,-1)


def test_features_missing_history_and_permutation():
    v=np.broadcast_to(np.arange(13)[:,None]+np.array([60.,80.,100.]),(13,3)).copy();q=np.full_like(v,4000);m=np.ones(v.shape,bool);m[:,2]=False;p=np.array([0.,1.,3.])
    a=causal_queue_features(v,q,m,p,100,8000,60,True).reshape(6,3,-1)
    assert a.shape==(6,3,58) and np.isfinite(a).all()
    np.testing.assert_allclose(a[0,:,39],[.72,.92,1.])
    np.testing.assert_allclose(a[0,:,43],[.01,.01,0.],atol=1e-7)
    permutation=[2,0,1];b=causal_queue_features(v[:,permutation],q[:,permutation],m[:,permutation],p[permutation],100,8000,60,True).reshape(6,3,-1)
    np.testing.assert_array_equal(b,a[:,permutation])
    poisoned=v.copy();poisoned[~m]=np.inf
    np.testing.assert_array_equal(causal_queue_features(poisoned,q,m,p,100,8000,60,True).reshape(a.shape),a)
    with pytest.raises(ValueError):causal_queue_features(v[:12],q[:12],m[:12],p,100,8000,60,True)


def test_full_domain_timing_force_and_scorer_separation():
    p=np.full((2,6,2),.1);p[0,0,0]=.99;p[0,5,1]=.2;p[1,2,0]=.4
    prediction=latent_queue_decisions(p,[False,True],{'onset':.5,'ongoing':.5},True)
    assert prediction.sum()==2 and prediction[0,5,1] and prediction[1,2,0] and not prediction[0,:5].any()
    mask=np.ones(p.shape,bool);mask[0,5,1]=False
    assert window_ious(np.zeros(p.shape,bool),prediction,mask)[0]==1
    assert window_ious(np.zeros(p.shape,bool),prediction,np.zeros(p.shape,bool)).tolist()==[0.,0.]
