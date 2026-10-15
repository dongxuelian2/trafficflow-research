import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.optimize import minimize
from trafficflow.entropy_od import entropy_inverse
from trafficflow.demand_residual_od import residual_from_demand,demand_residual_inverse


def test_zero_blend_exactly_preserves_strong_entropy_and_skips_group_fit(monkeypatch):
    import trafficflow.demand_residual_od as module
    def forbidden(*args,**kwargs):raise AssertionError('Zero blend must not fit a group model')
    monkeypatch.setattr(module,'log_demand_inverse',forbidden)
    A=sparse.csr_matrix([[1.,1.,0.],[0.,1.,1.]])
    c=np.array([80.,100.]);b=np.array([0.,50.,40.])
    f,info=demand_residual_inverse(A,c,b,None,center_blend=0.)
    reference,_=entropy_inverse(A,c,b,regularization=.05,prior_floor=20.)
    np.testing.assert_array_equal(f,reference)
    assert info['group_stage_skipped']


def test_residual_stage_matches_independent_primal_finite_difference_optimizer():
    A=sparse.csr_matrix([[1.,1.,0.,0.],[0.,1.,1.,1.],[1.,0.,0.,1.]])
    b=np.array([10.,30.,50.,60.]);g=np.array([25.,20.,80.,40.]);c=np.array([70.,140.,80.])
    alpha=.5;tau=.5;center=np.sqrt(np.maximum(b,20.)*g);variance=tau*np.maximum(c,10.)
    f,info=residual_from_demand(A,c,b,g,alpha,tau)
    def objective(flow):
        residual=A@flow-c
        return .5*np.sum(residual**2/variance)+np.sum(flow*np.log(flow/center)-flow+center)
    oracle=minimize(objective,center,method='L-BFGS-B',jac='3-point',bounds=[(1e-8,None)]*4,
        options={'ftol':1e-14,'gtol':1e-9,'maxiter':1000})
    assert oracle.success
    np.testing.assert_allclose(f,oracle.x,rtol=2e-6,atol=1e-5)
    assert info['entropy_info']['primal_stationarity_max_abs']<=2e-6


def test_demand_residual_rejects_invalid_centers_and_blends():
    A=sparse.eye(3);c=b=np.ones(3)
    with pytest.raises(ValueError,match='blend'):residual_from_demand(A,c,b,b,1.01)
    with pytest.raises(ValueError,match='Positive'):residual_from_demand(A,c,b,np.array([1.,0.,2.]))
    with pytest.raises(ValueError,match='matching'):residual_from_demand(A,c,b,np.ones(4))
    with pytest.raises(ValueError,match='floor'):residual_from_demand(A,c,b,b,residual_floor=0.)


def test_preserved_center_floor_changes_prior_without_bounding_predictions():
    A=sparse.eye(3,format='csr');c=np.array([.1,.2,.3]);b=np.full(3,100.);g=np.full(3,.1)
    f,info=residual_from_demand(A,c,b,g,.5,.05,residual_floor=20.)
    independent_center=np.maximum(np.sqrt(b*g),20.)
    expected,_=entropy_inverse(A,c,independent_center,regularization=.05,prior_floor=2.)
    np.testing.assert_array_equal(f,expected)
    assert (f<20.).all() and info['center_floor']==20.


def test_shift_generators_are_deterministic_nonnegative_and_distinct():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from od_shift_generators import shifted_synthetic,NEW_REGIMES
    p=pd.DataFrame({'origin_zone':['a','a','b','b','c','c'],'destination_zone':['a','b','b','c','a','c'],
        'link_seq':['1','1;2','2;3','1;2;3','3','2']})
    A=sparse.csr_matrix([[1.,1.,0.,1.,0.,0.],[0.,1.,1.,1.,0.,1.],[0.,0.,1.,1.,1.,0.]])
    signatures=[]
    for regime in NEW_REGIMES:
        first=shifted_synthetic(A,np.full(6,50.),p,11,regime)
        second=shifted_synthetic(A,np.full(6,50.),p,11,regime)
        for x,y in zip(first,second):
            np.testing.assert_array_equal(x,y)
            assert np.isfinite(x).all() and (x>=0).all()
        assert first[0].sum()>0
        signatures.append(b''.join(v.tobytes() for v in first))
    assert len(set(signatures))==len(NEW_REGIMES)
