import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.optimize import least_squares
from trafficflow.log_demand_od import group_design,demand_objective,log_demand_inverse


def paths():return pd.DataFrame({'origin_zone':['o1','o1','o2','o2'],'destination_zone':['d1','d2','d1','d2']})


@pytest.mark.parametrize('huber',[None,2.5])
def test_log_demand_analytic_gradient_matches_central_differences(huber):
    H,_=group_design(paths());A=sparse.csr_matrix([[1.,1.,0.,0.],[0.,1.,1.,1.]])
    c=np.array([15.,70.]);b=np.array([10.,20.,30.,40.]);z=np.array([.12,-.2,.1,.05])
    value,gradient,_=demand_objective(z,A,c,b,H,.6,.1,huber)
    numeric=[]
    for i in range(len(z)):
        change=np.zeros_like(z);change[i]=1e-6
        numeric.append((demand_objective(z+change,A,c,b,H,.6,.1,huber)[0]-demand_objective(z-change,A,c,b,H,.6,.1,huber)[0])/2e-6)
    np.testing.assert_allclose(gradient,numeric,rtol=2e-6,atol=1e-6)


def test_positive_group_map_matches_independent_numeric_jacobian_trust_region():
    p=paths();H,_=group_design(p);b=np.array([10.,20.,30.,40.]);A=sparse.eye(4,format='csr')
    c=b*np.exp(H@np.array([.2,-.1,.1,-.15]))
    f,info=log_demand_inverse(A,c,b,p,group_sd=.6,noise_cv=.05)
    def residual(z):return np.r_[(b*np.exp(H@z)-c)/(.05*np.maximum(c,10.)),z/.6]
    oracle=least_squares(residual,np.zeros(4),bounds=(-3.,3.),jac='3-point',ftol=1e-12,xtol=1e-12,gtol=1e-12)
    np.testing.assert_allclose(f,b*np.exp(H@oracle.x),rtol=2e-6,atol=1e-6)
    assert info['success'] and (f>0).all()


def test_group_model_rejects_invalid_scale_and_metadata_length():
    with pytest.raises(ValueError,match='Matching'):
        log_demand_inverse(sparse.eye(4),np.ones(4),np.ones(4),paths().iloc[:3])
    with pytest.raises(ValueError,match='Positive'):
        log_demand_inverse(sparse.eye(4),np.ones(4),np.ones(4),paths(),group_sd=0.)


def test_origin_destination_gauge_changes_parameters_without_changing_flows():
    H,columns=group_design(paths())
    gauge=np.array([1. if name=='origin_zone' else -1. for name,label in columns])
    np.testing.assert_array_equal(H@gauge,np.zeros(4))
    z=np.array([.2,-.1,.1,-.15]);b=np.array([10.,20.,30.,40.])
    np.testing.assert_allclose(b*np.exp(H@z),b*np.exp(H@(z+.7*gauge)),rtol=1e-15)
    assert np.dot(z,z)!=pytest.approx(np.dot(z+.7*gauge,z+.7*gauge))
