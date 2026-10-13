import numpy as np
import pandas as pd
from scipy import sparse
from trafficflow.bayesian_od import posterior


def test_od_nullspace_remains_uncertain():
    A=sparse.csr_matrix([[1.,1.,0.],[0.,1.,1.]])
    prior=np.array([4.,3.,2.])
    paths=pd.DataFrame({'origin_zone':['o1','o1','o2'],'destination_zone':['d1','d2','d2']})
    mean,sd,info=posterior(A,np.array([7.2,5.1]),prior,paths,group_cv=0.,noise_cv=.01)
    null=np.array([1.,-1.,1.])
    np.testing.assert_allclose(A@null,0)
    assert info['matrix_rank']==2 and info['nullity']==1
    assert (sd>0).all() and (mean>=0).all()
    # Independent analytical Gaussian posterior for a diagonal prior.
    covariance=np.diag((.2*prior)**2)
    noise=np.diag((.01*np.array([10.,10.]))**2)
    expected=prior+covariance@A.T@np.linalg.solve(A@covariance@A.T+noise,np.array([7.2,5.1])-A@prior)
    np.testing.assert_allclose(mean,expected,atol=1e-10)
