import numpy as np
import pytest
from trafficflow.density_assimilation import assimilate_density


def test_learned_increment_assimilation_keeps_visible_anchors_exact():
    prior=np.array([[0.],[100.],[0.]])
    observed=np.array([[1.],[np.nan],[3.]])
    target=np.array([[False],[True],[False]])
    delta=np.ones((2,1));valid=np.ones((2,1),bool)
    result=assimilate_density(prior,observed,target,delta,valid,10.)
    np.testing.assert_allclose(result[:,0],[1.,140/21,3.])
    np.testing.assert_allclose(assimilate_density(prior,observed,target,delta,valid,0.)[:,0],[1.,100.,3.])
    observed[0]=np.nan
    with pytest.raises(ValueError,match='Unavailable'):
        assimilate_density(prior,observed,target,delta,valid,10.)
    valid[0]=False
    result=assimilate_density(prior,observed,target,delta,valid,10.)
    assert np.isnan(result[0,0]) and result[2,0]==3.
