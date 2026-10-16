import numpy as np
import pytest
from trafficflow.v2.boundary_marginal import BoundaryMarginal, nuisance_information
from trafficflow.v2.ltm_assimilation import cumulative_inventory, prerequisite_decision


def test_woodbury_joint_profile_and_dense_gaussian_agree():
    rng = np.random.default_rng(41)
    B, L = rng.normal(size=(13, 4)), rng.normal(size=(4, 3))
    q, d, mean = rng.uniform(.1, 2., 13), rng.normal(size=13), rng.normal(size=4)
    model = BoundaryMarginal(B, q, L, mean)
    gamma = np.diag(q)+B@L@L.T@B.T
    r = d-B@mean
    np.testing.assert_allclose(model.precision_apply(r), np.linalg.solve(gamma, r), atol=1e-11)
    assert model.logdet == pytest.approx(np.linalg.slogdet(gamma)[1], abs=1e-11)
    _, profiled = model.joint_map(d)
    assert profiled == pytest.approx(model.loss(d, include_logdet=False), abs=1e-11)


def test_bad_covariance_and_total_boundary_absorption():
    with pytest.raises(ValueError, match='positive definite'):
        BoundaryMarginal(np.eye(2), [0, 1], np.eye(2))
    result = nuisance_information(np.eye(7), np.eye(7))
    assert result['effective_rank'] == 0
    assert result['nuisance_rank'] == 7


def test_ltm_identity_is_not_transport_oracle():
    np.testing.assert_equal(cumulative_inventory(2, [0, 2, 4], [0, 1, 4]), [2, 3, 2])
    with pytest.raises(ValueError):
        cumulative_inventory(0, [0, 1], [0, 2])
    assert not prerequisite_decision(2., .2)['launch_full_ltm_mhe']
