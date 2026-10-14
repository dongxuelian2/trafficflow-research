import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize, brentq
from trafficflow.entropy_od import entropy_inverse


def test_entropy_dual_matches_independent_path_space_minimization():
    rng = np.random.default_rng(91741)
    for regularization in (.005, .05, 1.):
        A = (rng.random((5, 13)) < .4).astype(float)
        b = rng.uniform(0, 20, 13)
        b[0] = 0.
        c = A@rng.uniform(0, 25, 13)
        reference = np.maximum(b, 2.)
        variance = regularization*np.maximum(c, 10.)
        def objective(f):
            residual = A@f-c
            loss = .5*np.dot(residual/variance, residual)+np.sum(f*np.log(f/reference)-f+reference)
            gradient = A.T@(residual/variance)+np.log(f/reference)
            return loss, gradient
        direct = minimize(objective, reference, jac=True, method='L-BFGS-B', bounds=[(1e-12, None)]*len(b),
            options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50})
        assert direct.success
        f, info = entropy_inverse(sparse.csr_matrix(A), c, b, regularization)
        np.testing.assert_allclose(f, direct.x, rtol=0, atol=2e-5)
        assert objective(f)[0] == pytest.approx(direct.fun, abs=1e-8)
        assert info['primal_stationarity_max_abs'] <= 2e-6
        assert np.all(f > 0)


def test_entropy_unmeasured_paths_keep_explicit_floor_and_invalid_data_rejected():
    f, info = entropy_inverse(sparse.csr_matrix([[1., 0.]]), [20.], [10., 0.], prior_floor=3.)
    assert f[1] == pytest.approx(3.)
    assert info['n_dual_variables'] == 1
    for kwargs in ({'regularization': 0.}, {'prior_floor': -1.}):
        with pytest.raises(ValueError, match='Positive'):
            entropy_inverse(sparse.eye(2), [2., 3.], [1., 1.], **kwargs)
    with pytest.raises(ValueError, match='nonnegative'):
        entropy_inverse(sparse.eye(2), [-1., 3.], [1., 1.])


def test_entropy_large_cancellation_keeps_original_stationarity_tolerance():
    prior = np.full(4, 1e8)
    counts = np.array([12.])
    regularization = .005
    variance = regularization*12.
    optimum_dual = brentq(lambda u: 12.-prior.sum()*np.exp(-u)+variance*u, 0., 30., xtol=1e-14)
    expected = prior*np.exp(-optimum_dual)
    f, info = entropy_inverse(sparse.csr_matrix(np.ones((1, 4))), counts, prior, regularization)
    np.testing.assert_allclose(f, expected, rtol=0, atol=1e-6)
    assert info['primal_stationarity_max_abs'] <= 2e-6
