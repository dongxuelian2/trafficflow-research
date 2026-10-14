import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import huber
from trafficflow.zero_aware_od import zero_aware_inverse, prior_top_mass


def test_zero_aware_map_matches_independent_bound_optimization():
    rng = np.random.default_rng(1369)
    A = (rng.random((5, 12)) < .4).astype(float)
    b = rng.uniform(0., 30., 12)
    c = A@rng.uniform(0., 20., 12)
    c[2] *= 1.6
    sd = np.full_like(b, .75*b.mean())
    design, y = A*sd[None, :]/(.08*np.maximum(c, 10.))[:, None], c/(.08*np.maximum(c, 10.))
    for delta in (None, 1.5):
        def objective(z):
            residual = design@z-y
            loss = .5*residual**2 if delta is None else huber(delta, residual)
            derivative = residual if delta is None else np.clip(residual, -delta, delta)
            return loss.sum()+.5*np.square(z-b/sd).sum()+.5*z.sum(), design.T@derivative+z-b/sd+.5
        direct = minimize(objective, np.maximum(b/sd-.5, 0), method='SLSQP', jac=True,
            bounds=[(0, None)]*len(b), options={'ftol': 1e-12, 'maxiter': 2000})
        assert direct.success
        f, info = zero_aware_inverse(sparse.csr_matrix(A), c, b, noise_cv=.08, huber_delta=delta)
        np.testing.assert_allclose(f, sd*direct.x, rtol=0, atol=2e-4)
        assert info['original_objective'] == pytest.approx(direct.fun, abs=1e-7)
        assert info['scaled_projected_KKT_max_abs'] <= 2e-5


def test_unmeasured_path_has_zero_boundary_solution_without_positive_floor():
    b = np.array([4., 1., 12.])
    f, _ = zero_aware_inverse([[1., 0., 0.]], [5.], b)
    sd = .75*b.mean()
    assert f[1] == 0.
    assert f[2] == pytest.approx(12.-.5*sd)
    assert prior_top_mass([1., 2., 3., 4., 10.], .2) == pytest.approx(.5)
    assert prior_top_mass([0., 0.]) == 0.
    with pytest.raises(ValueError, match='nonnegative'):
        zero_aware_inverse([[1.]], [2.], [-1.])
    with pytest.raises(ValueError, match='scales'):
        zero_aware_inverse([[1.]], [2.], [1.], path_cv=0.)
