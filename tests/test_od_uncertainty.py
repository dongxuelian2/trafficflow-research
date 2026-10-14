import numpy as np
import pytest
from scipy import sparse
from trafficflow.od_uncertainty import entropy_curvature_sd, calibrate_interval_scale


def test_curvature_diagonal_matches_independent_dense_path_hessian_inverse():
    rng = np.random.default_rng(621)
    A = (rng.random((5, 13)) < .4).astype(float)
    A[:, -1] = 0.
    f, variance = rng.uniform(.1, 30., 13), rng.uniform(.1, 4., 5)
    expected = np.diag(np.linalg.inv(np.diag(1/f)+A.T@np.diag(1/variance)@A))
    sd, info = entropy_curvature_sd(sparse.csr_matrix(A), f, variance, block_size=3)
    np.testing.assert_allclose(sd**2, expected, atol=1e-11, rtol=1e-11)
    assert sd[-1]**2 == pytest.approx(f[-1])
    assert 0 <= info['mean_relative_variance_reduction'] <= 1
    with pytest.raises(ValueError, match='positive'):
        entropy_curvature_sd(A, f, np.zeros(5))


def test_empirical_calibration_weights_cases_equally_and_seals_global_scale():
    records = [{'regime': 'a', 'standardized_errors': np.array([1., 2.])},
               {'regime': 'a', 'standardized_errors': np.full(100, 10.)},
               {'regime': 'b', 'standardized_errors': np.array([3., 4.])}]
    cal = calibrate_interval_scale(records, .75)
    assert cal['multiplier'] == pytest.approx(10.)
    assert min(cal['development_coverage_by_regime'].values()) >= .75
    assert cal['n_calibration_cases'] == 3
    with pytest.raises(ValueError, match='Finite'):
        calibrate_interval_scale([{'regime': 'a', 'standardized_errors': [np.inf]}])
