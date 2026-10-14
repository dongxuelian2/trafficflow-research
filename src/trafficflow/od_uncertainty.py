"""Curvature diagnostics and empirical intervals for entropy OD estimates.

An inverse optimization Hessian is not a calibrated posterior. One global
interval multiplier may be calibrated on explicitly generated development
truth, but carries no coverage guarantee under a new generating distribution.
"""
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, solve_triangular


def entropy_curvature_sd(A, prediction, variance, block_size=1024):
    """Diagonal of (diag(1/f) + A.T diag(1/variance) A)^-1.

    Woodbury avoids an n_paths-square matrix. Measurement-space Cholesky is
    exact for the local Hessian, with finite/positive checks and cancellation
    guards; the statistical model behind that curvature remains an assumption.
    """
    A = sparse.csr_matrix(A, dtype=float)
    f, var = np.asarray(prediction, float), np.asarray(variance, float)
    if f.ndim != 1 or var.ndim != 1 or A.shape != (len(var), len(f)) or min(A.shape) < 1:
        raise ValueError('Matching nonempty incidence/vector dimensions required')
    if not np.isfinite(A.data).all() or not np.isfinite(f).all() or not np.isfinite(var).all() or (A.data < 0).any() or (f <= 0).any() or (var <= 0).any() or block_size < 1:
        raise ValueError('Finite nonnegative incidence and positive curvature values required')
    system = (A.multiply(f)@A.T).toarray()
    system[np.diag_indices_from(system)] += var
    factor, _ = cho_factor(system, lower=True, check_finite=False)
    reduction = np.empty_like(f)
    for first in range(0, len(f), block_size):
        last = min(len(f), first+block_size)
        white = solve_triangular(factor, A[:, first:last].toarray(), lower=True, check_finite=False)
        reduction[first:last] = f[first:last]*np.sum(white*white, axis=0)
    if not np.isfinite(reduction).all() or (reduction < -1e-10).any() or (reduction > 1+1e-8).any():
        raise RuntimeError('Curvature diagonal failed its covariance contraction check')
    diagonal = f*(1-np.clip(reduction, 0., 1.))
    return np.sqrt(diagonal), {'curvature_method': 'Exact local Hessian diagonal via measurement-space Woodbury',
        'mean_relative_variance_reduction': float(reduction.mean()),
        'max_relative_variance_reduction': float(reduction.max()),
        'statistical_scope': 'Optimization curvature only; no calibrated posterior or identifiable-path claim'}


def calibrate_interval_scale(records, target_coverage=.95):
    """Smallest scale >=1 reaching case-mean target in every dev regime.

    Each record is one independent generated case with a regime name and sorted
    |estimate-truth|/(1.96*sd) values. Cells within a case are not counted as
    independent trials. Only supplied development records enter calibration.
    """
    if not records or not 0 < target_coverage < 1:
        raise ValueError('Nonempty calibration records and valid coverage required')
    groups = {}
    for record in records:
        values = np.sort(np.asarray(record['standardized_errors'], float))
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Finite nonnegative calibration errors required')
        groups.setdefault(record['regime'], []).append(values)
    def coverage(scale):
        return {regime: float(np.mean([np.searchsorted(v, scale, side='right')/len(v) for v in cases]))
                for regime, cases in groups.items()}
    lower, upper = 1., 1.
    while min(coverage(upper).values()) < target_coverage:
        upper *= 2.
    for _ in range(60):
        middle = (lower+upper)/2
        if min(coverage(middle).values()) >= target_coverage:
            upper = middle
        else:
            lower = middle
    upper = float(np.nextafter(upper, np.inf))
    return {'multiplier': upper, 'target_case_mean_path_coverage': target_coverage,
        'development_coverage_by_regime': coverage(upper), 'n_calibration_cases': len(records),
        'scope': 'Empirical equal-case development calibration, not exchangeable conformal coverage or a Kaggle confidence interval'}
