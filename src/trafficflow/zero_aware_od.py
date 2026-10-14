"""Nonnegative Gaussian/exponential-prior OD MAP with optional Huber counts.

Boundary MAP values can be exactly zero. This is not a discrete hurdle model
and does not report posterior path-activation probabilities or hidden truth.
"""
import numpy as np
from scipy import sparse
from .odme import bounded_ridge, solve


def prior_top_mass(prior, fraction=.15):
    b = np.asarray(prior, float)
    if b.ndim != 1 or not len(b) or not np.isfinite(b).all() or (b < 0).any() or not 0 < fraction <= 1:
        raise ValueError('Finite nonnegative nonempty prior and valid fraction required')
    number = max(1, int(np.ceil(len(b)*fraction)))
    return float(np.partition(b, len(b)-number)[-number:].sum()/b.sum()) if b.sum() else 0.


def apply_selector(A, counts, prior, config):
    """Frozen rule uses weak-prior concentration only; never truth or labels."""
    concentration = prior_top_mass(prior)
    kind, threshold = config['kind'], config.get('threshold')
    if kind not in {'ridge', 'zero_aware'} or threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Invalid OD selector configuration')
    active = kind == 'zero_aware' and (threshold is None or concentration >= threshold)
    if active:
        f, info = zero_aware_inverse(A, counts, prior, **config['settings'])
    else:
        f, info = solve(A, counts, prior, 'ridge', .05)
    return f, {**info, 'prior_top15_mass': concentration, 'selector_applied': active,
        'actual_solver': 'zero_aware_MAP' if active else 'ridge_0.05'}


def zero_aware_inverse(A, counts, prior, path_cv=.75, scale_mode='absolute',
                       shrinkage=.5, noise_cv=.02, huber_delta=None, maxiter=100):
    """Minimize count loss + .5 sum((f-b)^2/sd^2) + eta sum(f/sd), f>=0.

    z=f/sd yields a unit quadratic and a shifted prior center b/sd-eta.
    Absolute sd=path_cv*max(mean(b),2); relative sd=path_cv*max(b,2).
    Counts use noise_cv*max(c,10). Huber IRLS solves each nonnegative quadratic
    exactly, then certifies stationarity of the original Huber objective.
    """
    A = sparse.csr_matrix(A, dtype=float)
    c, b = np.asarray(counts, float), np.asarray(prior, float)
    if c.ndim != 1 or b.ndim != 1 or A.shape != (len(c), len(b)) or min(A.shape) < 1:
        raise ValueError('Matching nonempty incidence/vector dimensions required')
    if not np.isfinite(A.data).all() or (A.data < 0).any() or not np.isfinite(c).all() or not np.isfinite(b).all() or (c < 0).any() or (b < 0).any():
        raise ValueError('Finite nonnegative incidence, counts and priors required')
    if not np.isfinite([path_cv, noise_cv, shrinkage]).all() or min(path_cv, noise_cv) <= 0 or shrinkage < 0 or maxiter < 1:
        raise ValueError('Positive scales/budget and nonnegative shrinkage required')
    if huber_delta is not None and (not np.isfinite(huber_delta) or huber_delta <= 0):
        raise ValueError('Positive finite Huber threshold required')
    if scale_mode == 'absolute':
        sd = np.full_like(b, path_cv*max(float(b.mean()), 2.))
    elif scale_mode == 'relative':
        sd = path_cv*np.maximum(b, 2.)
    else:
        raise ValueError('Unknown prior scale mode')
    count_sd = noise_cv*np.maximum(c, 10.)
    design = A.multiply(sd).multiply((1/count_sd)[:, None]).tocsr()
    y = c/count_sd
    center = b/sd-shrinkage
    z = np.maximum(center, 0.)
    pivots = 0
    for iteration in range(maxiter if huber_delta is not None else 1):
        residual = np.asarray(design@z)-y
        weight = np.ones_like(y) if huber_delta is None else np.minimum(1., huber_delta/np.maximum(np.abs(residual), 1e-30))
        root = np.sqrt(weight)
        z, solved = bounded_ridge(design.multiply(root[:, None]), y*root, center, 1., np.ones_like(z))
        pivots += solved['iterations']
        residual = np.asarray(design@z)-y
        derivative = residual if huber_delta is None else np.clip(residual, -huber_delta, huber_delta)
        gradient = np.asarray(design.T@derivative)+z-center
        projected = np.where(z > 1e-8, gradient, np.minimum(gradient, 0.))
        certificate = float(np.max(np.abs(projected)))
        if certificate <= 2e-5:
            loss = .5*residual**2 if huber_delta is None else np.where(np.abs(residual) <= huber_delta,
                .5*residual**2, huber_delta*(np.abs(residual)-.5*huber_delta))
            f = sd*z
            return f, {'success': True, 'solver': 'Scaled_nonnegative_MAP_IRLS_measurement_space',
                'outer_iterations': iteration+1, 'total_bound_pivot_iterations': pivots,
                'scaled_projected_KKT_max_abs': certificate,
                'original_objective': float(loss.sum()+.5*np.square(z-b/sd).sum()+shrinkage*z.sum()),
                'path_cv': path_cv, 'scale_mode': scale_mode, 'shrinkage': shrinkage,
                'noise_cv': noise_cv, 'huber_delta': huber_delta,
                'zero_path_fraction': float((f <= 1e-8).mean()),
                'uncertainty': 'MAP point only; no spike probability or calibrated posterior claim'}
    raise RuntimeError('Huber MAP reached iteration budget without original-objective KKT certificate')
