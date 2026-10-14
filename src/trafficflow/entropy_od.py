"""Sparse entropy-regularized OD inversion in the measurement-space dual.

No reference path flows are accessed. Positive weak-prior floors are explicit
model assumptions, and their bias is assessed on independent known truth.
"""
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


def entropy_inverse(A, counts, prior, regularization=.05, prior_floor=2., maxiter=100):
    """Minimize .5 sum((Af-c)^2/variance) + KL(f || max(b,floor)).

    variance=regularization*max(c,10). The dual has only n_counted_links
    variables. Newton steps solve A diag(f) A.T + diag(variance), with Armijo
    backtracking and a primal stationarity certificate; no path-space Hessian.
    """
    A = sparse.csr_matrix(A, dtype=float)
    c, b = np.asarray(counts, float), np.asarray(prior, float)
    if A.shape != (len(c), len(b)) or not len(c) or not len(b):
        raise ValueError('OD incidence and vector dimensions must agree')
    if not np.isfinite(A.data).all() or (A.data < 0).any() or not np.isfinite(c).all() or not np.isfinite(b).all() or (c < 0).any() or (b < 0).any():
        raise ValueError('OD observations and incidence must be finite and nonnegative')
    if not np.isfinite(regularization) or not np.isfinite(prior_floor) or regularization <= 0 or prior_floor <= 0 or maxiter < 1:
        raise ValueError('Positive regularization, prior floor and iteration budget required')
    reference = np.maximum(b, prior_floor)
    variance = regularization*np.maximum(c, 10.)
    dual = np.zeros(len(c), float)
    backtracks = 0
    precision_fallbacks = 0
    def evaluate(u):
        exponent = -np.asarray(A.T@u)
        if np.max(np.abs(exponent)) > 650:
            return None
        f = reference*np.exp(exponent)
        if not np.isfinite(f).all():
            return None
        # Shift the dual objective by its constant at zero for better precision.
        entropy_term = reference*np.expm1(exponent)
        count_term = c*u
        quadratic_term = .5*variance*u*u
        objective = entropy_term.sum()+count_term.sum()+quadratic_term.sum()
        roundoff = 16*np.finfo(float).eps*(np.abs(entropy_term).sum()+np.abs(count_term).sum()+quadratic_term.sum()+1.)
        gradient = c-np.asarray(A@f)+variance*u
        if not np.isfinite(objective) or not np.isfinite(gradient).all():
            return None
        primal_gradient = np.asarray(A.T@(-gradient/variance))
        return f, float(objective), gradient, float(np.max(np.abs(primal_gradient))), float(roundoff)
    for iteration in range(maxiter):
        current = evaluate(dual)
        if current is None:
            raise RuntimeError('Entropy dual left the finite domain')
        f, value, gradient, certificate, roundoff = current
        if certificate <= 2e-6:
            residual = np.asarray(A@f)-c
            primal = .5*np.dot(residual/variance, residual)+np.sum(f*np.log(f/reference)-f+reference)
            return f, {'success': True, 'solver': 'measurement_space_entropy_Newton',
                'iterations': iteration, 'backtracks': backtracks, 'precision_fallbacks': precision_fallbacks,
                'primal_stationarity_max_abs': certificate,
                'dual_stationarity_max_abs': float(np.max(np.abs(gradient))),
                'primal_objective': float(primal), 'regularization': regularization,
                'prior_floor': prior_floor, 'n_dual_variables': len(c),
                'uncertainty': 'Point estimate only; no identifiable-path or posterior-coverage claim'}
        hessian = (A.multiply(f)@A.T).toarray()
        hessian[np.diag_indices_from(hessian)] += variance
        try:
            step = -cho_solve(cho_factor(hessian, check_finite=False), gradient, check_finite=False)
        except np.linalg.LinAlgError as error:
            raise RuntimeError('Entropy dual Hessian factorization failed') from error
        direction = float(np.dot(gradient, step))
        if not np.isfinite(step).all() or direction >= 0:
            raise RuntimeError('Entropy Newton step failed the descent check')
        length = 1.
        for _ in range(60):
            trial = evaluate(dual+length*step)
            armijo = trial is not None and trial[1] <= value+1e-4*length*direction
            precision_step = trial is not None and -length*direction <= 16*(roundoff+trial[4]) and trial[3] < .5*certificate and trial[1] <= value+roundoff+trial[4]
            if armijo or precision_step:
                dual += length*step
                precision_fallbacks += int(not armijo and precision_step)
                break
            length *= .5
            backtracks += 1
        else:
            raise RuntimeError('Entropy dual line search failed')
    raise RuntimeError('Entropy dual reached the iteration limit without a stationarity certificate')
