"""Hierarchical Gaussian OD prior with constrained posterior MAP and uncertainty.

Uncertainty is a Gaussian approximation before nonnegative truncation, not a
claim of identifiable hidden paths. All hyperparameters are explicit.
"""
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def posterior(A, counts, prior, paths, path_cv=.20, group_cv=.08,
              noise_cv=.02, prior_floor=2., maxiter=1000):
    A = sparse.csr_matrix(A)
    b, c = np.asarray(prior, float), np.asarray(counts, float)
    if min(path_cv, noise_cv, prior_floor) <= 0 or group_cv < 0:
        raise ValueError('Invalid uncertainty hyperparameters')
    diagonal = (path_cv * np.maximum(b, prior_floor)) ** 2
    group_columns = []
    for key in ('origin_zone', 'destination_zone'):
        labels = paths[key].to_numpy(str)
        for label in sorted(set(labels)):
            group_columns.append(np.where(labels == label, b * group_cv, 0.))
    U = np.stack(group_columns, axis=1) if group_columns else np.empty((len(b), 0))
    AU = np.asarray(A @ U)
    cross = A.T.toarray() * diagonal[:, None] + U @ AU.T
    variance = (noise_cv * np.maximum(c, 10.)) ** 2
    G = np.asarray(A @ cross)
    G[np.diag_indices_from(G)] += variance
    factor = cho_factor(G, check_finite=False)
    innovation = c - A @ b
    mean = b + cross @ cho_solve(factor, innovation, check_finite=False)
    sd2 = diagonal + np.square(U).sum(axis=1)
    sd2 -= np.einsum('ij,ji->i', cross, cho_solve(factor, cross.T, check_finite=False))
    info = {'nonnegative_optimization': False, 'iterations': 0}
    if (mean < 0).any():
        invdU = U / diagonal[:, None]
        small_factor = cho_factor(np.eye(U.shape[1]) + U.T @ invdU, check_finite=False)
        scale = np.sqrt(diagonal + np.square(U).sum(axis=1))

        def objective(z):
            f = z * scale
            difference = f - b
            precision_difference = difference / diagonal
            precision_difference -= invdU @ cho_solve(small_factor, U.T @ precision_difference, check_finite=False)
            residual = A @ f - c
            loss = .5 * (np.dot(difference, precision_difference) + np.dot(residual / variance, residual))
            grad = precision_difference + A.T @ (residual / variance)
            return loss, grad * scale

        result = minimize(objective, np.maximum(mean, 0) / scale, jac=True,
                          method='L-BFGS-B', bounds=[(0, None)] * len(b),
                          options={'maxiter': maxiter, 'ftol': 1e-11, 'gtol': 1e-6})
        if not result.success:
            raise RuntimeError(f'Constrained OD posterior did not converge: {result.message}')
        mean = result.x * scale
        info = {'nonnegative_optimization': True, 'iterations': int(result.nit)}
    eigenvalues = np.linalg.eigvalsh((A @ A.T).toarray())
    rank = int((eigenvalues > max(eigenvalues.max(), 1) * 1e-10).sum())
    info.update({'matrix_rank': rank, 'nullity': A.shape[1] - rank,
                 'identifiable_fraction': rank / A.shape[1],
                 'posterior_sd_median': float(np.median(np.sqrt(np.maximum(sd2, 0)))),
                 'uncertainty': 'untruncated Gaussian posterior approximation',
                 'path_cv': path_cv, 'group_cv': group_cv, 'noise_cv': noise_cv})
    return mean, np.sqrt(np.maximum(sd2, 0)), info
