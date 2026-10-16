"""Exact affine Gaussian boundary elimination; no giant covariance inverse."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class BoundaryMarginal:
    """Gamma=diag(q)+B L L' B', with independent boundary/noise priors.

    A prior factor L supports positive semidefinite Sigma. q must be positive.
    The formula is exact only for fixed affine dynamics/observation branch.
    """
    def __init__(self, B, q, prior_factor, mean=None):
        B, q, L = np.asarray(B, float), np.asarray(q, float), np.asarray(prior_factor, float)
        if B.ndim != 2 or q.shape != (B.shape[0],) or L.ndim != 2 or L.shape[0] != B.shape[1]:
            raise ValueError('Boundary covariance dimensions disagree')
        if not all(np.isfinite(a).all() for a in [B, q, L]) or (q <= 0).any():
            raise ValueError('Q must be positive definite; finite boundary factor required')
        self.B, self.q, self.L = B, q, L
        self.mean = np.zeros(B.shape[1]) if mean is None else np.asarray(mean, float)
        if self.mean.shape != (B.shape[1],) or not np.isfinite(self.mean).all():
            raise ValueError('Invalid prior mean')
        self.C = B@L
        self.weighted_C = self.C/q[:, None]
        self.small = np.eye(L.shape[1])+self.C.T@self.weighted_C
        self.cholesky = cho_factor(self.small, lower=True)
        self.logdet = float(np.log(q).sum()+2*np.log(np.diag(self.cholesky[0])).sum())

    def precision_apply(self, r):
        r = np.asarray(r, float)
        weighted = r/self.q if r.ndim == 1 else r/self.q[:, None]
        return weighted-self.weighted_C@cho_solve(self.cholesky, self.C.T@weighted)

    def loss(self, d, include_logdet=True):
        r = np.asarray(d, float)-self.B@self.mean
        return .5*float(r@self.precision_apply(r))+(.5*self.logdet if include_logdet else 0.)

    def joint_map(self, d):
        r = np.asarray(d, float)-self.B@self.mean
        z = cho_solve(self.cholesky, self.C.T@(r/self.q))
        eta = self.mean+self.L@z
        residual = np.asarray(d)-self.B@eta
        profiled = .5*float(np.square(residual)/self.q @ np.ones_like(self.q))+.5*float(z@z)
        return eta, profiled


def nuisance_information(Jx, Jeta, tolerance=1e-8):
    """Data-only effective information; prior stabilization is not identifiability."""
    Jx, Jeta = np.asarray(Jx, float), np.asarray(Jeta, float)
    if Jx.ndim != 2 or Jeta.ndim != 2 or Jx.shape[0] != Jeta.shape[0]:
        raise ValueError('Jacobian rows differ')
    u, s, _ = np.linalg.svd(Jeta, full_matrices=False)
    rank = int((s > tolerance*max(float(s.max(initial=0)), 1.)).sum())
    projected = Jx-u[:, :rank]@(u[:, :rank].T@Jx)
    effective = np.linalg.svd(projected, compute_uv=False)
    return {'nuisance_rank': rank, 'nuisance_singular_values': s.tolist(),
            'effective_singular_values': effective.tolist(),
            'effective_rank': int((effective > tolerance*max(float(effective.max(initial=0)), 1.)).sum())}
