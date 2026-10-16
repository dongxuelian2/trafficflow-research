"""Small frozen-predictor corrections and nonlinear macro bootstrap."""
import numpy as np
from scipy.optimize import minimize
from .score_aligned import state_macro


class LowRankCorrection:
    """Y=Y0+U alpha; macro RMSE + optional increment L1 + sparse penalty.

    The increment loss here is a local first-order surrogate with exact
    anchored held-out evaluation. This is a restricted reweighting of prior
    density-increment research, not new physical information.
    """
    def __init__(self, increment_weight=0., l1=1e-5, ridge=1e-3):
        self.increment_weight, self.l1, self.ridge = increment_weight, l1, ridge

    def fit(self, days):
        k = days[0]['basis'].shape[-1]
        def objective(flat):
            coefficient = flat.reshape(k, 2)
            loss, grad = 0., np.zeros_like(coefficient)
            for regime in ['R1', 'R2', 'R3']:
                subset = [d for d in days if d['regime'] == regime]
                n = sum(len(d['base']) for d in subset)
                if not n:
                    raise ValueError('Correction-fit lacks a regime')
                errors = [d['base']+d['basis']@coefficient-d['truth'] for d in subset]
                rmse = np.sqrt(sum(np.square(e).sum(axis=0) for e in errors)/n+1e-12)
                weight = np.array([.54, .46])/3
                loss += float(weight@rmse)
                for d, e in zip(subset, errors):
                    grad += d['basis'].T@(e*(weight/(n*rmse)))
            for d in days:
                if self.increment_weight and len(d['increment_residual']):
                    delta = d['increment_residual']+d['increment_design']@flat
                    scale = self.increment_weight/(len(days)*max(d['increment_denominator'], 1e-6))
                    # Smooth L1 only at the kink, never official score substitution.
                    smooth = np.sqrt(delta*delta+1e-8)
                    loss += scale*float(smooth.sum())
                    grad += (scale*(d['increment_design'].T@(delta/smooth))).reshape(k, 2)
            smooth = np.sqrt(flat*flat+1e-10)
            loss += self.l1*smooth.sum()+.5*self.ridge*np.square(flat).sum()
            g = grad.ravel()+self.l1*flat/smooth+self.ridge*flat
            return float(loss), g
        result = minimize(objective, np.zeros(k*2), jac=True, method='L-BFGS-B',
                          bounds=[(-.25, .25)]*(k*2), options={'maxiter': 120, 'ftol': 1e-10})
        self.coefficient = result.x.reshape(k, 2)
        self.fit_record = {'success': bool(result.success), 'message': str(result.message), 'iterations': result.nit, 'objective': float(result.fun)}
        if not np.isfinite(result.x).all():
            raise ValueError('Nonfinite correction')
        return self

    def predict(self, base, basis, strength=1.):
        return base+strength*basis@self.coefficient


def paired_macro_bootstrap(control, candidate, families, seed=20261016, repetitions=2000, block_days=7, date_strata=None):
    dates = sorted({r['date'] for r in control})
    lookup = {(r['panel'], r['date']): r for r in candidate}
    if len(lookup) != len(candidate) or len({(r['panel'], r['date']) for r in control}) != len(control):
        raise ValueError('Expected one sufficient-statistics row per panel/date')
    if {(r['panel'], r['date']) for r in control} != set(lookup):
        raise ValueError('Unpaired assessment records')
    rng, samples, excluded = np.random.default_rng(seed), [], 0
    if date_strata is not None and set(date_strata) != set(dates):
        raise ValueError('Every paired date needs a bootstrap stratum')
    strata = [dates] if date_strata is None else [[d for d in dates if date_strata[d] == s] for s in sorted(set(date_strata.values()))]
    for _ in range(repetitions):
        sampled = []
        for calendar in strata:
            length = min(block_days, len(calendar))
            starts = rng.integers(0, len(calendar), int(np.ceil(len(calendar)/length)))
            sampled.extend(np.asarray(calendar)[((starts[:, None]+np.arange(length)) % len(calendar)).ravel()[:len(calendar)]])
        chosen = np.asarray(sampled)
        counts = {d: int(np.count_nonzero(chosen == d)) for d in set(chosen)}
        a, b = [], []
        for r in control:
            for _ in range(counts.get(r['date'], 0)):
                a.append(r); b.append(lookup[(r['panel'], r['date'])])
        # A bootstrap missing an entire panel/regime is not a comparable macro.
        groups = {(r['panel'], r['regime']) for r in a}
        if any((p, g) not in groups for p in families for g in ['R1', 'R2', 'R3']):
            excluded += 1
            continue
        samples.append(state_macro(b, families)['S_state']-state_macro(a, families)['S_state'])
    point = state_macro(candidate, families)['S_state']-state_macro(control, families)['S_state']
    return {'pooled_macro_gain': point, 'ci95': np.quantile(samples, [.025, .975]).tolist() if samples else None,
            'accepted_resamples': len(samples), 'excluded_missing_regime': excluded, 'requested_resamples': repetitions,
            'block_days': block_days, 'strata': None if date_strata is None else sorted(set(date_strata.values())),
            'unit': 'paired calendar date; pooled macro recomputed inside every draw'}
