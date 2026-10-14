"""Frozen weak-prior-only routing between independently implemented OD MAPs."""
import math
from .zero_aware_od import zero_aware_inverse, prior_top_mass
from .entropy_od import entropy_inverse
from .odme import solve


def prior_expert_inverse(A, counts, prior, config):
    concentration = prior_top_mass(prior)
    threshold = config.get('threshold')
    if config['fallback'] not in {'ridge', 'entropy'} or threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Invalid OD expert fallback or concentration threshold')
    active = config.get('alternative') is not None and (threshold is None or concentration >= threshold)
    if active:
        f, info = zero_aware_inverse(A, counts, prior, **config['alternative'])
        method = 'zero_aware_MAP'
    elif config['fallback'] == 'entropy':
        f, info = entropy_inverse(A, counts, prior, regularization=.05, prior_floor=20.)
        method = 'entropy_D08'
    elif config['fallback'] == 'ridge':
        f, info = solve(A, counts, prior, 'ridge', .05)
        method = 'ridge_0.05'
    else:
        raise ValueError('Unknown OD expert fallback')
    return f, {'solver_info': info, 'prior_top15_mass': concentration,
        'alternative_applied': active, 'actual_solver': method}
