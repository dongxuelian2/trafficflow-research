"""LTM prerequisite checks. Full transport assimilation is conditional on them.

The cumulative identity is necessary, not sufficient for a triangular-FD LTM.
No forecast is manufactured when detector/transport mapping is unverified.
"""
import numpy as np


def cumulative_inventory(initial_inventory, cumulative_entries, cumulative_exits):
    entries, exits = np.asarray(cumulative_entries, float), np.asarray(cumulative_exits, float)
    if entries.shape != exits.shape or np.any(np.diff(entries, axis=0) < -1e-10) or np.any(np.diff(exits, axis=0) < -1e-10):
        raise ValueError('Cumulative boundary counts must have identical shapes and be nondecreasing')
    inventory = np.asarray(initial_inventory)+entries-exits
    if (inventory < -1e-10).any():
        raise ValueError('Boundary curves imply negative inventory')
    return inventory


def prerequisite_decision(observation_flux_relative_error, unresolved_fraction, prior_gain_signs=None):
    reasons = []
    if not np.isfinite(observation_flux_relative_error) or observation_flux_relative_error >= 1.:
        reasons.append('Detector-average flow does not close observed inventory increments; transport observation operator remains unverified')
    if unresolved_fraction > .5:
        reasons.append('More than half measured links have unresolved internal boundaries; no certified global transport domain')
    if prior_gain_signs is not None and min(prior_gain_signs) < 0 < max(prior_gain_signs):
        reasons.append('Gain changes sign under boundary-prior sensitivity')
    return {'launch_full_ltm_mhe': not reasons, 'reasons': reasons,
            'scope': 'Conservative engineering launch gate, not a theorem that LTM cannot work'}
