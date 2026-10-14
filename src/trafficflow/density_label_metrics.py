"""Public TRAIN observed-increment diagnostics, never formal Physics scores."""
import numpy as np


def public_increment_stats(day, prediction, factor):
    truth = np.asarray(day['y_full'], float)
    estimate = truth.copy().reshape(-1, 2)
    indices = day['target_indices']
    sums, counts = np.zeros_like(estimate), np.zeros(len(estimate))
    np.add.at(sums, indices, np.asarray(prediction, float)/np.array([25., 600.]))
    np.add.at(counts, indices, 1.)
    selected = counts > 0
    estimate[selected] = sums[selected]/counts[selected, None]
    estimate = estimate.reshape(truth.shape)
    def accumulation(x):
        return np.maximum(x[..., 1]*600., 0.)/np.maximum(x[..., 0]*25., 1.)*factor
    expected, observed = accumulation(estimate), accumulation(truth)
    difference, observed_delta = np.diff(expected, axis=0), np.diff(observed, axis=0)
    valid = day['lwr_pair_valid']
    return {'regime': str(day['regime']), 'error_L1': float(np.abs(difference-observed_delta)[valid].sum()),
            'observation_increment_L1': float(np.abs(observed_delta)[valid].sum()),
            'n_public_valid_pairs': int(valid.sum())}


def aggregate_public_increments(rows):
    values = []
    for regime in ['R1', 'R2', 'R3']:
        selected = [r for r in rows if r['regime'] == regime]
        if not selected:
            raise ValueError('Public increment diagnostic needs all three regimes')
        values.append({'regime': regime, 'relative_L1': sum(r['error_L1'] for r in selected)/max(
            sum(r['observation_increment_L1'] for r in selected), 1e-9),
            'n_pairs': sum(r['n_public_valid_pairs'] for r in selected)})
    return {'public_coverage_increment_relative_L1': float(np.mean([r['relative_L1'] for r in values])),
            'public_increment_by_regime': values}
