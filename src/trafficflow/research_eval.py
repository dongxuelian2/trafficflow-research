"""Independent research evaluation. No hidden labels or leaderboard tuning.

Metrics in this module have explicit provenance; simulated metrics are never
reported as Kaggle component scores. Time blocks, rather than cells, are the
unit of statistical resampling.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Fold:
    name: str
    profile_end: str
    fit_end: str
    selection_end: str
    assessment_end: str

    def assign(self, dates):
        boundaries = [self.profile_end, self.fit_end, self.selection_end, self.assessment_end]
        if boundaries != sorted(set(boundaries)):
            raise ValueError('Fold boundaries must be strictly increasing')
        result = {key: [] for key in ('profile', 'fit', 'selection', 'assessment')}
        for date in sorted(set(dates)):
            for key, end in zip(result, boundaries):
                if date <= end:
                    result[key].append(date)
                    break
        if any(not values for values in result.values()):
            raise ValueError('Every fold partition must contain dates')
        return result


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def paired_block_interval(a, b, seed=20261013, repetitions=4000, block_days=7):
    """Paired moving-block bootstrap on chronologically ordered day scores.

    This measures day-mean diagnostic gain, not the nonlinear pooled official
    RMSE score. Candidate selection still uses regime-pooled sufficient stats.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError('Expected matching ordered daily scores')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite daily score')
    delta = b - a
    rng = np.random.default_rng(seed)
    length = min(block_days, len(a))
    starts = rng.integers(0, len(a), (repetitions, int(np.ceil(len(a) / length))))
    indices = (starts[..., None] + np.arange(length)) % len(a)
    means = delta[indices.reshape(repetitions, -1)[:, :len(a)]].mean(axis=1)
    return {'paired_day_mean_gain': float(delta.mean()),
            'ci95': np.quantile(means, [.025, .975]).tolist(),
            'bootstrap_probability_gain_positive': float((means > 0).mean()),
            'n_days': len(a), 'block_days': length, 'resamples': repetitions,
            'note': 'Day-mean diagnostic, not a confidence interval for Kaggle score.'}


def state_from_sufficient_stats(rows):
    """Pool squared errors inside each regime before equal regime averaging."""
    scores = []
    details = []
    for regime in ('R1', 'R2', 'R3'):
        subset = [r for r in rows if r['regime'] == regime]
        n = sum(r['n'] for r in subset)
        if not n:
            raise ValueError(f'No targets for {regime}; cannot compute comparable State')
        rv = np.sqrt(sum(r['speed_sse'] for r in subset) / n)
        rq = np.sqrt(sum(r['lane_flow_sse'] for r in subset) / n)
        score = .54 * max(0, 1 - rv / 25) + .46 * max(0, 1 - rq / 600)
        scores.append(score)
        details.append({'regime': regime, 'n': n, 'rmse_speed': float(rv),
                        'rmse_lane_flow': float(rq), 'S_state': float(score)})
    return {'S_state': float(np.mean(scores)), 'details': details}


def register_protocol(path: Path, value):
    """Write once: subsequent runs must use the exact registered protocol."""
    envelope = {'protocol_sha256': canonical_hash(value), 'protocol': value}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != envelope:
            raise ValueError('Registered protocol differs; create a new version')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(envelope, stream, indent=2, ensure_ascii=False)
    return envelope
