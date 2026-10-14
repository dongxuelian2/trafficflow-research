"""Public-condition experts and independent-Bernoulli expected-IoU decisions.

Onset timing is public host metadata (discussion 742350). Actual future
locations remain unknown. Only public-train observation labels fit the experts.
"""
import numpy as np
import pandas as pd
from .queue import ProxyData, QueueModel
from .queue_calibration import RefinedQueueModel
from .metrics import grouped_iou


def expected_iou_prefix(probability, require_nonempty=True):
    """Choose a probability-ranked prefix by exact expected IoU.

    For a selected prefix of size k, TP is independent of the number of true
    cells outside the prefix. E[IoU] = E[TP] * E[1/(k + outside_true)].
    The latter distribution is Poisson-binomial. Independence is a modeling
    approximation for traffic; the calculation itself is exact under it.
    """
    p = np.asarray(probability, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected finite one-dimensional probabilities in [0,1]')
    order = np.argsort(-p, kind='stable')
    sorted_p = p[order]
    prefix = np.cumsum(sorted_p)
    tail = np.ones(1)
    scores = np.zeros(len(p) + 1)
    for k in range(len(p), 0, -1):
        scores[k] = prefix[k - 1] * np.dot(tail, 1 / (k + np.arange(len(tail))))
        value = sorted_p[k - 1]
        tail = np.r_[tail * (1 - value), 0] + np.r_[0, tail * value]
    scores[0] = 0 if require_nonempty else tail[0]
    offset = 1 if require_nonempty else 0
    k = int(np.argmax(scores[offset:]) + offset)
    prediction = np.zeros(len(p), dtype=int)
    prediction[order[:k]] = 1
    return prediction, {'selected_size': k, 'expected_IoU': float(scores[k]),
                        'assumption': 'Independent Bernoulli cells; no hidden truth'}


def event_prefix_decisions(probability, window, condition, lead_minutes):
    p = np.asarray(probability, float)
    window, condition, lead = map(np.asarray, (window, condition, lead_minutes))
    if p.ndim != 1 or any(a.shape != p.shape for a in (window, condition, lead)):
        raise ValueError('Queue decision shape mismatch')
    if not np.isin(lead, [5, 10, 15, 20, 25, 30]).all():
        raise ValueError('Unexpected Queue lead')
    result = np.zeros(len(p), dtype=int)
    for indices in pd.DataFrame({'window': window}).groupby('window', sort=False).indices.values():
        c = np.unique(condition[indices])
        if len(c) != 1 or c[0] not in ('queue_onset', 'queue_ongoing'):
            raise ValueError('Each window requires one published condition')
        allowed = indices[lead[indices] == 30] if c[0] == 'queue_onset' else indices
        if not len(allowed):
            raise ValueError('Onset window requires final-step candidates')
        result[allowed] = expected_iou_prefix(p[allowed], True)[0]
    return result


class OnsetSpecialistQueueModel(RefinedQueueModel):
    """One final-step onset expert and one six-step ongoing expert."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = {}
        self.uses_public_onset_timing = True
        self.training_counts = {}

    def fit(self, data):
        for condition in ('queue_onset', 'queue_ongoing'):
            use = data.condition == condition
            if condition == 'queue_onset':
                use &= data.x.lead_minutes.to_numpy() == 30
            if not use.any() or len(np.unique(data.y[use])) < 2:
                raise ValueError(f'Insufficient public-train classes for {condition}')
            subset = ProxyData(data.x.loc[use], data.y[use], data.window[use],
                               data.condition[use], data.date[use])
            expert = RefinedQueueModel(iterations=self.iterations, num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples, seed=self.seed).fit(subset)
            self.models[condition] = expert
            self.training_counts[condition] = {'cells': int(use.sum()),
                'events': int(len(np.unique(data.window[use]))),
                'positive_cells': int(data.y[use].sum())}
        return self

    def probability(self, x, condition=None):
        if condition is None:
            if '_published_queue_condition' not in x:
                raise ValueError('Published condition is required for expert routing')
            condition = x['_published_queue_condition'].to_numpy()
        condition = np.asarray(condition)
        if condition.shape != (len(x),) or not np.isin(condition, list(self.models)).all():
            raise ValueError('Unknown or missing published Queue condition')
        p = np.zeros(len(x))
        for c, expert in self.models.items():
            use = condition == c
            if c == 'queue_onset':
                use &= x.lead_minutes.to_numpy() == 30
            if use.any():
                p[use] = expert.probability(x.loc[use])
        return p

    def classify(self, x, window, condition):
        return super().classify(x.assign(_published_queue_condition=condition), window, condition)

    def tune(self, data):
        routed = ProxyData(data.x.assign(_published_queue_condition=data.condition),
                           data.y, data.window, data.condition, data.date)
        return super().tune(routed)


class ExpectedIoUQueueModel(QueueModel):
    """A fixed decision rule around an already fitted probability model."""
    def __init__(self, base):
        super().__init__(base.method, base.seed, base.iterations, base.threads)
        self.base = base
        for name in ('uses_origin_timestamp', 'uses_occupancy_features',
                     'uses_published_condition', 'latest_valid_queue_bookkeeping'):
            setattr(self, name, getattr(base, name, False))
        self.uses_public_onset_timing = True

    def probability(self, x, condition=None):
        if isinstance(self.base, OnsetSpecialistQueueModel):
            return self.base.probability(x, condition)
        return self.base.probability(x)

    def classify(self, x, window, condition):
        return event_prefix_decisions(self.probability(x, condition), window, condition, x.lead_minutes)

    def tune(self, data):
        p = self.classify(data.x, data.window, data.condition)
        return {'fixed_rule': 'Poisson-binomial expected-IoU ranked prefix',
                'calibration_proxy_IoU': grouped_iou(data.y, p, data.window, data.condition)['proxy_IoU'],
                'settings_fitted_on_assessment': False, 'formal_S_queue': None}
