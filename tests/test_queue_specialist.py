"""Independent enumeration checks for probabilistic set decisions."""
from itertools import product
import numpy as np
import pandas as pd
import pytest
from trafficflow.metrics import queue_iou
from trafficflow.queue_specialist import expected_iou_prefix, event_prefix_decisions, OnsetSpecialistQueueModel


@pytest.mark.parametrize('p', [[.9, .65, .2, .05], [.01, .02, .03, .04], [1., 0., .4, .4]])
def test_expected_iou_matches_exhaustive_labels_and_all_prediction_sets(p):
    p = np.asarray(p)
    outcomes = list(product([0, 1], repeat=len(p)))
    def expected(pred):
        return sum(np.prod(np.where(y, p, 1-p)) * queue_iou(np.asarray(y), pred) for y in outcomes)
    for nonempty in (False, True):
        pred, details = expected_iou_prefix(p, nonempty)
        best = max(expected(np.asarray(y)) for y in outcomes if not nonempty or any(y))
        assert expected(pred) == pytest.approx(best, abs=1e-12)
        assert details['expected_IoU'] == pytest.approx(expected(pred), abs=1e-12)


def test_prefix_uses_published_onset_timing_and_preserves_ongoing_domain():
    p = np.array([.99, .01, .8, .2, .99, .01])
    window = np.array(['a', 'a', 'a', 'a', 'b', 'b'])
    condition = np.array(['queue_onset']*4 + ['queue_ongoing']*2)
    lead = np.array([5, 5, 30, 30, 5, 30])
    pred = event_prefix_decisions(p, window, condition, lead)
    assert pred.tolist() == [0, 0, 1, 0, 1, 0]
    # An onset decision must remain nonempty within the allowed final step.
    assert event_prefix_decisions(np.array([.99, 0.]), np.array(['a', 'a']),
        np.array(['queue_onset', 'queue_onset']), np.array([5, 30])).tolist() == [0, 1]


def test_expert_routing_requires_published_condition():
    class Constant:
        def __init__(self, value): self.value = value
        def probability(self, x): return np.full(len(x), self.value)
    model = OnsetSpecialistQueueModel()
    model.models = {'queue_onset': Constant(.8), 'queue_ongoing': Constant(.3)}
    x = pd.DataFrame({'lead_minutes': [5, 30, 5]})
    with pytest.raises(ValueError, match='Published condition'):
        model.probability(x)
    assert model.probability(x, ['queue_onset', 'queue_onset', 'queue_ongoing']).tolist() == [0., .8, .3]
