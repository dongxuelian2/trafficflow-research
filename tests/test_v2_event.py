"""Causal information barriers and independently checked Queue set decisions."""
import importlib.util
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from trafficflow.v2.event_posterior import (ConditionalEventPosterior, expected_iou_decode,
    history_descriptor, independent_trajectories, signed_graph_distance, stable_origin_seed)


def fixture_data():
    rng = np.random.default_rng(104)
    x = np.zeros((12, 13, 3, 9))
    x[..., 0] = 2.+rng.normal(0, .03, x.shape[:3])
    x[..., 1] = .8
    x[..., [2, 3, 5]] = 1
    x[..., 4] = .3
    x[6:, ..., 8] = 1
    speed = np.full((12, 6, 3), 3.)
    speed[:6, -1, 0] = 1.
    speed[6:, :, 2] = 1.
    available = np.ones(speed.shape, bool)
    distance = np.array([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]])
    return x, speed, available, np.array([50., 50., 50.]), distance


def fitted():
    x, speed, available, cut, distance = fixture_data()
    return ConditionalEventPosterior.fit(x, speed, available, cut, distance, neighbors=4), x


def test_condition_separation_and_joint_onset_contract():
    model, x = fitted()
    weights, p = model.component_weights(x), model.probability(x)
    assert weights[:6, 6:].sum() == 0
    assert weights[6:, :6].sum() == 0
    assert np.allclose(weights.sum(axis=1), 1)
    assert np.all(p[:6, :5] == 0)
    assert np.all(p[:6, -1, 0] > .999)
    assert np.all(p[6:, :, 2] > .999)
    model.templates[:] = 1.
    assert np.all((model.probability(x) >= 0.) & (model.probability(x) <= 1.))
    model, x = fitted()
    z = model.sample(x[:2], 400, 60)
    assert z.shape == (2, 400, 6, 3)
    assert not z[:, :, :5].any()
    summary = model.event_summary(x)
    assert np.allclose(summary['bottleneck_probability'].sum(axis=1), 1)
    direction = summary['direction_probability_upstream_stationary_downstream_unknown']
    assert np.allclose(direction.sum(axis=1), 1)
    assert np.allclose(direction[:6, 3], 1)  # One onset time cannot identify direction.


def test_missing_fit_targets_cannot_poison_templates():
    x, speed, available, cut, distance = fixture_data()
    available[:, :, 1] = False
    first = ConditionalEventPosterior.fit(x, speed, available, cut, distance)
    speed[:, :, 1] = np.nan
    second = ConditionalEventPosterior.fit(x, speed, available, cut, distance)
    assert np.array_equal(first.templates, second.templates)


def test_threshold_observation_noise_explicit_at_cutoff():
    x, speed, available, cut, distance = fixture_data()
    speed[:, -1, 1] = cut[1]/25.
    model = ConditionalEventPosterior.fit(x, speed, available, cut, distance)
    assert 1.5 <= model.sigma_speed_kmh <= 8.
    assert np.all(model.templates[:, -1, 1] == .5)


def test_future_masks_and_future_slots_rejected_by_prediction_apis():
    model, x = fitted()
    with pytest.raises(TypeError):
        model.probability(x, available=np.ones((len(x), 6, 3)))
    with pytest.raises(TypeError):
        model.sample(x, future_quality=np.zeros((len(x), 6, 3)))
    with pytest.raises(ValueError, match='13'):
        model.probability(np.concatenate([x, x[:, :1]], axis=1))
    mixed = x.copy()
    mixed[0, 0, 0, 8] = 1
    with pytest.raises(ValueError, match='constant'):
        history_descriptor(mixed)
    p = np.zeros((6, 3))
    with pytest.raises(TypeError):
        expected_iou_decode(p, independent_trajectories(p), False, future_eligibility=np.zeros_like(p))


def test_graph_directions_do_not_depend_on_link_id_or_row_order():
    links = ['Z', 'A', 'M', 'D']
    net = pd.DataFrame({'length_km': [1., 1., 1., 1.]}, index=links)
    topo = pd.DataFrame({'outgoing_link_ids': ['A', 'M', '', '']}, index=links)
    distance = signed_graph_distance(links, net, topo)
    assert distance[0, 2] == 2
    assert distance[2, 0] == -2
    assert np.isnan(distance[0, 3])
    order = [2, 0, 3, 1]
    permuted = signed_graph_distance([links[i] for i in order], net, topo)
    assert np.allclose(permuted, distance[np.ix_(order, order)], equal_nan=True)


def scalar_iou(left, right):
    a, b = set(np.flatnonzero(left)), set(np.flatnonzero(right))
    return len(a & b)/len(a | b) if a | b else 1.


def test_expected_iou_matches_scalar_exhaustive_small_joint_laws():
    rng = np.random.default_rng(206)
    truths = np.array(list(itertools.product([False, True], repeat=2)))
    for _ in range(20):
        counts = rng.integers(1, 20, size=4)
        scenarios = np.zeros((int(counts.sum()), 6, 2), bool)
        scenarios[:, -1] = np.repeat(truths, counts, axis=0)
        probability = scenarios.mean(axis=0)
        action, detail = expected_iou_decode(probability, scenarios, False)
        values = [np.mean([scalar_iou(truth, value) for value in scenarios[:, -1]]) for truth in truths]
        actual = np.mean([scalar_iou(action[-1], value) for value in scenarios[:, -1]])
        assert actual == pytest.approx(max(values), abs=1e-12)
        assert detail['expected_IoU'] == pytest.approx(actual, abs=1e-12)
        assert not action[:5].any()
        assert detail['candidate_count'] <= 29
    zero = np.zeros((6, 3))
    action, detail = expected_iou_decode(zero, np.zeros((10, 6, 3), bool), True)
    assert not action.any()
    assert detail['expected_IoU'] == 1.


def test_joint_scenarios_preserve_marginals_and_origin_seed_repeatability():
    model, x = fitted()
    # Place all components exactly on cutoff at one cell: observation noise's
    # marginal remains .5 although its joint draws correlate over horizons.
    model.templates[:, :, 1] = .5
    draws = model.sample(x[6:7], 20000, 809)
    assert np.max(np.abs(draws.mean(axis=1)[0]-model.probability(x[6:7])[0])) < .015
    a = stable_origin_seed('2030-01-01T10:00:00+00:00', 30)
    b = stable_origin_seed('2030-01-01T10:05:00+00:00', 30)
    assert a != b
    assert np.array_equal(model.sample(x[:1], 32, a), model.sample(x[:1], 32, a))


def load_runner():
    path = Path(__file__).resolve().parents[1]/'scripts/v2/run_queue_experiments.py'
    spec = importlib.util.spec_from_file_location('v2_queue_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_way_predictions_ignore_any_evaluator_poison():
    runner = load_runner()
    model, x = fitted()
    x = x[[0, 6]]
    old = np.full((2, 6, 3), .4)
    condition = np.repeat(['queue_onset', 'queue_ongoing'], 18)
    metadata = {'window': np.repeat(['w0', 'w1'], 18), 'condition': condition,
                'lead_minutes': np.tile(np.repeat(np.arange(5, 31, 5), 3), 2)}
    setting = {'thresholds': {'global': .5}, 'require_nonempty': False, 'timing': 'before'}
    args = (model, old, x, ['o0', 'o1'], np.array([False, True]), setting, setting)
    first, _, _ = runner.predict_four(*args, metadata, 505)
    poisoned = {**metadata, 'labels': np.ones((2, 6, 3))*900,
                'available': np.zeros((2, 6, 3)), 'future_quality': np.full((2, 6, 3), np.nan)}
    second, _, _ = runner.predict_four(*args, poisoned, 505)
    assert set(first) == set(runner.ARMS)
    for name in first:
        assert np.array_equal(first[name], second[name])
        assert not first[name][0, :5].any()


def test_bootstrap_origin_dedup_and_same_calendar_case_synchrony():
    runner = load_runner()
    dates = [f'2030-01-{i:02}' for i in range(1, 16)]
    rows = [{'window': 'panel:'+date+'T10:00:00+00:00:2909',
             'condition': condition, 'IoU': .4}
            for date, condition in zip(dates, ['queue_onset', 'queue_ongoing']*7+['queue_onset'])]
    better = [{**row, 'IoU': .5} for row in rows]
    run = {'models': {runner.ARMS[0]: {'windows': rows}, runner.ARMS[3]: {'windows': better}}}
    report = {'case': 'small', 'dates': dates, 'runs': [run, run]}
    comparison = runner.paired_comparison([report], resamples=200)
    assert comparison['paired_macro_gain'] == pytest.approx(.1)
    assert comparison['ci95'] == pytest.approx([.1, .1])
    assert comparison['cases'][0]['unique_origins'] == 15
    assert comparison['cases'][0]['sampled_windows'] == 30
    assert not comparison['research_gate_pass']  # Less than 30 unique onset.
