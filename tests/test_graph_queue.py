import numpy as np
import pandas as pd
import pytest
torch = pytest.importorskip('torch')
from trafficflow.graph_queue import queue_history_input, CausalGraphQueue, queue_event_loss


def test_queue_sequence_rejects_future_and_ignores_ineligible_numeric_values():
    origin = pd.Timestamp('2030-07-20T12:00:00Z')
    history = pd.DataFrame({'timestamp': ['2030-07-20T11:00:00Z', '2030-07-20T12:00:00Z'],
        'link_id': ['a', 'a'], 'speed_kmh': [30., 90.], 'flow_vph': [900., 2700.],
        'occupancy': [.2, .1], 'pct_observed': [0., 100.], 'is_score_eligible': [False, True]})
    net = pd.DataFrame({'lanes': [3., 2.]}, index=['a', 'b'])
    x = queue_history_input(history, ['a', 'b'], net, origin, 'queue_onset')
    assert x.shape == (13, 2, 9)
    assert x[-1, 0, 0] == pytest.approx(90/25)
    assert x[-1, 0, 1] == pytest.approx(2700/3/600)
    assert not x[:, 1, 2:4].any()
    contaminated = history.copy()
    contaminated.loc[0, ['speed_kmh', 'flow_vph', 'occupancy']] = 9999.
    np.testing.assert_array_equal(x, queue_history_input(contaminated, ['a', 'b'], net, origin, 'queue_onset'))
    leaked = history.copy()
    leaked.loc[1, 'timestamp'] = '2030-07-20T12:05:00Z'
    with pytest.raises(ValueError, match='future input leakage'):
        queue_history_input(leaked, ['a', 'b'], net, origin, 'queue_onset')


def test_queue_forecast_variable_nodes_and_domain_loss():
    torch.set_num_threads(2)
    model = CausalGraphQueue(hidden=16)
    for nodes in (4, 7):
        x = torch.zeros(2, 9, 13, nodes)
        geometry = torch.tensor([[100.], [2000.], [120.], [50.]]).repeat(1, nodes)
        graph = torch.eye(nodes)
        logits, speed = model(x, graph, graph, geometry)
        assert logits.shape == speed.shape == (2, 6, nodes)
        # No observations: zero residual starts from public free speed.
        torch.testing.assert_close(speed, torch.full_like(speed, 4.))
        labels = torch.zeros_like(logits)
        available = torch.ones_like(labels, dtype=torch.bool)
        loss = queue_event_loss(logits, speed, labels, speed.detach(), available, torch.tensor([False, True]))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
    logits = torch.zeros(1, 6, 2, requires_grad=True)
    speed = torch.zeros_like(logits)
    queue_event_loss(logits, speed, torch.ones_like(logits), speed, torch.ones_like(logits, dtype=torch.bool), torch.tensor([False])).backward()
    assert not logits.grad[:, :5].any()
    assert logits.grad[:, 5:].abs().sum() > 0
