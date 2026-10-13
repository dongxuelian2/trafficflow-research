import numpy as np
import pandas as pd
import pytest
from trafficflow.digital_twin import Scenario, simulate, simulated_physics
from trafficflow.research_eval import Fold, paired_block_interval, register_protocol
from trafficflow.physics import conservation_operator


def test_ctm_conservation_and_anchoring():
    sim = simulate(Scenario(seed=42, days=1, cells=16, incident_cell=10))
    assert sim['max_conservation_error_vehicles'] < 1e-9
    assert sim['density'].min() >= 0
    assert sim['density'].max() <= 400
    # Noiseless observation-perfect candidate satisfies our projected flux.
    exact = simulated_physics(sim, sim['observation_speed'], sim['observation_flow'],
                              'observation_projected_synthetic_flux')
    assert exact['simulated_S_LWR'] > 1 - 1e-10
    wrong = simulated_physics(sim, sim['observation_speed'] * .8, sim['observation_flow'],
                              'observation_projected_synthetic_flux')
    assert wrong['simulated_S_LWR'] < exact['simulated_S_LWR']
    assert exact['formal_Physics'] is None


def test_fold_and_protocol_write_once(tmp_path):
    dates = ['2030-06-01', '2030-07-01', '2030-08-01', '2030-09-01']
    fold = Fold('f', *dates)
    assert fold.assign(dates)['assessment'] == ['2030-09-01']
    path = tmp_path / 'protocol.json'
    register_protocol(path, {'seed': 1})
    register_protocol(path, {'seed': 1})
    with pytest.raises(ValueError, match='differs'):
        register_protocol(path, {'seed': 2})


def test_paired_blocks_preserve_pairing():
    a = np.arange(21) / 100
    out = paired_block_interval(a, a + .02)
    np.testing.assert_allclose(out['ci95'], [.02, .02])


def test_external_connectors_are_valid_internal_missing_is_skipped():
    times = ['2030-06-01T00:00:00Z', '2030-06-01T00:05:00Z']
    net = pd.DataFrame({'length_km': [1.]}, index=['sensor'])
    topo = pd.DataFrame({'incoming_link_ids': ['external_in'],
                         'outgoing_link_ids': ['external_out']}, index=['sensor'])
    v = np.array([[100.], [100.]])
    q = np.array([[1000.], [1100.]])
    B, rhs = conservation_operator(times, ['sensor'], v, q, net, topo, np.zeros_like(q), np.zeros_like(q))
    assert B.shape[0] == 1
    np.testing.assert_allclose(B @ q.ravel() - rhs, [1.])
    topo.loc['external_in'] = ['', 'sensor']
    B, rhs = conservation_operator(times, ['sensor'], v, q, net, topo, np.zeros_like(q), np.zeros_like(q))
    assert B.shape[0] == 0
