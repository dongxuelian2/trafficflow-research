"""PSD, graph provenance, and finite-rank posterior identities."""
import numpy as np
import pandas as pd
import pytest
from trafficflow.v2.directional_kernel import (trusted_chains, propagation_kernel,
                                               legal_gate, SparseResidualGP,
                                               counterfactual_sampled_anchors, TrustedGeometry)


def small_topology():
    return pd.DataFrame({'link_id': ['a', 'latent', 'b', 'c', 'd'],
        'incoming_link_ids': ['', 'a', 'latent', '', 'c'],
        'outgoing_link_ids': ['latent', 'b', '', 'd', ''],
        'from_node': ['0', '1', '2', '3', '4'],
        'to_node': ['1', '2', '3', '4', '5']}).set_index('link_id')


@pytest.mark.parametrize('mode', ['ordinary', 'gated_ordinary', 'directional'])
def test_psd_arbitrary_gates_components_and_lawful_features(mode):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(60, 8))
    x[:, 2] = rng.uniform(size=60)
    x[:, 3] = rng.choice([-1, 0, 1, 2], size=60)
    gram = propagation_kernel(x, x, mode)
    np.testing.assert_allclose(gram, gram.T, atol=1e-14)
    assert np.linalg.eigvalsh(gram).min() > -1e-10
    assert np.all(gram[x[:, 3] < 0] == 0)
    assert np.all(gram[(x[:, 3, None] != x[None, :, 3])] == 0)


def test_propagation_peak_has_physical_direction_and_time_units():
    origin = np.array([[0., 0., 0., 0.]])
    downstream = np.array([[80./12, 1./12, 0., 0.]])
    stationary = np.array([[0., 1./12, 0., 0.]])
    assert propagation_kernel(origin, downstream)[0, 0] > propagation_kernel(origin, stationary)[0, 0]
    origin[:, 2] = 1.
    upstream = np.array([[-15./12, 1./12, 1., 0.]])
    wrong_way = np.array([[15./12, 1./12, 1., 0.]])
    assert propagation_kernel(origin, upstream)[0, 0] > propagation_kernel(origin, wrong_way)[0, 0]


def test_topology_crosses_latent_links_but_never_fills_mileage_gaps():
    topo = small_topology()
    lengths = pd.Series(1., index=topo.index)
    mile = pd.Series({'a': 0., 'b': 2., 'c': 3., 'd': 4., 'unconnected': 2.5})
    links = ['a', 'b', 'c', 'd', 'unconnected']
    geometry = trusted_chains(topo, lengths, mile, links)
    assert geometry.component[0] == geometry.component[1]
    assert geometry.component[1] != geometry.component[2]
    assert geometry.component[-1] == -1
    np.testing.assert_allclose(geometry.distance_km[:2], [0, 2])
    assert geometry.edges.tolist() == [[0, 1], [2, 3]]
    assert geometry.audit['mileage_used_to_create_edges'] is False


def test_bad_mileage_and_reciprocity_disable_untrusted_direction():
    topo = small_topology()
    lengths = pd.Series(1., index=topo.index)
    mile = pd.Series({'a': 0., 'latent': 1., 'b': .5, 'c': 0., 'd': 1.})
    geometry = trusted_chains(topo, lengths, mile, list(topo.index))
    assert np.all(geometry.component[:3] == -1)
    topo.loc['d', 'incoming_link_ids'] = ''
    geometry = trusted_chains(topo, lengths, mile, ['c', 'd'])
    assert np.all(geometry.component == -1)
    assert any(r['reason'] == 'not_reciprocal' for r in geometry.audit['rejected'])


def test_finite_rank_gp_matches_dense_posterior_and_prior_fallback():
    rng = np.random.default_rng(4)
    x = np.column_stack([rng.normal(size=20), rng.uniform(0, 1, 20), rng.uniform(0, 1, 20), np.zeros(20)])
    y = np.column_stack([np.sin(x[:, 0]), x[:, 0]/5])
    model = SparseResidualGP('directional', inducing=7, noise=.2, batch=6).fit(x, y)
    prediction, variance = model.predict(x, variance=True)
    phi = model._features(x)
    approximate = phi@phi.T
    expected = approximate@np.linalg.solve(approximate+.2*np.eye(len(x)), y)
    np.testing.assert_allclose(prediction, expected, rtol=2e-9, atol=2e-9)
    assert np.all(variance >= 0)
    assert np.all(variance <= np.square(phi).sum(axis=1)+1e-9)
    bad = x[:2].copy(); bad[:, 3] = -1
    np.testing.assert_allclose(model.predict(bad), 0.)


def test_gate_and_parameter_validation():
    gate = legal_gate([20, 60, 100], [100, 100, 100])
    assert gate[0] > gate[1] > gate[2]
    np.testing.assert_allclose(gate[1], .5)
    with pytest.raises(ValueError):
        propagation_kernel(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        SparseResidualGP(inducing=0)


def test_counterfactual_sample_masks_four_channels_and_never_exposes_targets(monkeypatch):
    import trafficflow.state as state
    channels = ['speed_kmh', 'flow_vph', 'occupancy', 'density_occ_linear_vehpkm']
    times = np.array(['2030-10-30T00:00:00Z', '2030-10-30T00:05:00Z'])
    links = np.array(['a', 'b', 'c'])
    frame = pd.MultiIndex.from_product([times, links], names=['timestamp', 'link_id']).to_frame(index=False)
    frame['station_id'] = frame.link_id
    frame['corridor_id'] = 'panel'
    frame['mask_regime'] = 'R1'
    for c in channels:
        frame[c] = 20.
    frame.loc[0, channels] = np.nan
    class FakeFeatures:
        def __init__(self, frame, *args, **kwargs):
            self.frame, self.links, self.times = frame, list(links), list(times)
            self.features = pd.DataFrame(np.zeros((6, 2)), columns=['a', 'b'])
    monkeypatch.setattr(state, 'StateFeatures', FakeFeatures)
    class FakeModel:
        profile, feature_version = None, 2
        def predict(self, features, target):
            assert features.frame.loc[0, channels].isna().all()
            selected = features.frame.merge(target[['timestamp', 'link_id']], on=['timestamp', 'link_id'])
            assert selected[channels].isna().all().all()
            return pd.DataFrame({'speed_kmh': np.full(len(target), 10.), 'flow_vph': np.full(len(target), 10.)})
    observed = np.full((2, 3, 2), 20.); observed[0, 0] = np.nan
    cache = {'times': times, 'links': links, 'legal_observed': observed, 'target_indices': np.array([0])}
    geometry = TrustedGeometry(tuple(links), np.zeros(3, int), np.arange(3.), np.array([[0, 1], [1, 2]]), {})
    result = counterfactual_sampled_anchors(FakeModel(), frame, pd.DataFrame({'lanes': [1., 1., 1.]}, index=links),
        pd.DataFrame(), cache, geometry, maximum=4, folds=2)
    assert len(result['flat_indices']) == 4
    assert 0 not in result['flat_indices']
    assert sum(r['n_withheld'] for r in result['certificates']) == 4
    assert np.isfinite(result['residual']).all()
    assert frame.loc[1:, channels].notna().all().all()


@pytest.mark.parametrize('same_regime, expected_pairs', [(True, 6), (False, 4)])
def test_fixed_pair_vector_score_and_microsecond_midnight_contract(monkeypatch, same_regime, expected_pairs):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/v2'))
    runner = importlib.import_module('run_directional_experiments')
    from trafficflow.v2.score_aligned import state_statistics
    from trafficflow.v2.anchored_increment import increment_proxy
    times = pd.date_range('2030-11-13 23:50:00', periods=4, freq='5min', tz='UTC').as_unit('us')
    all_rows = []
    for time_index, time in enumerate(times):
        for link in ['a', 'b']:
            all_rows.append({'timestamp': time, 'link_id': link, 'station_id': link,
                'speed_kmh': 20., 'flow_vph': 200.+20*time_index, 'length_km': 1.,
                'accumulation_N': 10.+time_index, 'on_ramp_valid': 1, 'off_ramp_valid': 1})
    truth = pd.DataFrame(all_rows)
    truth['timestamp'] = pd.DatetimeIndex(truth.timestamp).as_unit('us')
    topology = pd.DataFrame({'link_id': ['a', 'b'], 'incoming_link_ids': ['', 'a'],
                             'outgoing_link_ids': ['b', '']}).set_index('link_id')
    records, all_predicted = [], []
    for day, rows in truth.groupby(truth.timestamp.dt.date):
        rows = rows.reset_index(drop=True)
        target = rows.iloc[[2 if str(day) == '2030-11-13' else 0]].copy()
        predicted = target[['speed_kmh', 'flow_vph']].to_numpy(float, copy=True)
        predicted[:, 0] = 10.
        scorer = runner.FixedPairScorer(rows, target, target, [1.], [True], rows, topology)
        np.testing.assert_allclose(scorer.statistics(predicted)['speed_sse'],
            state_statistics(target, target.assign(speed_kmh=10.), [1.])['speed_sse'])
        metric = scorer.increment(predicted)
        row = {'panel': 'p', 'date': str(day), 'regime': 'R1' if same_regime or str(day) == '2030-11-13' else 'R3',
               **metric, '_boundary_endpoints': scorer.boundaries(predicted),
               '_eligible_links': scorer.eligible_links, 'increment_midnight_pairs': 0}
        records.append(row)
        output = rows.copy()
        output.loc[scorer.target_idx, ['speed_kmh', 'flow_vph']] = predicted
        output['accumulation_N'] = output.length_km*output.flow_vph/output.speed_kmh.clip(lower=1)
        all_predicted.append(output)
    runner.add_midnight_pairs(records, topology)
    assert sum(r['n_pairs'] for r in records) == expected_pairs
    assert sum(r['increment_midnight_pairs'] for r in records) == (2 if same_regime else 0)
    if same_regime:
        reference = increment_proxy(pd.concat(all_predicted, ignore_index=True), truth, topology)
        assert sum(r['increment_error_L1'] for r in records) == reference['increment_error_L1']
        assert sum(r['increment_truth_L1'] for r in records) == reference['increment_truth_L1']
