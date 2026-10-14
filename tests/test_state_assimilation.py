import numpy as np
import pytest
from trafficflow.state_assimilation import (AssimilationParameters,
    observation_residuals, precision_matrix, assimilate_residuals, normalized_prior_grid,
    counterfactual_anchor_prior)


def test_sparse_assimilation_matches_independent_dense_normal_equations():
    # Independently build temporal and spatial difference rows; this checks the
    # quadratic and flattening convention, rather than only another CG call.
    prior = np.array([[[2., .7], [1., .3]], [[1.8, .6], [1.1, .4]], [[1.6, .5], [1.2, .5]]])
    strict = np.full_like(prior, np.nan)
    strict[0, 0], strict[2, 1] = [2.2, .8], [.9, .6]
    quality = np.zeros((3, 2, 6))
    quality[1, 0] = [1.7, .55, 1., 1., .5, 1.]
    params = AssimilationParameters(.25, 2., .5, .3)
    delta, certificates = assimilate_residuals(prior, strict, quality, [[0, 1]], params)
    innovations, weights = observation_residuals(prior, strict, quality, params)
    differences = []
    penalties = []
    for first, second, penalty in [(0, 2, 2.), (1, 3, 2.), (2, 4, 2.), (3, 5, 2.),
                                    (0, 1, .5), (2, 3, .5), (4, 5, .5)]:
        row = np.zeros(6); row[first], row[second] = 1., -1.
        differences.append(row); penalties.append(penalty)
    d = np.array(differences)
    for channel in range(2):
        w = weights[..., channel].ravel()
        dense = np.diag(.25+w)+d.T@np.diag(penalties)@d
        expected = np.linalg.solve(dense, w*innovations[..., channel].ravel())
        np.testing.assert_allclose(delta[..., channel].ravel(), expected, atol=1e-9)
        assert certificates[channel]['relative_linear_residual'] < 2e-8


def test_unavailable_values_and_quality_unknown_are_never_observation_anchors():
    prior = np.ones((3, 2, 2))
    strict = np.full_like(prior, np.nan)
    quality = np.full((3, 2, 6), 999.)
    quality[..., 2:4], quality[..., 5] = 0., 0.
    params = AssimilationParameters(low_quality_weight=1.)
    delta, certificates = assimilate_residuals(prior, strict, quality, [[0, 1]], params)
    np.testing.assert_array_equal(delta, 0.)
    assert all(r['iterations'] == 0 for r in certificates)
    strict[0, 0] = [2., 1.]
    quality[0, 0, :2] = [-999., 999.]
    innovations, weights = observation_residuals(prior, strict, quality, params)
    np.testing.assert_array_equal(innovations[0, 0], [1., 0.])
    np.testing.assert_array_equal(weights[0, 0], 1.)


def test_robust_innovation_and_transition_gates_preserve_spd():
    prior = np.ones((3, 2, 2)); prior[1:] += 5.
    strict = prior+10.
    quality = np.zeros((3, 2, 6))
    params = AssimilationParameters(.5, 4., 1., .25, (.4, .2), (.4, .2))
    _, weights = observation_residuals(prior, strict, quality, params)
    np.testing.assert_allclose(weights[0, 0], [.04, .02])
    for channel in range(2):
        matrix = precision_matrix((3, 2), weights, prior, np.array([[0, 1]]), params, channel).toarray()
        np.testing.assert_allclose(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() >= .5
    with pytest.raises(ValueError, match='positive'):
        AssimilationParameters(prior_precision=0).validate()


def test_prior_grid_handles_pandas_time_resolution_and_rejects_real_gaps():
    from types import SimpleNamespace
    import pandas as pd
    times = ['2030-10-16T00:00:00Z', '2030-10-16T00:05:00Z']
    links = ['a', 'b']
    index = pd.MultiIndex.from_product([times, links], names=['timestamp', 'link_id'])
    frame = index.to_frame(index=False).assign(station_id=['s1', 's2']*2, mask_regime='R1')
    features = SimpleNamespace(times=times, links=links, frame=frame,
        net=pd.DataFrame({'lanes': [2., 2.]}, index=links),
        lookup=lambda t: index.get_indexer(pd.MultiIndex.from_frame(t[['timestamp', 'link_id']])))
    prior = SimpleNamespace(predict=lambda f, t: t.assign(speed_kmh=25., flow_vph=1200.))
    target = frame.assign(panel='public')
    result = normalized_prior_grid(prior, features, target)
    np.testing.assert_array_equal(result, np.ones((2, 2, 2)))
    features.times = ['2030-10-16T00:00:00Z', '2030-10-16T00:10:00Z']
    with pytest.raises(ValueError, match='five-minute'):
        normalized_prior_grid(prior, features, target)


def test_counterfactual_anchor_predictor_cannot_read_own_four_measurements():
    from types import SimpleNamespace
    import pandas as pd
    from trafficflow.state import StateFeatures
    times, links = ['2030-10-16T00:00:00Z', '2030-10-16T00:05:00Z'], ['a', 'b']
    index = pd.MultiIndex.from_product([times, links], names=['timestamp', 'link_id'])
    frame = index.to_frame(index=False).assign(station_id=['s1', 's2']*2, mask_regime='R1',
        speed_kmh=[10., 20., 15., 30.], flow_vph=[400., 500., 600., 800.],
        occupancy=.2, density_occ_linear_vehpkm=50., pct_observed=100., milepost=[1., 2.]*2)
    original = frame.copy(deep=True)
    net = pd.DataFrame({'lanes': [2., 2.], 'free_speed_kmh': [100., 100.],
                        'capacity_vph': [4000., 4000.], 'length_km': [1., 1.]}, index=links)
    topology = pd.DataFrame({'incoming_link_ids': ['', 'a'], 'outgoing_link_ids': ['b', '']}, index=links)
    profile = SimpleNamespace(predict=lambda rows, n: np.tile([40., 1200.], (len(rows), 1)))
    withheld_counts = []
    def predict(features, target):
        selected = features.frame[['speed_kmh', 'flow_vph']].isna().all(axis=1)
        if selected.any():
            assert features.frame.loc[selected, ['occupancy', 'density_occ_linear_vehpkm']].isna().all().all()
            withheld_counts.append(int(selected.sum()))
        return target.assign(speed_kmh=features.frame.speed_kmh.fillna(25.).to_numpy(),
                             flow_vph=features.frame.flow_vph.fillna(1200.).to_numpy())
    model = SimpleNamespace(predict=predict, profile=profile, topology=topology, feature_version=1)
    features = StateFeatures(frame, net, topology, profile, version=1)
    target = frame.assign(panel='public')
    prior = normalized_prior_grid(model, features, target)
    anchors, certificates = counterfactual_anchor_prior(model, features, target, prior, folds=3)
    np.testing.assert_array_equal(anchors, np.ones((2, 2, 2)))
    assert sum(withheld_counts) == 4
    assert sum(c['n_visible_cells_withheld'] for c in certificates) == 4
    pd.testing.assert_frame_equal(frame, original)
