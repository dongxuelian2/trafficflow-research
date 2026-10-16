"""Numerical contracts that protect evaluator scope and mathematical claims."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from trafficflow.v2.score_aligned import anchored_frame, conservation_pairs, physics_score, state_macro
from trafficflow.v2.score_decision import LowRankCorrection, paired_macro_bootstrap
from trafficflow.v2.anchored_increment import increment_proxy


def test_station_means_precede_ratio_and_unpublished_blanks_are_anchors():
    t = pd.Timestamp('2030-06-01', tz='UTC')
    raw = pd.DataFrame({'timestamp': [t]*3, 'station_id': ['1', '2', '3'], 'link_id': ['a', 'a', 'b'],
                        'speed_kmh': [10., 100., 30.], 'flow_vph': [100., 200., 300.], 'is_score_eligible': [True]*3})
    template = raw.iloc[[0]][['timestamp', 'station_id', 'link_id']]
    prediction = template.assign(speed_kmh=20., flow_vph=400.)
    anchor, missing = anchored_frame(raw, template, prediction)
    assert missing == 0
    a = anchor.set_index('link_id').loc['a']
    assert a.speed_kmh == 60 and a.flow_vph == 300 and a.is_target == .5
    assert a.flow_vph/a.speed_kmh == 5
    assert anchor.set_index('link_id').loc['b', 'speed_kmh'] == 30


def frame_fixture():
    rows = []
    for link in ['a', 'b', 'c']:
        for t, minute in enumerate([0, 5, 15]):
            rows.append(dict(panel='P', mask_regime='R1', timestamp=pd.Timestamp('2030-06-01', tz='UTC')+pd.Timedelta(minutes=minute),
                link_id=link, speed_kmh=50., flow_vph=1000., density_vpkm=20., inflow_vph=1000., outflow_vph=1000.,
                on_ramp_flow_vph=0., off_ramp_flow_vph=0., on_ramp_valid=1, off_ramp_valid=1,
                accumulation_N=10.+t, length_km=.5, lanes=2., free_speed_kmh=100., capacity_vph=4000., critical_density=40., k_jam=200., is_target=1.))
    f = pd.DataFrame(rows)
    topo = pd.DataFrame({'link_id': ['a', 'b', 'c', 'latent'], 'incoming_link_ids': ['external', 'latent', 'a', 'b'],
                        'outgoing_link_ids': ['c', 'external', 'external', 'b'], 'on_ramp_link_ids': ['', '', '', ''],
                        'off_ramp_link_ids': ['', '', '', '']}).set_index('link_id')
    return f, topo


def test_pairs_internal_missing_external_boundary_and_gap():
    frame, topo = frame_fixture()
    pairs = conservation_pairs(frame, topo)
    assert pairs.link_id.tolist() == ['a', 'c']
    assert (pairs.dt == 5/60).all()
    frame.loc[(frame.link_id == 'c') & (frame.timestamp.dt.minute == 0), 'on_ramp_valid'] = 0
    assert conservation_pairs(frame, topo).link_id.tolist() == ['a']


def test_independent_physics_matches_official_on_constructed_edges(tmp_path):
    f, topo = frame_fixture()
    panel = tmp_path/'corridors/P/network'; panel.mkdir(parents=True)
    params = f[['link_id', 'length_km', 'free_speed_kmh', 'capacity_vph', 'lanes', 'k_jam']].drop_duplicates('link_id')
    params.to_csv(panel/'links.csv', index=False)
    topo.reset_index().to_csv(panel/'lwr_mainline_topology.csv', index=False)
    path = Path('vendor/official/src/task3/score_task3.py')
    spec = importlib.util.spec_from_file_location('test_official_v2', path)
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    for mode in ['A', 'B', 'C']:
        for empty_share in [False, True]:
            current = f.copy()
            if empty_share:
                current['flow_vph'] = 0.
            own = physics_score(current, topo, mode)
            upstream = official.score_panel('P', current.drop(columns=['length_km', 'free_speed_kmh', 'capacity_vph', 'lanes', 'critical_density', 'k_jam']), tmp_path, {'panels': {'P': {'mode': mode}}}, 'R1')
            for key, value in own.items():
                assert value == pytest.approx(upstream[key], abs=1e-12)


def test_proxy_refuses_changed_coverage_and_zero_truth_error():
    f, topo = frame_fixture()
    assert increment_proxy(f, f, topo)['increment_error_L1'] == 0
    g = f.copy(); g.loc[g.link_id == 'a', 'on_ramp_valid'] = 0
    with pytest.raises(ValueError, match='coverage'):
        increment_proxy(g, f, topo)


def test_macro_weights_groups_and_bootstrap_recomputes_rmse():
    rows = [dict(panel=p, regime=['R1', 'R2', 'R3'][i % 3], date=f'2030-06-{i+1:02d}', n=10, speed_sse=40.+i, lane_flow_sse=900.)
            for p in ['a', 'b', 'c'] for i in range(9)]
    better = [{**r, 'speed_sse': r['speed_sse']*.81} for r in rows]
    fam = {'a': 'A', 'b': 'A', 'c': 'B'}
    ci = paired_macro_bootstrap(rows, better, fam, repetitions=40)
    assert ci['ci95'][0] > 0
    assert ci['pooled_macro_gain'] == pytest.approx(state_macro(better, fam)['S_state']-state_macro(rows, fam)['S_state'])


def test_low_rank_fit_learns_only_supplied_fit_days():
    days = [dict(regime=g, basis=np.ones((15, 1)), base=np.ones((15, 2)), truth=np.ones((15, 2))*1.1,
                 increment_residual=np.zeros(0), increment_design=np.zeros((0, 2)), increment_denominator=0.) for g in ['R1', 'R2', 'R3']]
    m = LowRankCorrection(l1=0, ridge=.001).fit(days)
    np.testing.assert_allclose(m.predict(np.ones((1, 2)), np.ones((1, 1))), 1.1, atol=2e-5)


def test_macro_bootstrap_preserves_separate_season_calendars():
    rows = [dict(panel='a', regime=['R1', 'R2', 'R3'][i % 3], date=f'{year}-06-{i+1:02d}',
                 n=10, speed_sse=40.+i, lane_flow_sse=900.) for year in [2030, 2031] for i in range(9)]
    better = [{**r, 'speed_sse': r['speed_sse']*.81} for r in rows]
    ci = paired_macro_bootstrap(rows, better, {'a': 'A'}, repetitions=30, date_strata={r['date']: r['date'][:4] for r in rows})
    assert ci['strata'] == ['2030', '2031'] and ci['ci95'][0] > 0
