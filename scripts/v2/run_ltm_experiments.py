"""Public-data launch audit; do not construct an unsupported full LTM/MHE."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from trafficflow.v2.score_aligned import network_parameters, ramp_flows, physics_frame, conservation_pairs, canonical
from trafficflow.v2.boundary_marginal import BoundaryMarginal, nuisance_information
from trafficflow.v2.ltm_assimilation import prerequisite_decision
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot, sha256_file, evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit-release', type=Path, default=Path('artifacts/research/v2_score_audit_v2/release'))
    ap.add_argument('--output', type=Path, default=Path('artifacts/research/v2_boundary_precheck_v1'))
    args = ap.parse_args()
    report = Path('reports/v2')/(args.output.name+'.json')
    if args.output.exists() or report.exists():
        raise FileExistsError('Never overwrite a prerequisite audit')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'V2_transport_observation_boundary_launch_gate', 'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'data': args.audit_release.as_posix(), 'panels': ['D12_I5_N', 'D7_I210_W'],
        'gate': 'Do not launch full transport reconstruction if observed-flux increment relative error>=1 or >half links have unresolved internal boundaries',
        'windows': 'First24 nonoverlapping12-pair link windows in date/link order; no assessment-based selection',
        'Gaussian_controls': 'Q positive diagonal, boundary knots linear trend+intercept; prior SD .1/1/10; fixed, joint profile, analytic marginal',
        'restriction': 'Algebraic boundary attenuation diagnostic on public observations, not a transport predictive score',
        'formal_Total': None})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    started = time.perf_counter()
    results = {}
    for panel in ['D12_I5_N', 'D7_I210_W']:
        root = args.audit_release/'corridors'/panel
        paths = sorted((root/'train/mainline_states').rglob('*.parquet'))
        raw = canonical(pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True))
        raw = raw[raw.is_score_eligible.astype(bool)]
        state = raw.groupby(['timestamp', 'link_id'], as_index=False)[['speed_kmh', 'flow_vph']].mean()
        state['is_target'] = 0.
        params, ramps = network_parameters(root), ramp_flows(root)
        topology = pd.read_csv(root/'network/lwr_mainline_topology.csv', dtype=str).set_index('link_id')
        # Split regimes before constructing pairs, as the official scorer does.
        from trafficflow.v2.score_aligned import REGIMES
        import importlib.util
        path = Path('vendor/official/src/task1/baseline_task1_historical_mean.py')
        spec = importlib.util.spec_from_file_location('boundary_official_dates', path)
        official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
        dates = state.timestamp.dt.strftime('%Y-%m-%d')
        unique_dates = pd.Series(dates.unique())
        mapping = dict(zip(unique_dates, official.regime_of_dates(panel, unique_dates)))
        groups = dates.map(mapping)
        pieces = []
        for g in REGIMES:
            f = physics_frame(state[groups == g], panel, g, params, topology, ramps)
            pieces.append(conservation_pairs(f, topology))
        pairs = pd.concat(pieces, ignore_index=True).sort_values(['timestamp', 'link_id'])
        rhs = pairs.dt*(pairs.inflow_vph+pairs.on_ramp_flow_vph-pairs.outflow_vph-pairs.off_ramp_flow_vph)
        error = float(np.abs(pairs.dN-rhs).sum()/(np.abs(pairs.dN).sum()+1e-9))
        observed = set(state.link_id)
        topo_ids = set(topology.index)
        unresolved = {link for link in observed if link not in topology.index or any(
            not ({s.strip() for s in str(topology.loc[link, field]).split(';') if s.strip()} & topo_ids) <= observed
            for field in ['incoming_link_ids', 'outgoing_link_ids'])}
        # Genuine unresolved interface incidence, no invented milepost edges.
        columns = {}
        observed_order = sorted(observed)
        index = {v: i for i, v in enumerate(observed_order)}
        for link in observed_order:
            if link not in topology.index:
                continue
            for field, sign in [('incoming_link_ids', 1.), ('outgoing_link_ids', -1.)]:
                for adjacent in str(topology.loc[link, field]).split(';'):
                    adjacent = adjacent.strip()
                    if adjacent and adjacent != 'nan' and adjacent not in observed:
                        edge = (adjacent, link) if sign > 0 else (link, adjacent)
                        col = columns.setdefault(edge, np.zeros(len(observed_order)))
                        col[index[link]] += sign
        Bspace = np.column_stack(list(columns.values())) if columns else np.zeros((len(observed_order), 0))
        singular = np.linalg.svd(Bspace, compute_uv=False)
        boundary_rank = int((singular > 1e-8).sum())
        windows = []
        for (date, link), group in pairs.groupby([pairs.timestamp.dt.strftime('%Y-%m-%d'), 'link_id'], sort=True):
            group = group.sort_values('timestamp')
            # Explicit contiguous intervals only.
            for offset in range(0, len(group)-11, 12):
                f = group.iloc[offset:offset+12]
                if not np.all(np.diff(f.timestamp.dt.as_unit('ns').astype('int64').to_numpy()) == 300_000_000_000):
                    continue
                net = (f.dt*(f.inflow_vph+f.on_ramp_flow_vph-f.outflow_vph-f.off_ramp_flow_vph)).to_numpy(float)
                residual = f.dN.to_numpy(float)-net
                scale = max(float(np.std(residual)), 1e-3)
                d = residual/scale
                B = np.column_stack([np.ones(12), np.linspace(-1, 1, 12)])
                controls = []
                for sd in [.1, 1., 10.]:
                    begin = time.perf_counter()
                    model = BoundaryMarginal(B, np.ones(12), np.eye(2)*sd)
                    eta, joint = model.joint_map(d)
                    marginal = model.loss(d, include_logdet=False)
                    np.testing.assert_allclose(joint, marginal, atol=1e-9)
                    controls.append({'prior_sd': sd, 'no_constraint_objective': 0., 'fixed_boundary_quadratic': float(.5*d@d),
                        'joint_profile_quadratic': joint, 'marginal_quadratic': marginal, 'marginal_nll_without_constant': model.loss(d),
                        'remaining_residual_norm': float(np.linalg.norm(d-B@eta)), 'solve_seconds': time.perf_counter()-begin})
                D = np.eye(12, 13, k=1)-np.eye(12, 13)
                information = nuisance_information(D, B)
                unrestricted = nuisance_information(D, np.eye(12))
                windows.append({'date': date, 'link': link, 'scale_vehicles': scale, 'controls': controls,
                                'low_rank_boundary_information': information, 'unrestricted_boundary_information': unrestricted})
                if len(windows) == 24:
                    break
            if len(windows) == 24:
                break
        p = params.set_index('link_id').loc[observed_order]
        wave = p.capacity_vph/(p.k_jam-p.critical_density)
        cfl_seconds = float(np.min(3600*p.length_km/np.maximum(p.free_speed_kmh, wave)))
        decision = prerequisite_decision(error, len(unresolved)/len(observed))
        results[panel] = {'n_pairs': len(pairs), 'observed_flow_inventory_relative_error': error,
            'observed_links': len(observed), 'unresolved_internal_links': len(unresolved),
            'boundary_incidence_shape': list(Bspace.shape), 'boundary_incidence_rank': boundary_rank,
            'boundary_incidence_singular_values': singular.tolist(), 'windows': windows,
            'explicit_CTM_substeps_per_5min_at_CFL08': int(np.ceil(300/(.8*cfl_seconds))),
            'decision': decision, 'predictive_gain_vs_strong_model': None,
            'full_CTM_LTM_cost_accuracy_comparison': 'Not launched: transport observation gate failed; inventing boundary curves would not be a fair prediction experiment',
            'data_sha256': {p.as_posix(): sha256_file(p) for p in paths}}
        print(panel, 'flux error', error, 'unresolved', len(unresolved), 'launch', decision['launch_full_ltm_mhe'], flush=True)
    result = {'registered': registered, 'source_snapshot': snapshot, 'panels': results,
              'seconds': time.perf_counter()-started, 'runtime': evidence(), 'formal_Total': None,
              'conclusion': 'Exact Gaussian profile/marginal equality is numerical algebra, not prediction evidence; no full transport model promoted'}
    report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
