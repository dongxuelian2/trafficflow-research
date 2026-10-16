"""Forward-held-out residual GP study; calibration sealed before assessment.

Consumes frozen phase-one caches without retraining the strong State predictor.
The same-day posterior uses <=512 four-channel-counterfactual visible anchors
and <=192 inducing points, with matching ordinary kernels and graph smoothing.
"""
import argparse
import json
import pickle
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from trafficflow.data import Release, VALUES
from trafficflow.experiments import partition_date
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import register_protocol
from trafficflow.v2.directional_kernel import (trusted_chains, legal_gate, SparseResidualGP,
                                              counterfactual_sampled_anchors)
from trafficflow.v2.score_aligned import (KEY, canonical, state_targets, state_statistics, state_macro,
    anchored_frame, network_parameters, ramp_flows, physics_frame, conservation_pairs)
from trafficflow.v2.anchored_increment import increment_proxy
from trafficflow.v2.score_decision import paired_macro_bootstrap
from trafficflow.state_assimilation import AssimilationParameters, assimilate_residuals
from trafficflow.state import StateModel
from run_state_experiments import CASES


SEED = 20261016
MODES = ['ordinary', 'gated_ordinary', 'directional']
STRENGTHS = [0., .25, .5, 1.]


class FixedPairScorer:
    """Exact vector evaluation after official pair guards have run once/day.

    The counterfactual implementation already requires one station/link/time.
    All finite candidate predictions preserve the official pair universe, so
    only q/v and its indexed differences change between candidates.
    """
    def __init__(self, raw, target, official, official_lanes, trusted, true_frame, topology):
        true_frame = true_frame.reset_index(drop=True)
        if raw.duplicated(['timestamp', 'link_id']).any():
            raise ValueError('Fixed-pair vector scorer requires unique station/link')
        self.truth = official[VALUES].to_numpy(float)
        self.lanes, self.trusted = np.asarray(official_lanes), np.asarray(trusted)
        self.official_idx = pd.MultiIndex.from_frame(target[KEY]).get_indexer(pd.MultiIndex.from_frame(official[KEY]))
        self.values = true_frame[VALUES].to_numpy(float)
        if not np.isfinite(self.values).all():
            raise ValueError('Pair coverage invariant needs finite eligible anchors')
        full = pd.MultiIndex.from_frame(true_frame[['timestamp', 'link_id']])
        self.target_idx = full.get_indexer(pd.MultiIndex.from_frame(target[['timestamp', 'link_id']]))
        if (self.target_idx < 0).any():
            raise ValueError('Published target missing from eligible anchored field')
        self.length = true_frame.length_km.to_numpy(float)
        pairs = conservation_pairs(true_frame, topology)
        self.left = full.get_indexer(pd.MultiIndex.from_frame(pairs[['timestamp', 'link_id']]))
        right = pairs[['next_timestamp', 'link_id']].rename(columns={'next_timestamp': 'timestamp'})
        self.right = full.get_indexer(pd.MultiIndex.from_frame(right))
        if (self.left < 0).any() or (self.right < 0).any():
            raise ValueError('Pair endpoint absent from anchored field')
        self.true_increment = pairs.dN.to_numpy(float)
        self.denominator = float(np.abs(self.true_increment).sum())
        endpoint_rows = np.unique(np.r_[true_frame.groupby('link_id').timestamp.idxmin().to_numpy(int),
                                        true_frame.groupby('link_id').timestamp.idxmax().to_numpy(int)])
        self.endpoint_rows = endpoint_rows
        self.endpoint_times = (pd.DatetimeIndex(true_frame.timestamp).as_unit('ns').asi8[endpoint_rows]//10**9).astype(int)
        self.endpoint_links = true_frame.link_id.to_numpy(str)[endpoint_rows]
        self.endpoint_ramps = true_frame[['on_ramp_valid', 'off_ramp_valid']].to_numpy(int)[endpoint_rows]
        self.endpoint_truth = true_frame.accumulation_N.to_numpy(float)[endpoint_rows]
        self.eligible_links = sorted(set(true_frame.link_id))

    def statistics(self, prediction, subset=None):
        values = np.asarray(prediction, float)
        valid = np.isfinite(self.truth).all(axis=1)
        if subset is not None:
            valid &= subset
        mapped = np.zeros_like(self.truth)
        present = self.official_idx >= 0
        mapped[present] = values[self.official_idx[present]]
        errors = mapped-self.truth
        return {'n': int(valid.sum()), 'speed_sse': float(np.square(errors[valid, 0]).sum()),
                'lane_flow_sse': float(np.square(errors[valid, 1]/self.lanes[valid]).sum()),
                'missing': int((valid & ~present).sum())}

    def increment(self, prediction):
        values = self.values.copy()
        values[self.target_idx] = prediction
        inventory = self.length*values[:, 1]/np.maximum(values[:, 0], 1.)
        predicted_increment = inventory[self.right]-inventory[self.left]
        numerator = float(np.abs(predicted_increment-self.true_increment).sum())
        return {'increment_error_L1': numerator, 'increment_truth_L1': self.denominator,
                'increment_relative_L1': numerator/(self.denominator+1e-9), 'n_pairs': len(self.left),
                'increment_pairs': len(self.left), 'formal_Physics': None}

    def boundaries(self, prediction):
        values = self.values.copy()
        values[self.target_idx] = prediction
        inventory = self.length*values[:, 1]/np.maximum(values[:, 0], 1.)
        return [[int(t), str(link), float(inventory[index]), float(truth), int(ramp[0]), int(ramp[1])]
                for t, link, index, truth, ramp in zip(self.endpoint_times, self.endpoint_links, self.endpoint_rows,
                                                      self.endpoint_truth, self.endpoint_ramps)]


def add_midnight_pairs(rows, topology):
    """Restore consecutive midnight pairs across dates within panel/regime.

    Pair coverage uses the complete selected-date eligible link set, exactly
    as the official link-set guard. No connection across regime boundaries.
    Cross-date pairs are charged to their start date for reporting only.
    """
    topo_ids = set(topology.index)
    for panel, regime in sorted({(r['panel'], r['regime']) for r in rows}):
        subset = [r for r in rows if r['panel'] == panel and r['regime'] == regime]
        submitted = set().union(*(set(r['_eligible_links']) for r in subset))
        if any(set(r['_eligible_links']) != submitted for r in subset):
            raise ValueError('Date-varying full-link pair coverage requires joint frame scoring; fixed-pair shortcut rejected')
        good = {}
        for link in submitted:
            good[link] = link in topo_ids and all(
                ({s.strip() for s in str(topology.loc[link, col]).split(';') if s.strip()} & topo_ids) <= submitted
                for col in ['incoming_link_ids', 'outgoing_link_ids']) if len(topology) else True
        by_link = {}
        for owner, row in enumerate(subset):
            for endpoint in row['_boundary_endpoints']:
                by_link.setdefault(endpoint[1], []).append((endpoint, owner))
        for link, points in by_link.items():
            if not good[link]:
                continue
            points.sort(key=lambda p: p[0][0])
            for (left, owner), (right, next_owner) in zip(points[:-1], points[1:]):
                dt = right[0]-left[0]
                if owner == next_owner or not 0 < dt <= 306 or min(left[4:6]) < 1:
                    continue
                pred, truth = right[2]-left[2], right[3]-left[3]
                row = subset[owner]
                row['increment_error_L1'] += abs(pred-truth)
                row['increment_truth_L1'] += abs(truth)
                row['increment_pairs'] += 1
                row['n_pairs'] += 1
                row['increment_midnight_pairs'] += 1
    for row in rows:
        row['increment_relative_L1'] = row['increment_error_L1']/(row['increment_truth_L1']+1e-9)
        row.pop('_boundary_endpoints'); row.pop('_eligible_links')


def cache_coordinates(cache, geometry, net):
    """Only explicitly lawful cache arrays are read here, never target_truth."""
    base, u = cache['base'], cache['basis']
    idx, times, links = cache['target_indices'], cache['times'], cache['links'].astype(str)
    if tuple(links) != geometry.links:
        raise ValueError('Cached grid link order changed')
    ti, li = idx//len(links), idx % len(links)
    dt = pd.to_datetime(times.astype(str), utc=True)
    hours = np.asarray(dt.hour+dt.minute/60+dt.second/3600)
    vf = net.free_speed_kmh.reindex(links).to_numpy(float)[li]
    gate = legal_gate(base[:, 0]*25, vf)
    # State similarities restrict historical residual transfer to similar legal
    # baselines, local interpolation disagreement and observation age.
    state = np.column_stack([base, u[:, 1], u[:, 3], .25*u[:, 7], .25*u[:, 8]])
    x = np.column_stack([geometry.distance_km[li], hours[ti], gate, geometry.component[li], state])
    return x, ti, li


def same_day_corrections(cache, x_target, masked, geometry, net, topology, base_model, scale, folder, role, date):
    """All models condition on identical counterfactually predicted anchors."""
    day_seed = SEED+int(date.replace('-', ''))
    begin = time.perf_counter()
    anchors = counterfactual_sampled_anchors(base_model, masked, net, topology, cache, geometry,
                                            maximum=512, folds=2, seed=day_seed)
    n = len(anchors['flat_indices'])
    # Build exactly the lawful state embedding used for cached missing targets.
    u = np.zeros((n, 12))
    for position, features, columns in anchors['lawful_feature_groups']:
        index = {c: j for j, c in enumerate(columns)}
        def get(column):
            return np.nan_to_num(features[:, index[column]], nan=0., posinf=0., neginf=0.)
        u[position, 1] = get('v_temporal')-anchors['prior'][position, 0]
        u[position, 3] = get('q_temporal')-anchors['prior'][position, 1]
        u[position, 7] = np.minimum(get('v_age'), 2.)
        u[position, 8] = np.minimum(get('q_age'), 2.)
    u = np.clip(u, -3., 3.)
    anchor_cache = {'base': anchors['prior'], 'basis': u, 'target_indices': anchors['flat_indices'],
                    'times': cache['times'], 'links': cache['links']}
    x_anchor, ti, li = cache_coordinates(anchor_cache, geometry, net)
    output, model_records = {}, {}
    day_folder = folder/role/date; day_folder.mkdir(parents=True)
    anchor_path = day_folder/'legal_counterfactual_anchors.npz'
    np.savez_compressed(anchor_path, coordinates=x_anchor, residual=anchors['residual'],
                        prior=anchors['prior'], flat_indices=anchors['flat_indices'], residual_scale=scale)
    for mode in MODES:
        fitted = SparseResidualGP(mode, seed=day_seed).fit(x_anchor, anchors['residual']/scale)
        output[mode] = fitted.predict(x_target)*scale
        path = day_folder/(mode+'.pkl'); path.write_bytes(pickle.dumps(fitted))
        model_records[mode] = {**fitted.fit_record, 'sha256': sha256_file(path)}
    prior = np.zeros((len(cache['times']), len(geometry.links), 2))
    strict = np.full_like(prior, np.nan)
    strict[ti, li] = anchors['residual']
    # Same anchor keys/residuals and existing sparse graph smoother; no
    # reconstruction of visible values through an unmasked base prediction.
    graph, diagnostics = assimilate_residuals(prior, strict, np.zeros((*prior.shape[:2], 6)), geometry.edges,
        AssimilationParameters(prior_precision=1., temporal_precision=1., spatial_precision=1.))
    target_idx = cache['target_indices']
    output['graph_residual'] = graph.reshape(-1, 2)[target_idx]
    path = day_folder/'graph_residual.npy'; np.save(path, graph)
    model_records['graph_residual'] = {'sha256': sha256_file(path), 'diagnostics': diagnostics,
                                     'precisions': [1., 1., 1.], 'n_anchors': n}
    record = {'date': date, 'role': role, 'n_anchors': n, 'n_available': anchors['n_available'],
              'anchor_sha256': sha256_file(anchor_path), 'four_channel_folds': anchors['certificates'],
              'models': model_records, 'seconds': time.perf_counter()-begin}
    print(date, role, 'same-day CF anchors/GP', record['seconds'], flush=True)
    return output, record


def aggregate(rows, families):
    result = state_macro(rows, families)
    result['trusted_component_State'] = state_macro([{**r, **r['trusted_state_stats']} for r in rows], families)
    result['trusted_target_fraction'] = float(sum(r['trusted_state_stats']['n'] for r in rows)/max(sum(r['n'] for r in rows), 1))
    result['increment_relative_L1'] = float(np.mean([
        sum(r['increment_error_L1'] for r in rows if r['regime'] == g)/
        (sum(r['increment_truth_L1'] for r in rows if r['regime'] == g)+1e-9)
        for g in ['R1', 'R2', 'R3']]))
    result['increment_pair_count'] = int(sum(r['increment_pairs'] for r in rows))
    result['increment_midnight_pair_count'] = int(sum(r['increment_midnight_pairs'] for r in rows))
    for label in ['sharp_drop', 'recovery']:
        sub = [r[label] for r in rows]
        n = sum(r['n'] for r in sub)
        result[label] = {'n': n, 'rmse_speed': float(np.sqrt(sum(r['speed_sse'] for r in sub)/n)) if n else None,
                         'rmse_flow_lane': float(np.sqrt(sum(r['lane_flow_sse'] for r in sub)/n)) if n else None}
    result['days'] = rows
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-root', type=Path, default=Path('artifacts/research/v2_state_v2'))
    ap.add_argument('--output', type=Path, default=Path('artifacts/research/GP_v2_same_day_v3'))
    ap.add_argument('--case', action='append')
    args = ap.parse_args()
    report = Path('reports/v2')/(args.output.name+'.json')
    if args.output.exists() or report.exists():
        raise FileExistsError('Use a new output; never overwrite experiment evidence')
    cases = [c for c in CASES if args.case is None or c[0] in args.case]
    if not cases:
        raise ValueError('No registered case selected')
    # Verify complete caches before registering; do not inspect held-out arrays.
    for name, _, _, first, fit_end, cal_end, assess_end in cases:
        ranges = [('correction_fit', first, fit_end),
                  ('calibration', str((pd.Timestamp(fit_end)+pd.Timedelta(days=1)).date()), cal_end),
                  ('assessment', str((pd.Timestamp(cal_end)+pd.Timedelta(days=1)).date()), assess_end)]
        for role, begin, end in ranges:
            expected_dates = {str(d.date()) for d in pd.date_range(begin, end)}
            actual_dates = {p.stem for p in (args.cache_root/name/role).glob('*.npz')}
            if len(expected_dates) != 14 or actual_dates != expected_dates:
                raise ValueError(f'Expected exact 14 registered {name}/{role} cache dates')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'V2_GP_same_day_counterfactual_directional_residual_Nystrom', 'code_base_commit': commit,
        'cases': cases, 'seed': SEED, 'kernels': MODES,
        'kernel': {'length_x_km': 2., 'length_t_hours': .5, 'downstream_kmh': 80., 'upstream_kmh': -15.,
                   'state_length': 1., 'gate': 'logistic((.6-predicted_speed/free_speed)/.08), sqrt branch weights'},
        'budget': {'fit_labels_per_date': 512, 'same_day_anchors': 512, 'anchor_counterfactual_folds': 2,
                   'inducing': 192, 'batch': 512, 'noise_variance': .10,
                   'dense_full_GP': False, 'gpu_used': 'frozen attention inference only', 'RSS_budget_MiB': 8192,
                   'BLAS_threads': 1, 'frozen_torch_CPU_threads_max': 2},
        'residual_roles': 'Single chronological forward cross-fit: base fit/checkpoint <=Oct29 or Dec29; next14 dates correction fit, next14 calibration, next14 assessment; I210 winter transfers frozen autumn prior. No in-sample base labels.',
        'temporal_model': 'Same-day local posterior from <=512 lawful visible anchors. Each anchor frozen prediction obtained while all four own measurement channels are withheld in one of2 seeded folds; target predictions remain the original frozen cache. Hours since current day start; no cross-date anchor pooling.',
        'residual_amplitude': 'Per-channel RMS fitted only on <=512 held-out correction_fit target residuals per date, clipped [.05,1]. Fixed before calibration. Same amplitudes for every kernel.',
        'graph_control': 'Existing state_assimilation.assimilate_residuals with the exact same sampled counterfactual anchor residuals and trusted edges; fixed precisions (1,1,1). Not a claim to reproduce historical A58 exact settings.',
        'selection': 'Highest calibration official State among strengths [0,.25,.5,1] separately per preregistered kernel/control; ties prefer 0. Save seal then open assessment.',
        'assessment': 'Report baseline, all matched cal-selected strengths and fixed full-strength variants. No selection on assessment.',
        'primary_comparison': 'directional vs ordinary and gated_ordinary selected on calibration; graph control and frozen strong model also reported',
        'bootstrap': '2000 paired circular date-block samples of length7; nonlinear State macro recomputed; omit missing-regime draws',
        'promotion': 'Requires State gain>=.002, pooled lower95>0, each direction gain>=-.001, both seasons and independent gain over ordinary kernel. Otherwise do not expand.',
        'metric_scope': 'Official eligibility/regime/family formula on selected two directions, not all ten-direction competition State. Trusted-component-only State and coverage are secondary diagnostics. Public observed inventory-increment proxy only; no hidden b* equality.',
        'vector_scoring': 'One-station finite-target invariants keep official pairs fixed; first calibration/assessment date of each case every candidate checked against original scorer. Cross-date same-regime midnight pairs restored with exact topology/start-ramp/dt guards.',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    started, results, failures = time.perf_counter(), {}, []
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    release = Release('data/kaggle_public')
    family_map = {p['corridor_id']: p['family_id'] for p in json.loads(Path('vendor/official/config/corridors.json').read_text())['panels']}
    with threadpool_limits(limits=1):
        for name, panel, model_path, first, fit_end, cal_end, assess_end in cases:
            case_start = time.perf_counter()
            folder = args.output/name; folder.mkdir()
            net, topology = release.network(panel), release.topology(panel)
            base_model = StateModel.load(Path(model_path))
            masked_paths = {partition_date(release, p): p for p in release.files(panel, 'train')}
            raw_paths = {partition_date(release, p): p for p in release.files(panel, 'train', False)}
            template = canonical(release.template(1, panel, 'train'))
            # Date lookup is built once; strftime of the multi-million-row
            # public template inside every date loop is prohibitively slow.
            template_dates = template.timestamp.dt.floor('D')
            first_mask = release.read(masked_paths[first])
            first_cache = sorted((args.cache_root/name/'correction_fit').glob('*.npz'))[0]
            with np.load(first_cache, allow_pickle=False) as z:
                links = z['links'].astype(str)
            geometry = trusted_chains(topology, net.length_km, first_mask.groupby('link_id').milepost.mean(), links)
            (folder/'topology_audit.json').write_text(json.dumps(geometry.audit, indent=2), encoding='utf-8')
            if geometry.audit['n_trusted_measured'] < 2:
                failure = {'case': name, 'reason': 'No mileage-validated connected chain; directional model disabled'}
                failures.append(failure); print('FAILED', failure, flush=True); continue
            provenance, train, daily_models, scorer_verifications = [], [], [], []
            for path in sorted((args.cache_root/name/'correction_fit').glob('*.npz')):
                if not first <= path.stem <= fit_end:
                    raise ValueError('Correction-fit cache outside registered forward split')
                with np.load(path, allow_pickle=False) as z:
                    x, ti, li = cache_coordinates(z, geometry, net)
                    residual = z['target_truth']-z['base']
                rng = np.random.default_rng(SEED+int(path.stem.replace('-', '')))
                trusted = np.flatnonzero(x[:, 3] >= 0)
                chosen = np.sort(rng.choice(trusted, min(512, len(trusted)), replace=False))
                train.append((x[chosen], residual[chosen], ti[chosen], li[chosen]))
                provenance.append({'role': 'correction_fit', 'date': path.stem, 'cache_sha256': sha256_file(path)})
            x_fit, y_fit = np.concatenate([d[0] for d in train]), np.concatenate([d[1] for d in train])
            scale = np.clip(np.sqrt(np.mean(y_fit*y_fit, axis=0)), .05, 1.)
            fit_records = {'n_forward_heldout_fit_residuals': len(y_fit), 'residual_scale': scale.tolist(),
                           'base_training_residuals_used': False, 'noise_variance_after_scaling': .10}
            path = folder/'fitted_hyperparameters.json'
            path.write_text(json.dumps(fit_records, indent=2), encoding='utf-8')
            fit_records['sha256'] = sha256_file(path)
            del train, x_fit, y_fit
            params = network_parameters(release.root/'corridors'/panel)
            ramps = ramp_flows(release.root/'corridors'/panel)

            def evaluate(role, strengths):
                records = {key: [] for key in strengths}
                for day_number, cache in enumerate(sorted((args.cache_root/name/role).glob('*.npz'))):
                    date = cache.stem
                    with np.load(cache, allow_pickle=False) as z:
                        x, ti, li = cache_coordinates(z, geometry, net)
                        base, lane = z['base'], z['target_lanes']
                        regime = str(z['regime'])
                    begin_date = pd.Timestamp(date, tz='UTC')
                    target = template[template_dates == begin_date].copy()
                    if len(target) != len(base):
                        raise ValueError('Cached target count does not match template')
                    target_links = target.link_id.to_numpy(str)
                    if not np.array_equal(target_links, np.asarray(geometry.links)[li]):
                        raise ValueError('Cached target order does not match template')
                    # Both link and timestamp order are part of the cache key.
                    with np.load(cache, allow_pickle=False) as z:
                        cached_times = pd.to_datetime(z['times'].astype(str), utc=True)
                    if not np.array_equal(target.timestamp.to_numpy(), cached_times[ti].to_numpy()):
                        raise ValueError('Cached timestamp order does not match template')
                    raw, masked = canonical(release.read(raw_paths[date])), release.read(masked_paths[date])
                    official = state_targets(raw, masked)
                    truth = target[KEY].merge(raw[KEY+VALUES], on=KEY, validate='one_to_one')[VALUES].to_numpy(float)
                    stationary_lanes = release.station_lanes(panel, target)
                    official_lanes = release.station_lanes(panel, official)
                    trusted_map = dict(zip(geometry.links, geometry.component >= 0))
                    trusted_official = official.link_id.map(trusted_map).fillna(False).to_numpy(bool)
                    with np.load(cache, allow_pickle=False) as z:
                        corrections, model_record = same_day_corrections(z, x, masked, geometry, net, topology,
                            base_model, scale, folder, role, date)
                    daily_models.append(model_record)
                    true_anchor = raw[raw.is_score_eligible.astype(bool)].groupby(['timestamp', 'link_id'], as_index=False)[VALUES].mean()
                    true_anchor['is_target'] = 0.
                    local_ramps = ramps[(ramps.timestamp >= begin_date) &
                                        (ramps.timestamp < begin_date+pd.Timedelta(days=1))] if len(ramps) else ramps
                    true_frame = physics_frame(true_anchor, panel, regime, params, topology, local_ramps)
                    scorer = FixedPairScorer(raw, target, official, official_lanes, trusted_official, true_frame, topology)
                    delta = raw.pivot_table(index='timestamp', columns='link_id', values='speed_kmh', aggfunc='mean').diff()
                    selections = {}
                    for label, condition in [('sharp_drop', delta <= -20), ('recovery', delta >= 20)]:
                        selections[label] = condition.stack().reindex(pd.MultiIndex.from_frame(target[['timestamp', 'link_id']])).fillna(False).to_numpy(bool)
                    for key, (mode, strength) in strengths.items():
                        normalized = base if mode is None or not strength else base+strength*corrections[mode]
                        prediction = target[KEY].copy()
                        prediction[VALUES] = np.column_stack([
                            np.clip(normalized[:, 0]*25, 0, target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05),
                            np.clip(normalized[:, 1]*600*lane, 0, target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)])
                        predicted_values = prediction[VALUES].to_numpy(float)
                        if not np.isfinite(predicted_values).all():
                            raise ValueError('Candidate must preserve finite pair coverage')
                        increment = scorer.increment(predicted_values)
                        stats = scorer.statistics(predicted_values)
                        trusted_stats = scorer.statistics(predicted_values, trusted_official)
                        if day_number == 0:
                            anchor, missing = anchored_frame(raw, target, prediction)
                            if missing:
                                raise ValueError('Candidate dropped a published target')
                            frame = physics_frame(anchor, panel, regime, params, topology, local_ramps)
                            reference_increment = increment_proxy(frame, true_frame, topology)
                            reference_stats = state_statistics(official, prediction, official_lanes)
                            reference_trusted = state_statistics(official.loc[trusted_official], prediction,
                                                                 official_lanes[trusted_official])
                            errors = {}
                            for label, actual, expected in [('State', stats, reference_stats),
                                    ('trusted_State', trusted_stats, reference_trusted),
                                    ('increment', increment, reference_increment)]:
                                for column in expected:
                                    if expected[column] is None:
                                        continue
                                    difference = abs(actual[column]-expected[column])
                                    errors[label+'_'+column] = difference
                                    if not np.isclose(actual[column], expected[column], rtol=1e-12, atol=1e-8):
                                        raise ValueError(f'Vector scorer mismatch {role}/{date}/{key}/{label}/{column}: {difference}')
                            scorer_verifications.append({'role': role, 'date': date, 'candidate': key,
                                                         'absolute_errors': errors})
                        row = {'date': date, 'panel': panel, 'regime': regime,
                               **stats, **increment, 'trusted_state_stats': trusted_stats,
                               '_boundary_endpoints': scorer.boundaries(predicted_values),
                               '_eligible_links': scorer.eligible_links, 'increment_midnight_pairs': 0}
                        # Keep the public proxy's exact pair count under a stable key.
                        row['increment_pairs'] = int(increment.get('increment_pairs', increment.get('n_pairs', 0)))
                        errors = prediction[VALUES].to_numpy(float)-truth
                        for label, selected in selections.items():
                            row[label] = {'n': int(selected.sum()), 'speed_sse': float(np.square(errors[selected, 0]).sum()),
                                          'lane_flow_sse': float(np.square(errors[selected, 1]/stationary_lanes[selected]).sum())}
                        records[key].append(row)
                    provenance.append({'role': role, 'date': date, 'cache_sha256': sha256_file(cache),
                                       'masked_sha256': sha256_file(masked_paths[date]), 'raw_sha256': sha256_file(raw_paths[date])})
                    print(name, role, date, 'evaluated', flush=True)
                for rows in records.values():
                    add_midnight_pairs(rows, topology)
                return {k: aggregate(rows, {panel: family_map[panel]}) for k, rows in records.items()}

            candidates = {'baseline': (None, 0.)}
            candidates.update({f'{mode}@{strength}': (mode, strength) for mode in MODES+['graph_residual'] for strength in STRENGTHS[1:]})
            cal = evaluate('calibration', candidates)
            selected = {}
            for mode in MODES+['graph_residual']:
                keys = ['baseline']+[k for k in candidates if k.startswith(mode+'@')]
                selected[mode] = max(keys, key=lambda k: (cal[k]['S_state'], -candidates[k][1]))
            seal = {'selected': selected, 'calibration': cal, 'assessment_policy': 'selected plus each fixed full strength'}
            seal_path = folder/'selection_sealed.json'
            seal_path.write_text(json.dumps(seal, indent=2, allow_nan=False), encoding='utf-8')
            seal_sha = sha256_file(seal_path)
            chosen_keys = dict.fromkeys(['baseline', *selected.values(), *[f'{mode}@1.0' for mode in MODES+['graph_residual']]])
            assessment = evaluate('assessment', {k: candidates[k] for k in chosen_keys})
            intervals = {k: paired_macro_bootstrap(assessment['baseline']['days'], value['days'], {panel: family_map[panel]}) for k, value in assessment.items()}
            directional_key = selected['directional']
            matched_intervals = {mode: paired_macro_bootstrap(assessment[selected[mode]]['days'], assessment[directional_key]['days'], {panel: family_map[panel]})
                                 for mode in ['ordinary', 'gated_ordinary', 'graph_residual']}
            result = {'panel': panel, 'base_model': model_path, 'base_sha256': sha256_file(model_path),
                      'topology': geometry.audit, 'source_hashes': provenance, 'fit': fit_records,
                      'daily_posterior_models': daily_models,
                      'vector_scorer_verifications': scorer_verifications,
                      'selection': selected, 'selection_sha256_before_assessment': seal_sha,
                      'calibration': cal, 'assessment': assessment, 'intervals_vs_baseline': intervals,
                      'directional_vs_matched': matched_intervals, 'seconds': time.perf_counter()-case_start,
                      'runtime': evidence(gpu_used=bool(torch.cuda.is_available()))}
            results[name] = result
            (folder/'experiment.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            print(name, 'RESULT', {k: v['S_state'] for k, v in assessment.items()}, selected, flush=True)
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    pooled = {}
    if results:
        selected_families = {r['panel']: family_map[r['panel']] for r in results.values()}
        pooled_rows = {'baseline': [row for r in results.values() for row in r['assessment']['baseline']['days']]}
        pooled_rows.update({mode: [row for r in results.values() for row in r['assessment'][r['selection'][mode]]['days']]
                            for mode in MODES+['graph_residual']})
        pooled = {mode: aggregate(rows, selected_families) for mode, rows in pooled_rows.items()}
        for mode in MODES+['graph_residual']:
            pooled[mode]['interval_vs_baseline'] = paired_macro_bootstrap(pooled_rows['baseline'], pooled_rows[mode], selected_families)
        independent = {mode: paired_macro_bootstrap(pooled_rows[mode], pooled_rows['directional'], selected_families)
                       for mode in ['ordinary', 'gated_ordinary', 'graph_residual']}
        seasons_complete = all(any(name.endswith('_'+season) and r['panel'] == panel for name, r in results.items())
                               for panel in ['D12_I5_N', 'D7_I210_W'] for season in ['autumn', 'winter'])
        interval = pooled['directional']['interval_vs_baseline']
        direction_gain = {panel: pooled['directional']['panels'][panel]-pooled['baseline']['panels'][panel] for panel in selected_families}
        stable_comparison = all(independent[mode]['ci95'] is not None and independent[mode]['ci95'][0] > 0 and
            all(r['directional_vs_matched'][mode]['pooled_macro_gain'] > 0 for r in results.values())
            for mode in ['ordinary', 'gated_ordinary'])
        passed = (seasons_complete and interval['pooled_macro_gain'] >= .002 and interval['ci95'] is not None and
                  interval['ci95'][0] > 0 and min(direction_gain.values()) >= -.001 and stable_comparison)
        pooled['directional_independent_comparisons'] = independent
        pooled['promotion_gate'] = {'pass': bool(passed), 'both_directions_and_seasons': seasons_complete,
                                    'direction_gain': direction_gain, 'stable_kernel_comparison': bool(stable_comparison)}
    output = {'registered': registered, 'source_snapshot': snapshot, 'cases': results, 'failures': failures, 'pooled': pooled,
              'seconds': time.perf_counter()-started, 'runtime': evidence(gpu_used=bool(torch.cuda.is_available())),
              'GPU_peak_allocated_MiB': torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0.,
              'GPU_peak_reserved_MiB': torch.cuda.max_memory_reserved()/1024**2 if torch.cuda.is_available() else 0.,
              'formal_State': None, 'formal_Physics': None, 'formal_Total': None,
              'decision': 'Promote only if pooled.promotion_gate passes; otherwise retain frozen baseline and stop expansion of this bounded GP variant.',
              'limitations': ['Sparse <=512 visible anchors per date; two-fold sampled-anchor counterfactual context adds missingness and is not an unbiased posterior guarantee',
                  'Graph control reuses existing quadratic solver and identical counterfactual anchors; historical A58/A59 exact settings not rerun',
                  'Observed increment proxy uses official public topology/ramp pair guards; hidden boundary b* not observed',
                  'Selected two directions only; previously used public development calendar; no new Kaggle score']}
    report.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print('REPORT', report.as_posix(), 'seconds', output['seconds'], flush=True)


def replay_lightgbm(primary_path, cache_root, output):
    """Fixed pure-LGB diagnostic after GP selection; no GP choices change.

    Also compare the complete calendar's increment proxy with one joint call
    of the original pair scorer, independently validating midnight stitching.
    """
    from trafficflow.state import StateFeatures
    import hashlib
    primary_path, cache_root, output = map(Path, [primary_path, cache_root, output])
    report_path = Path('reports/v2')/(output.name+'.json')
    if output.exists() or report_path.exists():
        raise FileExistsError('Use a new fixed replay output')
    primary = json.loads(primary_path.read_text())
    cases = primary['registered']['protocol']['cases']
    registration = register_protocol(report_path.with_suffix('.registered.json'), {
        'id': 'V2_GP_fixed_pure_LightGBM_diagnostic_and_joint_pair_replay',
        'primary_GP_report_sha256': sha256_file(primary_path), 'cases': cases,
        'selection': 'None. Original nested model.baseline is a fixed pure LGB diagnostic; all GP seals remain unchanged.',
        'dates': 'All original14 calibration +14 assessment dates per case',
        'bounds': 'Same [0,1.05*free_speed] and [0,1.05*capacity] as frozen primary',
        'additional_verification': 'For every role/regime jointly call original increment_proxy on all selected timestamps; compare with stitched counts/L1 for pureLGB and original frozen baseline',
        'CPU_threads': 2, 'gpu_used': False})
    output.mkdir(parents=True)
    snapshot = capture_source_snapshot(output/'source_snapshot.zip')
    started, results = time.perf_counter(), {}
    release = Release('data/kaggle_public')
    families = {p['corridor_id']: p['family_id'] for p in json.loads(Path('vendor/official/config/corridors.json').read_text())['panels']}
    with threadpool_limits(limits=1):
        for name, panel, model_path, first, *_ in cases:
            begin_case = time.perf_counter()
            folder = output/name; folder.mkdir()
            strong = StateModel.load(Path(model_path))
            model = strong.baseline
            if model.method != 'lightgbm':
                raise ValueError('Fixed nested baseline is not pure LightGBM')
            model_hash = hashlib.sha256(pickle.dumps(model)).hexdigest()
            # Operational thread cap only; no learned parameter changes.
            for fitted in model.models:
                fitted.set_params(n_jobs=2)
            net, topology = release.network(panel), release.topology(panel)
            template = canonical(release.template(1, panel, 'train'))
            template_dates = template.timestamp.dt.floor('D')
            masked_paths = {partition_date(release, p): p for p in release.files(panel, 'train')}
            raw_paths = {partition_date(release, p): p for p in release.files(panel, 'train', False)}
            params = network_parameters(release.root/'corridors'/panel)
            ramps = ramp_flows(release.root/'corridors'/panel)
            with np.load(sorted((cache_root/name/'correction_fit').glob('*.npz'))[0], allow_pickle=False) as z:
                links = z['links'].astype(str)
            first_masked = release.read(masked_paths[first])
            geometry = trusted_chains(topology, net.length_km, first_masked.groupby('link_id').milepost.mean(), links)
            source_records, parity = [], []
            case_result = {}
            for role in ['calibration', 'assessment']:
                rows, true_parts, lgb_parts, base_parts = [], {}, {}, {}
                for path in sorted((cache_root/name/role).glob('*.npz')):
                    date = path.stem; date_start = pd.Timestamp(date, tz='UTC')
                    target = template.loc[template_dates == date_start].copy()
                    masked = release.read(masked_paths[date])
                    raw = canonical(release.read(raw_paths[date]))
                    features = StateFeatures(masked, net, topology, model.profile, version=model.feature_version)
                    infer = target.copy(); infer['timestamp'] = infer.timestamp.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                    prediction = canonical(model.predict(features, infer))
                    values = prediction[VALUES].to_numpy(float, copy=True)
                    values[:, 0] = np.clip(values[:, 0], 0, target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                    values[:, 1] = np.clip(values[:, 1], 0, target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)
                    prediction[VALUES] = values
                    with np.load(path, allow_pickle=False) as z:
                        normalized, lane, regime = z['base'], z['target_lanes'], str(z['regime'])
                    base_values = np.column_stack([
                        np.clip(normalized[:, 0]*25, 0, target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05),
                        np.clip(normalized[:, 1]*600*lane, 0, target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)])
                    official = state_targets(raw, masked)
                    official_lanes = release.station_lanes(panel, official)
                    station_lanes = release.station_lanes(panel, target)
                    trusted = official.link_id.map(dict(zip(geometry.links, geometry.component >= 0))).fillna(False).to_numpy(bool)
                    true_anchor = raw[raw.is_score_eligible.astype(bool)].groupby(['timestamp', 'link_id'], as_index=False)[VALUES].mean()
                    true_anchor['is_target'] = 0.
                    local_ramps = ramps[(ramps.timestamp >= date_start) & (ramps.timestamp < date_start+pd.Timedelta(days=1))]
                    true_frame = physics_frame(true_anchor, panel, regime, params, topology, local_ramps).reset_index(drop=True)
                    scorer = FixedPairScorer(raw, target, official, official_lanes, trusted, true_frame, topology)
                    row = {'date': date, 'panel': panel, 'regime': regime, **scorer.statistics(values),
                           **scorer.increment(values), 'trusted_state_stats': scorer.statistics(values, trusted),
                           'increment_midnight_pairs': 0, '_boundary_endpoints': scorer.boundaries(values),
                           '_eligible_links': scorer.eligible_links}
                    truth = target[KEY].merge(raw[KEY+VALUES], on=KEY, validate='one_to_one')[VALUES].to_numpy(float)
                    delta = raw.pivot_table(index='timestamp', columns='link_id', values='speed_kmh', aggfunc='mean').diff()
                    for label, condition in [('sharp_drop', delta <= -20), ('recovery', delta >= 20)]:
                        selected = condition.stack().reindex(pd.MultiIndex.from_frame(target[['timestamp', 'link_id']])).fillna(False).to_numpy(bool)
                        error = values-truth
                        row[label] = {'n': int(selected.sum()), 'speed_sse': float(np.square(error[selected, 0]).sum()),
                                      'lane_flow_sse': float(np.square(error[selected, 1]/station_lanes[selected]).sum())}
                    rows.append(row)
                    fields = ['timestamp', 'link_id', 'accumulation_N', 'on_ramp_valid', 'off_ramp_valid']
                    true_parts.setdefault(regime, []).append(true_frame[fields])
                    for predicted, collection in [(values, lgb_parts), (base_values, base_parts)]:
                        anchored = scorer.values.copy(); anchored[scorer.target_idx] = predicted
                        part = true_frame[fields].copy()
                        part['accumulation_N'] = scorer.length*anchored[:, 1]/np.maximum(anchored[:, 0], 1.)
                        collection.setdefault(regime, []).append(part)
                    prediction_path = folder/(role+'_'+date+'.npz')
                    np.savez_compressed(prediction_path, prediction=values, target_indices=features.lookup(infer))
                    source_records.append({'role': role, 'date': date, 'prediction_sha256': sha256_file(prediction_path),
                                           'cache_sha256': sha256_file(path), 'masked_sha256': sha256_file(masked_paths[date]),
                                           'raw_sha256': sha256_file(raw_paths[date])})
                    print(name, role, date, 'fixed pureLGB replay', flush=True)
                add_midnight_pairs(rows, topology)
                metric = aggregate(rows, {panel: families[panel]})
                for label, collection, reference_rows in [('pureLGB', lgb_parts, rows),
                        ('frozen_strong', base_parts, primary['cases'][name][role]['baseline']['days'])]:
                    for regime in ['R1', 'R2', 'R3']:
                        joint = increment_proxy(pd.concat(collection[regime], ignore_index=True),
                                                pd.concat(true_parts[regime], ignore_index=True), topology)
                        subset = [r for r in reference_rows if r['regime'] == regime]
                        expected = {'increment_error_L1': sum(r['increment_error_L1'] for r in subset),
                                    'increment_truth_L1': sum(r['increment_truth_L1'] for r in subset),
                                    'n_pairs': sum(r['n_pairs'] for r in subset)}
                        differences = {k: abs(joint[k]-v) for k, v in expected.items()}
                        if any(not np.isclose(joint[k], v, rtol=1e-12, atol=1e-8) for k, v in expected.items()):
                            raise ValueError(f'Joint midnight reference mismatch {name}/{role}/{label}/{regime}: {differences}')
                        parity.append({'role': role, 'model': label, 'regime': regime, 'joint_reference': joint,
                                       'absolute_errors': differences})
                case_result[role] = metric
            case_result.update({'panel': panel, 'baseline_method': model.method, 'parent_ensemble_sha256': sha256_file(model_path),
                'nested_frozen_model_pickle_sha256_before_thread_cap': model_hash, 'source_hashes': source_records,
                'joint_official_pair_replay': parity, 'seconds': time.perf_counter()-begin_case})
            results[name] = case_result
            (folder/'experiment.json').write_text(json.dumps(case_result, indent=2, allow_nan=False), encoding='utf-8')
            print(name, 'pureLGB RESULT', case_result['assessment']['S_state'], flush=True)
    result = {'registered': registration, 'source_snapshot': snapshot, 'cases': results,
              'seconds': time.perf_counter()-started, 'runtime': evidence(gpu_used=False),
              'GP_selection_changed': False, 'formal_State': None, 'formal_Physics': None, 'formal_Total': None}
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('FIXED_REPLAY_REPORT', report_path.as_posix(), flush=True)


if __name__ == '__main__':
    try:
        import sys
        if '--replay-lightgbm' in sys.argv:
            parser = argparse.ArgumentParser()
            parser.add_argument('--replay-lightgbm', type=Path, required=True)
            parser.add_argument('--cache-root', type=Path, default=Path('artifacts/research/v2_state_v2'))
            parser.add_argument('--output', type=Path, required=True)
            replay_args = parser.parse_args()
            replay_lightgbm(replay_args.replay_lightgbm, replay_args.cache_root, replay_args.output)
        else:
            main()
    except Exception as exc:
        import datetime
        import sys
        import traceback
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        failure_path = Path('reports/v2')/('GP_failure_'+stamp+'.json')
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps({'command': sys.argv, 'type': type(exc).__name__, 'reason': str(exc),
            'traceback': traceback.format_exc(), 'source_sha256': {
                'script': sha256_file(__file__), 'kernel': sha256_file('src/trafficflow/v2/directional_kernel.py')},
            'note': 'Failure retained; no assessment-driven parameter retuning authorized by this failure log'}, indent=2), encoding='utf-8')
        raise
