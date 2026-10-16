"""Frozen strong-State, low-rank score corrections; no assessment selection."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from trafficflow.data import Release, VALUES
from trafficflow.experiments import partition_date
from trafficflow.state import StateModel, StateFeatures
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import register_protocol, canonical_hash
from trafficflow.v2.score_aligned import (canonical, KEY, anchored_frame, network_parameters,
    ramp_flows, physics_frame, physics_score, conservation_pairs, state_targets, state_statistics, state_macro)
from trafficflow.v2.anchored_increment import increment_proxy
from trafficflow.v2.score_decision import LowRankCorrection, paired_macro_bootstrap
from trafficflow.v2.gated_predictor import state_promotion


CASES = [
    ('I5_N_autumn', 'D12_I5_N', 'artifacts/research/A33_adaptive_attention_strong_autumn/State_selected.pkl', '2030-10-30', '2030-11-12', '2030-11-26', '2030-12-10'),
    ('I210_W_autumn', 'D7_I210_W', 'artifacts/research/A49_adaptive_attention_D7_I210_W/State_selected.pkl', '2030-10-30', '2030-11-12', '2030-11-26', '2030-12-10'),
    ('I5_N_winter', 'D12_I5_N', 'artifacts/research/A34_adaptive_attention_strong_winter/State_selected.pkl', '2030-12-30', '2031-01-12', '2031-01-26', '2031-02-09'),
    ('I210_W_winter', 'D7_I210_W', 'artifacts/research/A49_adaptive_attention_D7_I210_W/State_selected.pkl', '2030-12-30', '2031-01-12', '2031-01-26', '2031-02-09'),
]
OPTIONS = [('linear_state', 0., 0.), ('sparse_state', 0., 1e-4), ('sparse_increment_002', .02, 1e-4), ('sparse_increment_010', .10, 1e-4)]


def basis(features, target, base):
    idx = features.lookup(target)
    f = features.features.iloc[idx]
    def col(name, fallback=0.):
        return np.nan_to_num(f[name].to_numpy(float), nan=fallback, posinf=fallback, neginf=fallback)
    values = [np.ones(len(target)), col('v_temporal')-base[:, 0], col('v_spatial')-base[:, 0],
              col('q_temporal')-base[:, 1], col('q_spatial')-base[:, 1],
              col('v_prior')-base[:, 0], col('q_prior')-base[:, 1],
              np.minimum(col('v_age'), 2.), np.minimum(col('q_age'), 2.),
              (base[:, 0]*25 < target.link_id.map(features.net.free_speed_kmh).to_numpy(float)*.6).astype(float),
              col('sin_tod'), col('cos_tod')]
    return np.clip(np.column_stack(values), -3., 3.)


def linear_increment_design(anchor, truth_frame, params, topology, ramps, panel, regime, target, u, lane):
    """Derivative of official station-average N wrt the 24 coefficients."""
    frame = physics_frame(anchor, panel, regime, params, topology, ramps)
    true = physics_frame(truth_frame, panel, regime, params, topology, ramps)
    pairs = conservation_pairs(frame, topology)
    full_index = pd.MultiIndex.from_frame(frame[['timestamp', 'link_id']])
    target_index = full_index.get_indexer(pd.MultiIndex.from_frame(target[['timestamp', 'link_id']]))
    if (target_index < 0).any():
        raise ValueError('Target missing in official anchored field')
    # Current release has one station per sensor link. Reject unsupported station
    # multiplicity here; the independent scorer itself fully supports aggregation.
    if target.duplicated(['timestamp', 'link_id']).any():
        raise ValueError('Low-rank pilot requires unique target station/link')
    k = u.shape[1]
    jac = np.zeros((len(frame), k, 2))
    v, q, length = frame.speed_kmh.to_numpy(float), frame.flow_vph.to_numpy(float), frame.length_km.to_numpy(float)
    dv = -length*q/np.maximum(v, 1.)**2*(v > 1)*25
    dq = length/np.maximum(v, 1.)
    jac[target_index, :, 0] = dv[target_index, None]*u
    jac[target_index, :, 1] = dq[target_index, None]*600*lane[:, None]*u
    # The pilot also checks raw station uniqueness before reaching this function.
    left = full_index.get_indexer(pd.MultiIndex.from_frame(pairs[['timestamp', 'link_id']]))
    right_keys = pairs[['next_timestamp', 'link_id']].rename(columns={'next_timestamp': 'timestamp'})
    right = full_index.get_indexer(pd.MultiIndex.from_frame(right_keys))
    true_index = pd.MultiIndex.from_frame(true[['timestamp', 'link_id']])
    tl = true_index.get_indexer(pd.MultiIndex.from_frame(pairs[['timestamp', 'link_id']]))
    tr = true_index.get_indexer(pd.MultiIndex.from_frame(right_keys))
    if min(np.min(tl, initial=0), np.min(tr, initial=0)) < 0:
        raise ValueError('Truth/predicted endpoint mismatch')
    dn_true = true.accumulation_N.to_numpy()[tr]-true.accumulation_N.to_numpy()[tl]
    return (jac[right]-jac[left]).reshape(len(pairs), 2*k), pairs.dN.to_numpy()-dn_true, float(np.abs(dn_true).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('artifacts/research/v2_state_v1'))
    ap.add_argument('--case', action='append')
    args = ap.parse_args()
    report = Path('reports/v2')/(args.output.name+'.json')
    if args.output.exists() or report.exists():
        raise FileExistsError('Use a new experiment output')
    cases = [c for c in CASES if args.case is None or c[0] in args.case]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'V2_frozen_score_low_rank', 'code_base_commit': commit, 'cases': cases,
        'base_model_sha256': {c[0]: sha256_file(c[2]) for c in cases}, 'seed': 20261016,
        'roles': 'Base fit/checkpoint <=Oct29 or Dec29; next14 dates correction_fit, next14 calibration, next14 assessment. I210 winter transfers frozen autumn base.',
        'residuals': 'Single chronological forward cross-fit; correction-fit labels were excluded from base fitting AND checkpoint selection. Not in-sample base residuals or a multi-fold CV claim.',
        'basis': '12 clipped lawful features shared across all cells; 24 coefficients; frozen base; no truth residual as inference input',
        'options': OPTIONS, 'ridge': .001, 'coefficient_bound': .25, 'strengths': [0., .25, .5, 1.],
        'primary': 'Highest calibration official State, tie baseline then smaller strength',
        'joint': 'Smallest cal observed full-coverage inventory-increment L1; State>=base AND >=primary-.001',
        'scope': 'Complete targets on all selected dates. State official RMSE/regime/family hierarchy on 2/10 directions; not a ten-direction competition score.',
        'assessment': 'Previously used public development calendar; no hidden labels, no assessment parameter tuning',
        'bootstrap': '2000 circular paired calendar-date blocks of7, recompute nonlinear pooled regime macro; exclude draws missing regime',
        'gate': 'State gain>=.002, pooled 95%lower>0, every direction gain>=-.001, both seasons; proxy alone cannot promote',
        'formal_Total': None, 'formal_Physics': None})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    started = time.perf_counter()
    import torch
    torch.cuda.reset_peak_memory_stats()
    release = Release('data/kaggle_public')
    modes = json.loads(Path('vendor/official/config/task3_lwr_modes.json').read_text())
    family_map = {p['corridor_id']: p['family_id'] for p in json.loads(Path('vendor/official/config/corridors.json').read_text())['panels']}
    results = {}
    for name, panel, model_path, first, fit_end, cal_end, assess_end in cases:
        case_start = time.perf_counter()
        folder = args.output/name
        folder.mkdir()
        model = StateModel.load(Path(model_path))
        net, topology = release.network(panel), release.topology(panel)
        params = network_parameters(release.root/'corridors'/panel)
        ramps = ramp_flows(release.root/'corridors'/panel)
        template = canonical(release.template(1, panel, 'train'))
        masked_paths = {partition_date(release, p): p for p in release.files(panel, 'train')}
        raw_paths = {partition_date(release, p): p for p in release.files(panel, 'train', False)}
        sources = []

        def prepare(role, lower, upper):
            days = []
            out = folder/role; out.mkdir()
            for date in sorted(d for d in masked_paths if lower <= d <= upper):
                raw = canonical(release.read(raw_paths[date]))
                masked = release.read(masked_paths[date])
                if raw.duplicated(['timestamp', 'link_id']).any():
                    raise ValueError('Pilot derivative requires single-station links')
                begin = pd.Timestamp(date, tz='UTC')
                end = begin+pd.Timedelta(days=1)
                target = template[(template.timestamp >= begin) & (template.timestamp < end)].copy()
                features = StateFeatures(masked, net, topology, model.profile, version=model.feature_version)
                infer_target = target.copy(); infer_target['timestamp'] = infer_target.timestamp.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                prediction = canonical(model.predict(features, infer_target))
                lanes = target.link_id.map(net.lanes).to_numpy(float)
                base = prediction[VALUES].to_numpy(float)/np.column_stack([np.full(len(target), 25.), 600*lanes])
                u = basis(features, infer_target, base)
                truth = target[KEY].merge(raw[KEY+VALUES], on=KEY, validate='one_to_one')[VALUES].to_numpy(float)
                normalized_truth = truth/np.column_stack([np.full(len(target), 25.), 600*lanes])
                regime = str(target.mask_regime.iloc[0])
                day_raw = raw[raw.is_score_eligible.astype(bool)]
                anchor, missing = anchored_frame(raw, target, prediction)
                assert missing == 0
                true_anchor = day_raw.groupby(['timestamp', 'link_id'], as_index=False)[VALUES].mean()
                true_anchor['is_target'] = anchor.is_target.to_numpy()
                local_ramps = ramps[(ramps.timestamp >= begin) & (ramps.timestamp < end)] if len(ramps) else ramps
                design, residual, denominator = linear_increment_design(anchor, true_anchor, params, topology, local_ramps, panel, regime, target, u, lanes)
                official_target = state_targets(raw, masked)
                d = dict(date=date, regime=regime, base=base, basis=u, truth=normalized_truth,
                    increment_design=design, increment_residual=residual, increment_denominator=denominator,
                    target=target, official_target=official_target, prediction=prediction, raw=raw,
                    true_anchor=true_anchor, ramps=local_ramps, lanes=lanes,
                    station_lanes=release.station_lanes(panel, target),
                    official_lanes=release.station_lanes(panel, official_target))
                # Explicitly labeled artifacts: inference basis/base separate from scorer labels.
                cache = out/(date+'.npz')
                np.savez_compressed(cache, base=base, basis=u, target_truth=normalized_truth,
                    increment_design=design, increment_residual=residual, increment_denominator=denominator,
                    target_indices=features.lookup(infer_target), times=np.asarray(features.times), links=np.asarray(features.links),
                    regime=regime, legal_observed=features.raw, legal_features=features.features.to_numpy(np.float32),
                    feature_columns=np.asarray(features.features.columns, dtype=str), target_lanes=lanes)
                sources.append({'date': date, 'role': role, 'cache_sha256': sha256_file(cache),
                                'raw_sha256': sha256_file(raw_paths[date]), 'masked_sha256': sha256_file(masked_paths[date])})
                days.append(d)
                print(name, role, date, 'prepared', flush=True)
            if len(days) != 14:
                raise ValueError(f'Expected14 dates for {role}, found{len(days)}')
            return days

        def evaluate(days, candidates):
            records = {key: [] for key in candidates}
            for d in days:
                for key, (corrector, strength) in candidates.items():
                    normalized = d['base'] if corrector is None else corrector.predict(d['base'], d['basis'], strength)
                    prediction = d['prediction'].copy()
                    v = np.clip(normalized[:, 0]*25, 0, d['target'].link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                    q = np.clip(normalized[:, 1]*600*d['lanes'], 0, d['target'].link_id.map(net.capacity_vph).to_numpy(float)*1.05)
                    prediction[VALUES] = np.column_stack([v, q])
                    anchor, missing = anchored_frame(d['raw'], d['target'], prediction)
                    if missing:
                        raise ValueError('Candidate lost published predictions')
                    frame = physics_frame(anchor, panel, d['regime'], params, topology, d['ramps'])
                    true_frame = physics_frame(d['true_anchor'], panel, d['regime'], params, topology, d['ramps'])
                    stats = state_statistics(d['official_target'], prediction, d['official_lanes'])
                    published = state_statistics(d['target'].merge(d['raw'][KEY+VALUES], on=KEY, suffixes=('_template', '')),
                        prediction, d['station_lanes'])
                    row = {'date': d['date'], 'panel': panel, 'regime': d['regime'], **stats,
                           'published_state_stats': published, **increment_proxy(frame, true_frame, topology),
                           'public_topology_fallback': physics_score(frame, topology)}
                    # Evaluator-only condition labels, derived after prediction.
                    truth_grid = d['raw'].pivot_table(index='timestamp', columns='link_id', values='speed_kmh', aggfunc='mean')
                    delta = truth_grid.diff()
                    for label, condition in [('sharp_drop', delta <= -20), ('recovery', delta >= 20)]:
                        select = condition.stack().reindex(pd.MultiIndex.from_frame(d['target'][['timestamp', 'link_id']])).fillna(False).to_numpy(bool)
                        errors = prediction[VALUES].to_numpy(float)-d['truth']*np.column_stack([np.full(len(prediction), 25.), 600*d['lanes']])
                        row[label] = {'n': int(select.sum()), 'speed_sse': float(np.square(errors[select, 0]).sum()),
                                      'lane_flow_sse': float(np.square(errors[select, 1]/d['station_lanes'][select]).sum())}
                    records[key].append(row)
            metrics = {}
            for key, rows in records.items():
                metric = state_macro(rows, {panel: family_map[panel]})
                metric['published_template_State'] = state_macro([{**z, **z['published_state_stats']} for z in rows], {panel: family_map[panel]})['S_state']
                metric['increment_relative_L1'] = float(np.mean([sum(z['increment_error_L1'] for z in rows if z['regime'] == g)/
                    (sum(z['increment_truth_L1'] for z in rows if z['regime'] == g)+1e-9) for g in ['R1', 'R2', 'R3']]))
                metric['days'] = rows
                metrics[key] = metric
            return metrics

        with threadpool_limits(limits=1):
            fit_days = prepare('correction_fit', first, fit_end)
            models = {}
            fit_records = {}
            for label, weight, l1 in OPTIONS:
                fitted = LowRankCorrection(weight, l1).fit(fit_days)
                path = folder/(label+'.pkl')
                path.write_bytes(pickle.dumps(fitted))
                fit_records[label] = {'sha256': sha256_file(path), **fitted.fit_record, 'coefficients': fitted.coefficient.tolist()}
                models[label] = fitted
            del fit_days
            # cal loading occurs only after correction fitting is complete.
            next_fit = str((pd.Timestamp(fit_end)+pd.Timedelta(days=1)).date())
            calibration_days = prepare('calibration', next_fit, cal_end)
            candidates = {'baseline': (None, 0.)}
            candidates.update({f'{label}@{s}': (m, s) for label, m in models.items() for s in [.25, .5, 1.]})
            cal = evaluate(calibration_days, candidates)
            primary = max(cal, key=lambda k: (cal[k]['S_state'], -candidates[k][1]))
            eligible = [k for k in cal if cal[k]['S_state'] >= cal['baseline']['S_state'] and cal[k]['S_state'] >= cal[primary]['S_state']-.001]
            joint = min(eligible, key=lambda k: (cal[k]['increment_relative_L1'], candidates[k][1]))
            # Every preregistered fixed configuration is reported, with cal-selected
            # strengths; no assessment-informed choice among unsuccessful variants.
            matched = {label: max(['baseline']+[k for k in cal if k.startswith(label+'@')], key=lambda k: (cal[k]['S_state'], -candidates[k][1])) for label, _, _ in OPTIONS}
            seal = {'primary': primary, 'joint': joint, 'matched': matched, 'calibration': cal}
            (folder/'selection_sealed.json').write_text(json.dumps(seal, indent=2, allow_nan=False), encoding='utf-8')
            seal_sha = sha256_file(folder/'selection_sealed.json')
            del calibration_days
            next_cal = str((pd.Timestamp(cal_end)+pd.Timedelta(days=1)).date())
            assessment_days = prepare('assessment', next_cal, assess_end)
            chosen = {k: candidates[k] for k in dict.fromkeys(['baseline', primary, joint, *matched.values()])}
            assess = evaluate(assessment_days, chosen)
            intervals = {key: paired_macro_bootstrap(assess['baseline']['days'], value['days'], {panel: family_map[panel]}) for key, value in assess.items()}
            for key, value in assess.items():
                gain = intervals[key]['pooled_macro_gain']
                value['case_gate_pass'] = state_promotion(gain, intervals[key]['ci95'], {panel: gain})
        result = {'panel': panel, 'base_model': model_path, 'base_sha256': sha256_file(model_path),
                  'source_hashes': sources, 'fit': fit_records, 'selection': {'primary': primary, 'joint': joint, 'matched': matched},
                  'selection_sha256_before_assessment': seal_sha, 'calibration': cal, 'assessment': assess, 'intervals': intervals,
                  'seconds': time.perf_counter()-case_start, 'runtime': evidence(gpu_used=True)}
        results[name] = result
        (folder/'experiment.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        print(name, 'RESULT', {k: z['S_state'] for k, z in assess.items()}, 'selected', primary, joint, flush=True)
        del assessment_days, model
        torch.cuda.empty_cache()
    base_rows = [row for case in results.values() for row in case['assessment']['baseline']['days']]
    selected_rows = [row for case in results.values() for row in case['assessment'][case['selection']['primary']]['days']]
    campaign_families = {case['panel']: family_map[case['panel']] for case in results.values()}
    strata = {row['date']: name.rsplit('_', 1)[-1] for name, case in results.items() for row in case['assessment']['baseline']['days']}
    campaign = paired_macro_bootstrap(base_rows, selected_rows, campaign_families, date_strata=strata)
    before, after = state_macro(base_rows, campaign_families), state_macro(selected_rows, campaign_families)
    panel_gains = {p: after['panels'][p]-before['panels'][p] for p in campaign_families}
    case_gains = {name: case['intervals'][case['selection']['primary']]['pooled_macro_gain'] for name, case in results.items()}
    campaign['pass'] = bool(len(results) == 4 and state_promotion(campaign['pooled_macro_gain'], campaign['ci95'], panel_gains)
                            and min(case_gains.values()) >= -.001)
    campaign['panel_gains'], campaign['case_gains'] = panel_gains, case_gains
    campaign['scope'] = 'Two-direction/two-season campaign gate; single-case screens never deploy a model'
    result = {'registered': registered, 'source_snapshot': snapshot, 'cases': results, 'campaign_gate': campaign,
              'seconds': time.perf_counter()-started, 'runtime': evidence(gpu_used=True),
              'GPU_peak_allocated_MiB': torch.cuda.max_memory_allocated()/1024**2,
              'GPU_peak_reserved_MiB': torch.cuda.max_memory_reserved()/1024**2,
              'formal_State': None, 'formal_Physics': None, 'formal_Total': None,
              'interpretation': 'Public development evidence, observed increment proxy, no hidden boundary equality assumed; not Kaggle score'}
    report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('complete', report, 'seconds', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
