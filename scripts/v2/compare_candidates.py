"""Independent exact across-date replay of already sealed State decisions.

No refitting or candidate selection. Adds midnight pairs and pools FD/LWR using
the official regime universe. Original day-local training diagnostics remain.
"""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release, VALUES
from trafficflow.experiments import partition_date
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.v2.score_aligned import (canonical, KEY, anchored_frame, network_parameters,
    ramp_flows, physics_frame, physics_score, state_statistics, state_targets, state_macro)
from trafficflow.v2.anchored_increment import increment_proxy
from trafficflow.v2.score_decision import paired_macro_bootstrap
from trafficflow.v2.gated_predictor import state_promotion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state-root', type=Path, default=Path('artifacts/research/v2_state_v2'))
    ap.add_argument('--output', type=Path, default=Path('artifacts/research/v2_exact_replay_v1'))
    args = ap.parse_args()
    report = Path('reports/v2')/(args.output.name+'.json')
    if args.output.exists() or report.exists():
        raise FileExistsError('Replay evidence is immutable')
    case_reports = sorted(args.state_root.glob('*/experiment.json'))
    if len(case_reports) != 4:
        raise ValueError('Wait for all four preregistered cases')
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    manifest = {'procedure': 'Exact full-regime replay of frozen cal decisions; no tuning or selection',
                'case_reports': {p.as_posix(): sha256_file(p) for p in case_reports},
                'coverage_correction': 'Initial training/evaluation diagnostic was date-local. This replay includes <=5.1minute cross-midnight pairs when adjacent dates belong to the same regime.',
                'choices_frozen_before_replay': True}
    (args.output/'registered.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    start = time.perf_counter()
    release = Release('data/kaggle_public')
    families = {p['corridor_id']: p['family_id'] for p in json.loads(Path('vendor/official/config/corridors.json').read_text())['panels']}
    results = {}
    for path in case_reports:
        old = json.loads(path.read_text())
        panel, case = old['panel'], path.parent.name
        if sha256_file(path.parent/'selection_sealed.json') != old['selection_sha256_before_assessment']:
            raise ValueError('Sealed selection changed')
        if sha256_file(old['base_model']) != old['base_sha256']:
            raise ValueError('Frozen base artifact changed')
        net = release.network(panel)
        params = network_parameters(release.root/'corridors'/panel)
        topology = release.topology(panel)
        ramps = ramp_flows(release.root/'corridors'/panel)
        template = canonical(release.template(1, panel, 'train'))
        raw_paths = {partition_date(release, p): p for p in release.files(panel, 'train', False)}
        masked_paths = {partition_date(release, p): p for p in release.files(panel, 'train')}
        keys = sorted(old['assessment'])
        models = {key.split('@')[0]: pickle.loads((path.parent/(key.split('@')[0]+'.pkl')).read_bytes()) for key in keys if key != 'baseline'}
        for label in models:
            if sha256_file(path.parent/(label+'.pkl')) != old['fit'][label]['sha256']:
                raise ValueError('Frozen correction artifact changed')
        result = {'selection_preserved': old['selection'], 'selection_sha256': old['selection_sha256_before_assessment'], 'parts': {}}
        for part in ['calibration', 'assessment']:
            data = []
            for cache in sorted((path.parent/part).glob('*.npz')):
                with np.load(cache, allow_pickle=False) as archive:
                    # Early immutable v2 caches have an unused object-dtype
                    # feature_columns metadata array. Never unpickle it.
                    day = {k: archive[k] for k in ['base', 'basis', 'target_lanes', 'regime']}
                date = cache.stem
                witness = next(r for r in old['source_hashes'] if r['date'] == date and r['role'] == part)
                if (sha256_file(cache) != witness['cache_sha256'] or sha256_file(raw_paths[date]) != witness['raw_sha256']
                        or sha256_file(masked_paths[date]) != witness['masked_sha256']):
                    raise ValueError('Frozen inference cache or public evaluator source changed')
                raw = canonical(release.read(raw_paths[date]))
                masked = release.read(masked_paths[date])
                begin = pd.Timestamp(date, tz='UTC')
                target = template[(template.timestamp >= begin) & (template.timestamp < begin+pd.Timedelta(days=1))]
                data.append((date, day, raw, target, state_targets(raw, masked)))
            rows = {key: [] for key in keys}
            predictions = {key: [] for key in keys}
            raw_all = pd.concat([d[2] for d in data])
            for date, day, raw, target, official in data:
                for key in keys:
                    normalized = day['base'] if key == 'baseline' else models[key.split('@')[0]].predict(day['base'], day['basis'], float(key.split('@')[1]))
                    pred = target[['panel']+KEY+['mask_regime']].copy()
                    pred[VALUES] = np.column_stack([
                        np.clip(normalized[:, 0]*25, 0, target.link_id.map(net.free_speed_kmh).to_numpy()*1.05),
                        np.clip(normalized[:, 1]*600*day['target_lanes'], 0, target.link_id.map(net.capacity_vph).to_numpy()*1.05)])
                    predictions[key].append(pred)
                    rows[key].append({'date': date, 'panel': panel, 'regime': str(day['regime']),
                        **state_statistics(official, pred, release.station_lanes(panel, official))})
            part_results = {}
            for key in keys:
                prediction = pd.concat(predictions[key])
                regime_results = {}
                for regime in ['R1', 'R2', 'R3']:
                    selected_dates = [date for date, day, *_ in data if str(day['regime']) == regime]
                    group = raw_all[raw_all.timestamp.dt.normalize().isin(pd.to_datetime(selected_dates, utc=True))]
                    target = pd.concat([d[3] for d in data if str(d[1]['regime']) == regime])
                    pred = prediction[prediction.mask_regime == regime]
                    anchor, missing = anchored_frame(group, target, pred)
                    assert missing == 0
                    true = group[group.is_score_eligible.astype(bool)].groupby(['timestamp', 'link_id'], as_index=False)[VALUES].mean()
                    true['is_target'] = 0.
                    f = physics_frame(anchor, panel, regime, params, topology, ramps)
                    truth_frame = physics_frame(true, panel, regime, params, topology, ramps)
                    proxy = increment_proxy(f, truth_frame, topology)
                    source = old[part][key]['days'] if part == 'assessment' else json.loads((path.parent/'selection_sealed.json').read_text())['calibration'][key]['days']
                    within_pairs = sum(d['n_pairs'] for d in source if d['regime'] == regime)
                    regime_results[regime] = {**proxy, 'added_cross_midnight_pairs': proxy['n_pairs']-within_pairs,
                                             'public_topology_fallback': physics_score(f, topology)}
                state = state_macro(rows[key], {panel: families[panel]})
                # Exact State must reproduce initial SSE aggregation.
                expected = old[part][key]['S_state'] if part == 'assessment' else json.loads((path.parent/'selection_sealed.json').read_text())['calibration'][key]['S_state']
                np.testing.assert_allclose(state['S_state'], expected, atol=1e-12)
                part_results[key] = {'S_state': state['S_state'], 'state_details': state['details'], 'regimes': regime_results,
                    'macro_increment_relative_L1': float(np.mean([r['increment_relative_L1'] for r in regime_results.values()])),
                    'macro_public_topology_fallback_Physics': float(np.mean([r['public_topology_fallback']['S_physics'] for r in regime_results.values()])),
                    'prediction_numeric_sha256': __import__('hashlib').sha256(prediction[VALUES].to_numpy(np.float64).tobytes()).hexdigest()}
            result['parts'][part] = part_results
            print(case, part, 'exact across-date replay passed', flush=True)
        result['max_proxy_macro_change'] = max(abs(v['macro_increment_relative_L1']-old['assessment'][k]['increment_relative_L1'])
                                               for k, v in result['parts']['assessment'].items())
        results[case] = result
    old_reports = {p.parent.name: json.loads(p.read_text()) for p in case_reports}
    base_rows = [row for c in old_reports.values() for row in c['assessment']['baseline']['days']]
    selected_rows = [row for c in old_reports.values() for row in c['assessment'][c['selection']['primary']]['days']]
    campaign_families = {c['panel']: families[c['panel']] for c in old_reports.values()}
    strata = {row['date']: name.rsplit('_', 1)[-1] for name, c in old_reports.items() for row in c['assessment']['baseline']['days']}
    campaign = paired_macro_bootstrap(base_rows, selected_rows, campaign_families, date_strata=strata)
    before, after = state_macro(base_rows, campaign_families), state_macro(selected_rows, campaign_families)
    panel_gains = {p: after['panels'][p]-before['panels'][p] for p in campaign_families}
    case_gains = {name: c['intervals'][c['selection']['primary']]['pooled_macro_gain'] for name, c in old_reports.items()}
    campaign['pass'] = bool(len(old_reports) == 4 and state_promotion(campaign['pooled_macro_gain'], campaign['ci95'], panel_gains)
                            and min(case_gains.values()) >= -.001)
    campaign['panel_gains'], campaign['case_gains'] = panel_gains, case_gains
    campaign['scope'] = 'Campaign primary fixed by original seals; stratified by season. Historical per-case gate_pass fields were only screens.'
    report.write_text(json.dumps({'manifest': manifest, 'source_snapshot': snapshot, 'cases': results, 'campaign_gate': campaign,
        'seconds': time.perf_counter()-start, 'runtime': evidence(), 'formal_Physics': None, 'formal_Total': None}, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
