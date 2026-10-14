"""Matched State errors on public-observation speed-change scenarios.

Scenario labels are computed after inference and never enter model inputs.
Station/link-specific five-minute differences avoid mixing colocated stations.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release, VALUES, eligible
from trafficflow.state import StateModel, StateFeatures
from trafficflow.experiments import partition_date, day_template, target_truth, raw_for
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import register_protocol, paired_block_interval, state_from_sufficient_stats
from train_graph_experiment import stats


def pooled(rows):
    n = sum(row['n'] for row in rows)
    if not n:
        return None
    rv = np.sqrt(sum(row['speed_sse'] for row in rows)/n)
    rq = np.sqrt(sum(row['lane_flow_sse'] for row in rows)/n)
    return {'n': n, 'rmse_speed_kmh': float(rv), 'rmse_station_lane_flow': float(rq),
        'channel_score_diagnostic': float(.54*max(0, 1-rv/25)+.46*max(0, 1-rq/600))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite scenario diagnostics')
    report = Path('reports/research')/(args.output.name+'.json')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'Adaptive_State_public_observation_scenario_diagnostics',
        'studies': [p.as_posix() for p in args.directory],
        'scenario_source': 'Eligible public train observations; not latent incident annotations',
        'speed_difference': 'Previous eligible value for the same station AND link, exactly 300 seconds apart',
        'stratum_metric': 'Pool cells across regimes in each stratum; channel score is diagnostic, not official macro State',
        'choice_policy': 'Use each original calibration-sealed winner, no scenario-driven model reselection',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    release = Release('data/kaggle_public')
    result = {'registered': registered, 'source_snapshot': snapshot, 'studies': {}}
    for directory in args.directory:
        original = json.loads((directory/'experiment.json').read_text())
        config = original['registered']['protocol']
        deployment = json.loads((directory/'deployment_verification.json').read_text())
        selected_path = directory/'State_selected.pkl'
        baseline_path = Path(config['baseline_file'])
        if sha256_file(selected_path) != deployment['model_sha256'] or sha256_file(baseline_path) != config['baseline_sha256']:
            raise ValueError('Sealed models changed')
        selected = StateModel.load(selected_path)
        baseline = StateModel.load(baseline_path)
        panel = config['panel']
        net, topo = release.network(panel), release.topology(panel)
        paths = {partition_date(release, p): p for p in release.files(panel, 'train')}
        template = release.template(1, panel, 'train')
        strata = {}
        full = {'baseline': [], 'selected': []}
        for cached in sorted((Path(config['cache'])/'assessment').glob('*.npz')):
            day = dict(np.load(cached, allow_pickle=False))
            path = paths[cached.stem]
            f = StateFeatures(release.read(path), net, topo, baseline.profile, version=baseline.feature_version)
            target = day_template(release, panel, 'train', f.frame, template)
            if not np.array_equal(f.lookup(target), day['target_indices']):
                raise ValueError('Target ordering changed')
            lane = target.link_id.map(net.lanes).to_numpy(float)
            pred = {}
            for name, model in (('baseline', baseline), ('selected', selected)):
                value = model.predict(f, target)[VALUES].to_numpy(float, copy=True)
                value[:, 0] = np.minimum(value[:, 0], target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                value[:, 1] = np.minimum(value[:, 1], target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/lane
                pred[name] = value
                full[name].append(stats(day, value))
            # Public train truth is accessed only after both predictions are made.
            raw = next(raw_for(release, panel, [path]))
            truth = target_truth(release, panel, target, raw)
            np.testing.assert_allclose(truth[VALUES].to_numpy(float)/np.column_stack([np.ones(len(lane)), lane]), day['station_truth'])
            raw = raw.sort_values(['station_id', 'link_id', 'timestamp']).copy()
            raw.loc[~eligible(raw), 'speed_kmh'] = np.nan
            grouped = raw.groupby(['station_id', 'link_id'], sort=False)
            raw['drop_kmh'] = grouped.speed_kmh.shift()-raw.speed_kmh
            gap = pd.to_datetime(raw.timestamp, utc=True).groupby([raw.station_id, raw.link_id]).diff().dt.total_seconds()
            raw.loc[gap.ne(300), 'drop_kmh'] = np.nan
            delta = target[['timestamp', 'station_id', 'link_id']].merge(
                raw[['timestamp', 'station_id', 'link_id', 'drop_kmh']],
                on=['timestamp', 'station_id', 'link_id'], how='left', validate='one_to_one', sort=False).drop_kmh.to_numpy(float)
            ratio = truth.speed_kmh.to_numpy(float)/target.link_id.map(net.free_speed_kmh).to_numpy(float)
            groups = {'rapid_drop_15_to_30': (delta >= 15)&(delta < 30),
                'severe_drop_at_least_30': delta >= 30, 'recovery_at_least_15': delta <= -15,
                'stable_change_below_5': np.abs(delta) < 5, 'previous_observation_unavailable': ~np.isfinite(delta),
                'congested_observation_proxy': ratio <= .6, 'free_observation_proxy': ratio > .6}
            for key, mask in groups.items():
                bucket = strata.setdefault(key, {'baseline': [], 'selected': []})
                if not mask.any():
                    continue
                subset = {**day, 'station_truth': day['station_truth'][mask],
                    'station_lanes': day['station_lanes'][mask], 'target_link_lanes': day['target_link_lanes'][mask]}
                for name in pred:
                    bucket[name].append({**stats(subset, pred[name][mask]), 'date': cached.stem})
        summary = {}
        for key, rows in strata.items():
            summary[key] = {name: pooled(values) for name, values in rows.items()}
            if len(rows['baseline']) >= 2:
                summary[key]['paired_day_interval'] = paired_block_interval(
                    [r['daily_State'] for r in rows['baseline']], [r['daily_State'] for r in rows['selected']])
            summary[key]['days_with_targets'] = len(rows['baseline'])
        score = {name: state_from_sufficient_stats(rows) for name, rows in full.items()}
        if abs(score['selected']['S_state']-original['selected_assessment']['S_state']) > 1e-7:
            raise ValueError('Scenario replay differs from the sealed assessment')
        result['studies'][directory.name] = {'panel': panel, 'fold': config['fold'],
            'model_sha256': deployment['model_sha256'], 'overall_regime_macro_State': score, 'strata': summary}
        print(directory.name, {key: (row['baseline']['n'] if row['baseline'] else 0,
            row['selected']['channel_score_diagnostic']-row['baseline']['channel_score_diagnostic'] if row['baseline'] else None)
            for key, row in summary.items()}, flush=True)
        del selected, baseline
    result.update(seconds=time.perf_counter()-start, source=evidence(gpu_used=True), formal_State=None,
        formal_Physics=None, formal_Total=None,
        limitations='Public-observation strata are diagnostic, not true incident labels. Empty days are excluded from stratum day intervals. Existing development dates and model choices are reused; no untouched final generalization claim.')
    (args.output/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
