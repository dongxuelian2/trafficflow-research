"""Frozen State models on an additional public-train time period.

No model/alpha selection, raw-label input features or final seed generation.
This is additional development evidence, not an untouched competition test.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.data import Release, VALUES
from trafficflow.state import StateModel, StateFeatures
from trafficflow.experiments import partition_date, day_template, target_truth
from trafficflow.research_eval import register_protocol, state_from_sufficient_stats, paired_block_interval
from trafficflow.runtime import capture_source_snapshot, sha256_file, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--baseline-file', type=Path, required=True)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():
        raise FileExistsError('Never overwrite a temporal replication')
    prepared = json.loads((args.cache/'prepared.json').read_text())
    old_dates = {d for values in prepared['dates'].values() for d in values}
    release = Release('data/kaggle_public')
    panel = prepared['panel']
    files = {partition_date(release, p): p for p in release.files(panel, 'train')}
    dates = sorted(d for d in files if args.start <= d <= args.end)
    if not dates or old_dates & set(dates):
        raise ValueError('Extension period overlaps original model protocol or is empty')
    paths = [args.baseline_file, *args.model]
    registered = register_protocol(args.output.with_suffix('.registered.json'), {
        'id': 'Frozen_State_additional_development_time_replication', 'panel': panel,
        'dates': dates, 'models_sha256': {p.as_posix(): sha256_file(p) for p in paths},
        'source_prepared_sha256': sha256_file(args.cache/'prepared.json'),
        'selection': 'None; all previously sealed model/alpha choices retained',
        'inputs': 'Public masked train frame only, fixed pre-fit profiles; raw train accessed by scoring only',
        'scope': 'Additional disjoint dates for these models, existing public train development data, not final or hidden score',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
    folder.mkdir(parents=True)
    snapshot = capture_source_snapshot(folder/'source_snapshot.zip')
    start = time.perf_counter()
    models = {'baseline': StateModel.load(args.baseline_file)}
    models.update({p.parent.name: StateModel.load(p) for p in args.model})
    net, topo = release.network(panel), release.topology(panel)
    template = release.template(1, panel, 'train')
    raw = {partition_date(release, p): p for p in release.files(panel, 'train', False)}
    records = {name: [] for name in models}
    sources = []
    for date in dates:
        frame = release.read(files[date])
        features = StateFeatures(frame, net, topo, models['baseline'].profile, version=models['baseline'].feature_version)
        target = day_template(release, panel, 'train', frame, template)
        # Ground truth is loaded after input features and frozen models exist.
        truth = target_truth(release, panel, target, release.read(raw[date]))
        lanes = release.station_lanes(panel, target)
        for name, model in models.items():
            p = model.predict(features, target)[VALUES].to_numpy(float, copy=True)
            p[:, 0] = np.clip(p[:, 0], 0., target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
            p[:, 1] = np.clip(p[:, 1], 0., target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)
            error_v = p[:, 0]-truth.speed_kmh.to_numpy(float)
            error_q = (p[:, 1]-truth.flow_vph.to_numpy(float))/lanes
            rv, rq = np.sqrt(np.mean(error_v**2)), np.sqrt(np.mean(error_q**2))
            records[name].append({'date': date, 'regime': str(target.mask_regime.iloc[0]), 'n': len(target),
                'speed_sse': float(np.sum(error_v**2)), 'lane_flow_sse': float(np.sum(error_q**2)),
                'daily_State': .54*max(0., 1-rv/25)+.46*max(0., 1-rq/600)})
        sources.append({'date': date, 'masked_source_sha256': sha256_file(files[date]), 'raw_train_score_source_sha256': sha256_file(raw[date])})
        print(date, 'complete', flush=True)
    summaries = {name: state_from_sufficient_stats(rows) for name, rows in records.items()}
    control = next(name for name in models if name.startswith('A49_'))
    intervals = {name: paired_block_interval([r['daily_State'] for r in records[control]], [r['daily_State'] for r in records[name]]) for name in models if name not in ['baseline', control]}
    result = {'registered': registered, 'source_snapshot': snapshot, 'daily_records': records,
        'summary': summaries, 'matched_control_day_intervals': intervals,
        'regime_day_counts': {r: sum(d['regime'] == r for d in records['baseline']) for r in ['R1', 'R2', 'R3']},
        'public_source_hashes': sources, 'seconds': time.perf_counter()-start,
        'source': evidence(gpu_used=any(getattr(m, '_network', None) is not None and next(m._network.parameters()).is_cuda for m in models.values())), 'formal_State': None, 'formal_Physics': None, 'formal_Total': None,
        'limitations': 'Additional public-train period; all architecture/cal choices unchanged. Not a final competition score or independent training seed. No mode/alpha reselected after viewing replication.'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(summaries, intervals, 'seconds', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
