"""Extend an immutable graph cache using only released masked observations."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.data import Release, VALUES, eligible
from trafficflow.state import StateModel, StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.quality_state import quality_reconstruction_input
from trafficflow.experiments import partition_date, day_template
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--baseline-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a parent or quality cache')
    parent = json.loads((args.parent/'prepared.json').read_text())
    report = Path('reports/research')/(args.output.name+'.json')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'Masked_public_quality18_cache', 'parent_prepared_sha256': sha256_file(args.parent/'prepared.json'),
        'panel': parent['panel'], 'fold': parent['fold'], 'baseline_sha256': sha256_file(args.baseline_file),
        'extension': ['any_speed/25', 'any_lane_flow/600', 'any_speed_available', 'any_flow_available', 'pct_observed/100', 'pct_observed_available'],
        'inputs': 'Masked public train frame and pre-fit historical profile only; no raw labels used to construct inputs',
        'label_handling': 'Original cache labels copied unchanged for fit/cal/assessment. No assessment metric or model choice at preparation.',
        'original12': 'Bit-exact reconstruction and preservation for every source day', 'occupancy': 'Original strict policy unchanged; no target occupancy restored',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    release = Release('data/kaggle_public')
    panel = parent['panel']
    baseline = StateModel.load(args.baseline_file)
    original = StateModel.load(args.parent/'lightgbm.pkl')
    if not original.profile.slots.equals(baseline.profile.slots) or not original.profile.fallback.equals(baseline.profile.fallback):
        raise ValueError('Strong baseline and original profile differ')
    net, topology = release.network(panel), release.topology(panel)
    files = {partition_date(release, p): p for p in release.files(panel, 'train')}
    template = release.template(1, panel, 'train')
    records, audit = [], {}
    for part in ['fit', 'selection', 'assessment']:
        (args.output/part).mkdir()
        counts = dict(rows=0, low_quality_rows=0, low_quality_finite_both=0,
            extra_speed_available=0, extra_flow_available=0, target_rows=0, target_occupancy_finite=0)
        for path in sorted((args.parent/part).glob('*.npz')):
            day = dict(np.load(path, allow_pickle=False))
            frame = release.read(files[path.stem])
            f = StateFeatures(frame, net, topology, baseline.profile, version=baseline.feature_version)
            strict = reconstruction_input(f)
            if not np.array_equal(strict, day['x']):
                raise ValueError('Original input reconstruction differs: '+path.name)
            target = day_template(release, panel, 'train', frame, template)
            if not np.array_equal(f.lookup(target), day['target_indices']):
                raise ValueError('Target ordering differs')
            joined = target[['timestamp', 'station_id', 'link_id']].merge(
                frame[['timestamp', 'station_id', 'link_id', *VALUES, 'occupancy']],
                on=['timestamp', 'station_id', 'link_id'], validate='one_to_one', how='left')
            if len(joined) != len(target) or joined[VALUES].notna().any().any():
                raise ValueError('Target numeric observations exposed')
            extended = quality_reconstruction_input(f, strict)
            day['x_control'] = day['x']
            day['x'] = extended
            if part != 'fit':
                prediction = baseline.predict(f, target)[VALUES].to_numpy(float, copy=True)
                prediction[:, 0] = np.minimum(prediction[:, 0], target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                prediction[:, 1] = np.minimum(prediction[:, 1], target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/target.link_id.map(net.lanes).to_numpy(float)
                day['comparator'] = prediction
            destination = args.output/part/path.name
            np.savez_compressed(destination, **day)
            low = frame.pct_observed.lt(75).to_numpy()
            counts['rows'] += len(frame)
            counts['low_quality_rows'] += int(low.sum())
            counts['low_quality_finite_both'] += int((low & np.isfinite(frame[VALUES].to_numpy(float)).all(axis=1)).sum())
            counts['extra_speed_available'] += int(((strict[..., 2] == 0) & (extended[..., 14] > .5)).sum())
            counts['extra_flow_available'] += int(((strict[..., 3] == 0) & (extended[..., 15] > .5)).sum())
            counts['target_rows'] += len(target)
            counts['target_occupancy_finite'] += int(np.isfinite(joined.occupancy).sum())
            records.append({'part': part, 'date': path.stem, 'parent_sha256': sha256_file(path),
                'masked_source_sha256': sha256_file(files[path.stem]), 'extended_cache_sha256': sha256_file(destination)})
        audit[part] = counts
        print(part, 'complete', counts, flush=True)
    (args.output/'graph.npz').write_bytes((args.parent/'graph.npz').read_bytes())
    info = {'panel': panel, 'fold': parent['fold'], 'seed': parent['seed'], 'dates': parent['dates'],
        'parent_cache': args.parent.as_posix(), 'baseline_file': args.baseline_file.as_posix(),
        'baseline_sha256': sha256_file(args.baseline_file), 'parent_prepared': parent,
        'channels': parent['channels']+registered['protocol']['extension'], 'records': records,
        'registered': registered, 'source_snapshot': snapshot, 'quality_audit': audit,
        'graph_sha256': sha256_file(args.output/'graph.npz'), 'seconds': time.perf_counter()-start,
        'source': evidence(), 'formal_Total': None}
    (args.output/'prepared.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    report.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print('prepared_seconds', info['seconds'], flush=True)


if __name__ == '__main__':
    main()
