"""Public-train masked-input/target geometry only; no reconstruction values.

Daily runs are not stitched across midnight, and cache masks aggregate links.
This audit cannot establish validation/private missingness or final accuracy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.runtime import sha256_file


def run_lengths(values):
    marker = np.diff(np.r_[False, values, False].astype(np.int8))
    return np.flatnonzero(marker == -1)-np.flatnonzero(marker == 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a missingness audit')
    panels = []
    for cache in args.cache:
        prepared = json.loads((cache/'prepared.json').read_text())
        summary = {}
        for part in ['fit', 'selection', 'assessment']:
            for path in sorted((cache/part).glob('*.npz')):
                with np.load(path, allow_pickle=False) as day:
                    x, mask, regime = day['x'], day['mask'], str(day['regime'].item())
                if x.shape[:2] != mask.shape or x.shape[-1] != 12:
                    raise ValueError('Unexpected public-input cache contract')
                group = summary.setdefault(regime, {'days': 0, 'grid_cells': 0, 'targets': 0, 'unavailable': 0,
                    'unavailable_time_runs': [], 'target_time_runs': [], 'whole_day_unavailable_links': 0,
                    'time_rows_over_half_unavailable': 0, 'time_rows': 0})
                absent = (x[..., 2] == 0) | (x[..., 3] == 0)
                group['days'] += 1
                group['grid_cells'] += mask.size
                group['targets'] += int(mask.sum())
                group['unavailable'] += int(absent.sum())
                group['whole_day_unavailable_links'] += int(absent.all(axis=0).sum())
                group['time_rows_over_half_unavailable'] += int((absent.mean(axis=1) >= .5).sum())
                group['time_rows'] += len(mask)
                for j in range(mask.shape[1]):
                    group['unavailable_time_runs'].extend(run_lengths(absent[:, j]).tolist())
                    group['target_time_runs'].extend(run_lengths(mask[:, j]).tolist())
        for group in summary.values():
            for name in ['unavailable_time_runs', 'target_time_runs']:
                values = np.asarray(group.pop(name))*5
                group[name+'_minutes'] = {'count': len(values),
                    'p50_p95_p99': np.quantile(values, [.5, .95, .99]).tolist() if len(values) else None,
                    'maximum': int(values.max()) if len(values) else None,
                    'share_runs_at_least_120min': float((values >= 120).mean()) if len(values) else None}
            group['target_grid_share'] = group['targets']/group['grid_cells']
            group['unavailable_grid_share'] = group['unavailable']/group['grid_cells']
            group['share_time_rows_over_half_unavailable'] = group['time_rows_over_half_unavailable']/group['time_rows']
        panels.append({'panel': prepared['panel'], 'cache': cache.as_posix(),
            'prepared_sha256': sha256_file(cache/'prepared.json'), 'date_partitions': prepared['dates'], 'regimes': summary})
    result = {'input_scope': 'Masked public-train cached x availability and target mask only; no y/station_truth loaded',
        'panels': panels, 'formal_State': None, 'formal_Total': None,
        'limitations': 'Only selected three autumn caches. Availability includes quality filtering, distinct from scored targets. Node-grid masks aggregate stations; runs truncated at each daily file. Counts share days, not independent samples. CTM stress patterns are not official R3 replicas.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print({p['panel']: {r: {'input_p99_min': v['unavailable_time_runs_minutes']['p50_p95_p99'][-1],
        'input_max_min': v['unavailable_time_runs_minutes']['maximum'],
        'target_p99_min': v['target_time_runs_minutes']['p50_p95_p99'][-1],
        'target_max_min': v['target_time_runs_minutes']['maximum'],
        'target_share': v['target_grid_share']} for r, v in p['regimes'].items()} for p in panels}, flush=True)


if __name__ == '__main__':
    main()
