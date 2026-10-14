"""Audit whether released paths permit the registered route-choice stress.

Read public network path sets only. Reinterpret retained historical metrics
without changing the original generators, numeric results or selected models.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
from trafficflow.data import Release, csv
from trafficflow.runtime import capture_source_snapshot, sha256_file, evidence
from entropy_od_experiment import interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():
        raise FileExistsError('Never overwrite semantic audit evidence')
    folder.mkdir(parents=True)
    snapshot = capture_source_snapshot(folder/'source_snapshot.zip')
    release = Release('data/kaggle_public')
    rows = []
    for panel in release.panels():
        path = release.root/'corridors'/panel/'network/path_set.csv'
        frame = csv(path)
        sizes = frame.groupby(['origin_zone', 'destination_zone'], sort=True).size()
        if int(sizes.sum()) != len(frame) or frame.path_id.duplicated().any():
            raise ValueError('Missing OD groups or repeated path IDs')
        rows.append({'panel': panel, 'asset_path': path.as_posix(), 'asset_sha256': sha256_file(path),
            'n_paths': len(frame), 'n_OD_pairs': len(sizes),
            'n_multiple_path_OD_pairs': int((sizes > 1).sum()),
            'n_paths_in_multiple_path_OD_pairs': int(sizes[sizes > 1].sum()),
            'max_paths_per_OD_pair': int(sizes.max())})
    if any(row['n_multiple_path_OD_pairs'] for row in rows):
        raise ValueError('Current all-single-path interpretation no longer holds; manually audit before using this conclusion')
    study = json.loads(args.study.read_text())
    families = sorted({row['panel'].rsplit('_', 1)[0] for row in rows})
    genuine = ['sparse_active_paths', 'count_outliers']
    subset = [r for r in study['assessment'] if r['regime'] in genuine]
    revised = interval(subset, 'ridge_0.05', 'selected_entropy', study['registered']['protocol']['assessment_seeds'], families)
    result = {'source_snapshot': snapshot, 'study_sha256': sha256_file(args.study), 'panels': rows,
        'all_released_OD_pairs_have_one_path': True,
        'historical_label': 'route_choice_shift',
        'corrected_interpretation': 'Independent OD-demand lognormal .25 perturbation of floored public prior. Same generating family as D08 independent_paths; different deterministic RNG draws/seed offset.',
        'algebra': 'For a singleton OD group, preference/preference.sum()=1, so its lognormal 1.5 path preference cancels exactly; truth is group demand only.',
        'original_three_slot_numbers_changed': False,
        'n_genuinely_new_generating_regimes': len(genuine), 'genuine_regimes': genuine,
        'genuine_new_regime_paired_interval': revised,
        'source': evidence(), 'formal_ODME': None, 'formal_Total': None,
        'model_retuned': False, 'promotion': False,
        'limitations': 'Read-only retrospective semantic audit, not a new blind assessment. Public assets lack alternative paths for an OD pair; this says nothing about real-world or nonpublic route availability. No public_reference or hidden truth accessed.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print('all_public_OD_pairs_single_path', True, 'genuine_new_interval', revised, flush=True)


if __name__ == '__main__':
    main()
