"""Verify all sealed calibration arrays and old-regime point-score replay."""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.runtime import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--previous-study', type=Path, required=True)
    parser.add_argument('--failed-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite interval verification')
    current, old = [json.loads(p.read_text()) for p in (args.source, args.previous_study)]
    directory = Path('artifacts/research')/args.source.stem
    config = current['registered']['protocol']
    if config['fixed_point_settings'] != old['selection']['best_entropy']['settings']:
        raise ValueError('Frozen point setting changed')
    old_registration = Path('reports/research')/(args.failed_directory.name+'.registered.json')
    if json.loads(old_registration.read_text()) != current['registered']:
        raise ValueError('Failure repair changed generating/calibration settings')
    seal = json.loads((directory/'interval_sealed.json').read_text())
    if sha256_file(directory/'interval_sealed.json') != current['interval_sealed_sha256'] or seal['calibration'] != current['calibration']:
        raise ValueError('Interval calibration changed after sealing')
    previous_seal = json.loads((args.failed_directory/'interval_sealed.json').read_text())
    if previous_seal['calibration'] != current['calibration']:
        raise ValueError('Empirical interval calibration changed in the repair')
    for row in current['development']:
        name = f"{row['panel']}_{row['seed']}_{row['regime']}.npz"
        if sha256_file(directory/name) != row['calibration_array_sha256']:
            raise ValueError('Calibration array changed after recording')
        with np.load(directory/name) as a, np.load(args.failed_directory/name) as b:
            if not np.array_equal(a['standardized_errors'], b['standardized_errors']):
                raise ValueError('Repair changed a calibration error array')
    metrics = ['simulated_S_od', 'simulated_S_link', 'simulated_S_dev', 'simulated_S_attr', 'simulated_S_ODME', 'relative_path_L1']
    index = lambda row: (row['panel'], row['seed'], row['regime'], row['method'])
    old_rows = {index(r): r for r in old['assessment'] if r['method'] in ['ridge_0.05', 'selected_entropy']}
    new_rows = {index(r): r for r in current['assessment'] if r['regime'] in config['development_regimes']}
    if set(old_rows) != set(new_rows):
        raise ValueError('Old-regime assessment cases do not match')
    drift = max(abs(old_rows[k][metric]-new_rows[k][metric]) for k in old_rows for metric in metrics)
    if drift > 1e-10:
        raise ValueError('Uncertainty instrumentation changed point recovery scores')
    new_keys = [index(r) for r in current['assessment']]
    expected = 10*len(config['assessment_seeds'])*(len(config['development_regimes'])+len(config['new_assessment_regimes']))*2
    if len(new_keys) != expected or len(set(new_keys)) != expected or current['optimizer_failures']:
        raise ValueError('Incomplete or failed uncertainty assessment')
    result = {'source_sha256': sha256_file(args.source), 'previous_study_sha256': sha256_file(args.previous_study),
        'registered_generators_point_settings_calibration_identical_to_failed_run': True,
        'n_calibration_arrays_exact': len(current['development']), 'calibration_multiplier_exact': True,
        'old_regime_point_comparison_cases': len(old_rows), 'max_score_drift': drift,
        'complete_distinct_assessment_method_cases': expected, 'optimizer_failures': 0,
        'formal_Total': None,
        'scope': 'Numerical replay and identity/coverage audit. Old simulator truth was already inspected; no independent final or formal score claim.'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__':
    main()
