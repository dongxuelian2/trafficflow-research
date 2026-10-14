"""Independent replay and scope checks for OD selection and CTM factors."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from trafficflow.data import Release, csv, SPLITS
from trafficflow.odme import operator
from trafficflow.zero_aware_od import zero_aware_inverse, prior_top_mass
from trafficflow.runtime import sha256_file
from zero_aware_od_experiment import select, synthetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite verification')
    report = Path('reports/research')
    load = lambda name: json.loads((report/name).read_text())
    b8, repeat = [load(n) for n in ['B08_contiguous_missing_CTM.json', 'B08_contiguous_missing_CTM_repeat.json']]
    for key in ['registered', 'rows', 'summary_by_pattern']:
        if b8[key] != repeat[key]:
            raise ValueError('Independent B08 replay differs: '+key)
    b9 = load('B09_CTM_historical_profile_factorial.json')
    indexed = {(r['seed'], r['scenario'], r['pattern']): r for r in b8['rows']}
    reproduced = 0
    for row in b9['rows']:
        if row['incident_hour_shift'] != 0.:
            continue
        old = indexed[(row['seed'], row['scenario'], row['pattern'])]
        mode = row['adapters']['geometry__temporal_first']
        if old['n_targets'] != row['n_targets'] or old['input_sha256'] != mode['input_sha256']:
            raise ValueError('Factorial control target/input differs from B08')
        if any(old['candidates'][m] != mode['candidates'][m] for m in old['candidates']):
            raise ValueError('New adapter options changed the retained default')
        reproduced += 1
    if reproduced != 72 or len(b9['rows']) != 144:
        raise ValueError('Incomplete factorial control')
    folder = Path('artifacts/research/B09_CTM_historical_profile_factorial')
    if sha256_file(folder/'historical_profile.npz') != b9['historical_profile_sha256'] or sha256_file(folder/'history_sources.json') != b9['history_sources_sha256']:
        raise ValueError('Historical input changed')
    config = b9['registered']['protocol']
    if set(config['profile_seeds']) & set(config['assessment_seeds']):
        raise ValueError('Historical profile leaked assessment seeds')
    od = load('D12_zero_aware_OD_nine_regimes.json')
    directory = Path('artifacts/research/D12_zero_aware_OD_nine_regimes')
    seal = json.loads((directory/'selection_sealed.json').read_text())
    if seal != od['selection'] or select(od['development']) != seal:
        raise ValueError('OD selection changed or does not reproduce from development only')
    if json.loads((directory/'OD_selector.json').read_text()) != seal['deployment_config']:
        raise ValueError('OD deployment config changed')
    previous = load('D10_entropy_OD_uncertainty_mismatch_repaired.json')
    previous_rows = {(r['panel'], r['seed'], r['regime'], r['method']): r for r in previous['assessment']}
    metrics = ['simulated_S_od', 'simulated_S_link', 'simulated_S_dev', 'simulated_S_attr', 'simulated_S_ODME', 'relative_path_L1']
    drift, matched, inactive_exact = 0., 0, 0
    ridge = {(r['panel'], r['seed'], r['regime']): r for r in od['assessment'] if r['method'] == 'ridge_0.05'}
    for row in od['assessment']:
        if row['method'] in ['ridge_0.05', 'fixed_entropy_D08']:
            method = 'selected_entropy' if row['method'] == 'fixed_entropy_D08' else row['method']
            prior = previous_rows[(row['panel'], row['seed'], row['regime'], method)]
            drift = max(drift, max(abs(row[m]-prior[m]) for m in metrics))
            matched += 1
        if row['method'] == 'development_winner' and not row['selector_applied']:
            if row['prediction_sha256'] != ridge[(row['panel'], row['seed'], row['regime'])]['prediction_sha256']:
                raise ValueError('Inactive selector changes Ridge predictions')
            inactive_exact += 1
    if matched != 540 or drift > 1e-12 or od['optimizer_failures']:
        raise ValueError('OD old-case replay or full optimizer success failed')
    release = Release('data/kaggle_public')
    # One full released operator is checked against a separate path-space LSQR/TRF solve.
    panel, seed, regime = 'D12_I405_N', 2909, 'sparse_active_paths'
    A, _, b, paths = operator(release, panel, 'train')
    truth, c, b = synthetic(A, b, paths, seed, regime)
    settings = seal['deployment_config']['settings']
    f, info = zero_aware_inverse(A, c, b, **settings)
    sd = settings['path_cv']*(np.maximum(b, 2.) if settings['scale_mode'] == 'relative' else np.full_like(b, max(b.mean(), 2.)))
    count_sd = settings['noise_cv']*np.maximum(c, 10.)
    if settings['huber_delta'] is not None:
        raise ValueError('Full quadratic reference only applies to selected Gaussian setting')
    design = A.multiply(sd).multiply((1/count_sd)[:, None]).tocsr()
    augmented = sparse.vstack([design, sparse.eye(len(b))], format='csr')
    target = np.r_[c/count_sd, b/sd-settings['shrinkage']]
    reference = lsq_linear(augmented, target, bounds=(0, np.inf), tol=1e-11, lsmr_tol=1e-12, max_iter=2000)
    difference = float(np.max(np.abs(f-sd*reference.x)))
    if not reference.success or difference > 1e-3:
        raise ValueError('Independent full path-space reference differs')
    visible = []
    for p in release.panels():
        for split in SPLITS:
            path = release.root/'task4'/p/split/'synthetic_weak_prior.csv'
            frame = csv(path)
            mass = prior_top_mass(frame.path_flow.to_numpy(float))
            visible.append({'panel': p, 'split': split, 'asset_sha256': sha256_file(path),
                'prior_top15_mass': mass, 'frozen_selector_would_apply': mass >= seal['deployment_config']['threshold']})
    result = {'B08_independent_registered_rows_summary_exact': True, 'B08_cases': 72,
        'B09_retained_default_control_cases_exact': reproduced, 'B09_factorial_cases': 144,
        'B09_history_assessment_seed_disjoint': True,
        'OD_development_choice_recomputed_exact': True, 'OD_old_case_method_replays': matched,
        'OD_max_old_score_drift': drift, 'OD_inactive_selector_prediction_SHA_matches': inactive_exact,
        'OD_complete_assessment_method_cases': len(od['assessment']), 'optimizer_failures': 0,
        'full_operator_independent_reference': {'panel': panel, 'n_paths': len(b), 'method': 'scipy.lsq_linear augmented path-space TRF/LSMR',
            'success': bool(reference.success), 'prediction_max_absolute_difference': difference,
            'scaled_MAP_KKT_max_abs': info['scaled_projected_KKT_max_abs']},
        'visible_prior_feature_audit': visible,
        'n_released_split_panels_triggering_selector': sum(r['frozen_selector_would_apply'] for r in visible),
        'source_report_sha256': {n: sha256_file(report/n) for n in ['B08_contiguous_missing_CTM.json', 'B09_CTM_historical_profile_factorial.json', 'D12_zero_aware_OD_nine_regimes.json']},
        'formal_Total': None, 'uploaded': False,
        'limitations': 'Public weak-prior features only, no hidden answers. Prior-trigger audit does not measure OD truth. One full-operator independent numeric reference, not a new generalization test. Original formal submission preserved.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print({k: v for k, v in result.items() if k not in ['visible_prior_feature_audit', 'source_report_sha256']}, flush=True)


if __name__ == '__main__':
    main()
