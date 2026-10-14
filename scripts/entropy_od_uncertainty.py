"""Frozen entropy point estimates, curvature coverage and new OD mismatch.

Seven previously used distributions calibrate one global empirical width;
assessment uses separate seeds and three additional fixed mismatch regimes.
Final seeds remain unopened. Real hidden OD/attraction truth is never accessed.

Historical route_choice_shift labels are preserved for exact D09/D10 replay.
D11 later found one released path per OD pair in all panels: that slot is an
independent OD-demand perturbation, not a route-choice experiment. See the
separate immutable path-structure audit; never claim alternative-route evidence.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.data import Release
from trafficflow.odme import operator, solve
from trafficflow.entropy_od import entropy_inverse
from trafficflow.od_uncertainty import entropy_curvature_sd, calibrate_interval_scale
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot, evidence, sha256_file
from entropy_od_experiment import generated, REGIMES, interval
from od_probability_experiment import scores


NEW_REGIMES = ['route_choice_shift', 'sparse_active_paths', 'count_outliers']


def mismatch(A, public_prior, paths, seed, regime):
    rng = np.random.default_rng(seed)
    scale = max(float(np.mean(public_prior)), 2.)
    if regime == 'route_choice_shift':
        b = public_prior.copy()
        truth = np.empty(len(b))
        for indices in paths.groupby(['origin_zone', 'destination_zone'], sort=True).indices.values():
            reference = np.maximum(b[indices], 2.)
            demand = reference.sum()*rng.lognormal(-.5*.25**2, .25)
            preference = reference*rng.lognormal(-.5*1.5**2, 1.5, len(indices))
            truth[indices] = demand*preference/preference.sum()
    else:
        truth = rng.gamma(.7, scale/.7, A.shape[1])
        if regime == 'sparse_active_paths':
            truth *= (rng.random(len(truth)) < .15)/.15
            b = truth*rng.lognormal(-.5*.4**2, .4, len(truth))+.25*scale*rng.exponential(1., len(truth))
        elif regime == 'count_outliers':
            b = np.maximum(0., truth+rng.normal(0., .75*scale, len(truth)))
        else:
            raise ValueError(regime)
    clean = np.asarray(A@truth)
    counts = np.maximum(0., clean+rng.normal(0., .02*np.maximum(clean, 10.)))
    if regime == 'count_outliers':
        corrupt = rng.random(len(counts)) < .05
        counts[corrupt] *= rng.uniform(.5, 1.5, corrupt.sum())
    return truth, counts, b


def coverage_record(truth, f, sd, multiplier):
    half_width = 1.96*sd
    def measured(scale):
        lower, upper = np.maximum(0., f-scale*half_width), f+scale*half_width
        return {'case_path_coverage': float(((truth >= lower) & (truth <= upper)).mean()),
            'mean_interval_width': float((upper-lower).mean()),
            'mean_interval_width_over_mean_true_flow': float((upper-lower).sum()/max(truth.sum(), 1e-9)),
            'simultaneous_all_paths_covered': bool(((truth >= lower) & (truth <= upper)).all())}
    return {'raw_curvature_interval': measured(1.), 'empirical_development_interval': measured(multiplier)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path('artifacts/research')/args.output.stem
    if args.output.exists() or directory.exists():
        raise FileExistsError('Never overwrite uncertainty evidence')
    frozen = json.loads(args.frozen_study.read_text())
    if frozen['selection']['winner'] != 'selected_entropy' or frozen['optimizer_failures']:
        raise ValueError('Require a numerically verified frozen entropy setting')
    settings = frozen['selection']['best_entropy']['settings']
    protocol = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())['protocol']
    registered = register_protocol(args.output.with_suffix('.registered.json'), {
        'id': 'Frozen_entropy_OD_curvature_and_new_mismatch', 'frozen_study_sha256': sha256_file(args.frozen_study),
        'fixed_point_settings': settings, 'development_regimes': REGIMES,
        'new_assessment_regimes': NEW_REGIMES, 'development_seeds': protocol['simulator_development_seeds'],
        'assessment_seeds': protocol['simulator_assessment_seeds'], 'final_seeds_generated': False,
        'point_model_selection': 'None: retain D08 development-sealed parameters; no adjustment to new regimes',
        'interval_calibration': 'One global multiplier >=1, smallest with >=.95 mean path coverage in every development regime; equal generated-case weights',
        'curvature': 'Exact local KL Hessian inverse diagonal via Woodbury; no posterior or conformal guarantee',
        'new_regime_generator': {'route_choice_shift': 'OD-group demand lognormal .25; path-preference lognormal 1.5',
            'sparse_active_paths': 'Gamma .7 truth with 15% active paths; diffuse positive spurious prior',
            'count_outliers': 'Gamma .7 truth/additive prior; 5% count outliers multiplied uniformly .5..1.5'},
        'truth_scope': 'Own generated truth and public train incidence/prior only; no public_reference',
        'assessment_scope': 'Old seeds used for other research; new generating regimes, not an untouched final test',
        'formal_ODME': None, 'formal_Total': None, 'uploaded': False})
    directory.mkdir(parents=True)
    snapshot = capture_source_snapshot(directory/'source_snapshot.zip')
    start = time.perf_counter()
    release = Release('data/kaggle_public')
    panels = release.panels()
    cached = {p: operator(release, p, 'train') for p in panels}
    records, development = [], []
    for panel in panels:
        A, _, prior, paths = cached[panel]
        for seed in protocol['simulator_development_seeds']:
            for index, regime in enumerate(REGIMES):
                truth, c, b = generated(A, prior, paths, seed+100*index, regime)
                f, info = entropy_inverse(A, c, b, **settings)
                sd, curvature = entropy_curvature_sd(A, f, settings['regularization']*np.maximum(c, 10.))
                ratios = np.sort(np.abs(f-truth)/(1.96*sd))
                records.append({'regime': regime, 'standardized_errors': ratios})
                calibration_path = directory/f'{panel}_{seed}_{regime}.npz'
                np.savez_compressed(calibration_path, standardized_errors=ratios)
                development.append({'panel': panel, 'seed': seed, 'regime': regime,
                    'calibration_array_sha256': sha256_file(calibration_path),
                    'point_sha256': hashlib.sha256(f.tobytes()).hexdigest(),
                    'raw_coverage': float((ratios <= 1.).mean()), **info, **curvature})
        print('calibration', panel, flush=True)
    calibrated = calibrate_interval_scale(records)
    (directory/'development.json').write_text(json.dumps(development, indent=2), encoding='utf-8')
    sealed = {'registered_protocol_sha256': registered['protocol_sha256'], 'calibration': calibrated,
        'calibration_record_sha256': hashlib.sha256(json.dumps(development, sort_keys=True).encode()).hexdigest()}
    (directory/'interval_sealed.json').write_text(json.dumps(sealed, indent=2), encoding='utf-8')
    # Assessment truth is first generated after the interval calibration is sealed.
    rows = []
    for panel in panels:
        A, _, prior, paths = cached[panel]
        for seed in protocol['simulator_assessment_seeds']:
            for index, regime in enumerate(REGIMES+NEW_REGIMES):
                truth, c, b = (generated if regime in REGIMES else mismatch)(A, prior, paths, seed+100*index, regime)
                for method in ['ridge_0.05', 'selected_entropy']:
                    tic = time.perf_counter()
                    try:
                        if method == 'ridge_0.05':
                            f, info = solve(A, c, b, 'ridge', .05)
                            extras = {}
                        else:
                            f, info = entropy_inverse(A, c, b, **settings)
                            sd, curvature = entropy_curvature_sd(A, f, settings['regularization']*np.maximum(c, 10.))
                            extras = {'curvature_diagnostic': curvature, **coverage_record(truth, f, sd, calibrated['multiplier'])}
                        row = {'success': True, **scores(A, c, b, truth, f, paths), **info, **extras}
                    except RuntimeError as error:
                        row = {'success': False, 'error': str(error), 'simulated_S_ODME': None}
                    rows.append({**row, 'panel': panel, 'seed': seed, 'regime': regime, 'method': method,
                        'seconds': time.perf_counter()-tic})
        (directory/'assessment_progress.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print('assessment', panel, flush=True)
    expected = len(panels)*len(protocol['simulator_assessment_seeds'])*(len(REGIMES)+len(NEW_REGIMES))*2
    keys = {(r['panel'], r['seed'], r['regime'], r['method']) for r in rows}
    if len(rows) != expected or len(keys) != expected or {r['method'] for r in rows} != {'ridge_0.05', 'selected_entropy'}:
        raise ValueError('Incomplete or overwritten assessment identities')
    summaries = {}
    for regime in REGIMES+NEW_REGIMES:
        subset = [r for r in rows if r['regime'] == regime]
        valid = all(r['success'] for r in subset)
        entropy = [r for r in subset if r['method'] == 'selected_entropy']
        summaries[regime] = {'complete_paired_comparison': valid,
            'simulated_S_ODME': {method: float(np.mean([r['simulated_S_ODME'] for r in subset if r['method'] == method])) if valid else None
                for method in ['ridge_0.05', 'selected_entropy']},
            'raw_curvature_coverage': float(np.mean([r['raw_curvature_interval']['case_path_coverage'] for r in entropy])) if valid else None,
            'calibrated_coverage': float(np.mean([r['empirical_development_interval']['case_path_coverage'] for r in entropy])) if valid else None,
            'calibrated_width_over_mean_true_flow': float(np.mean([r['empirical_development_interval']['mean_interval_width_over_mean_true_flow'] for r in entropy])) if valid else None,
            'simultaneous_case_coverage': float(np.mean([r['empirical_development_interval']['simultaneous_all_paths_covered'] for r in entropy])) if valid else None}
    families = sorted({p.rsplit('_', 1)[0] for p in panels})
    intervals = {scope: interval([r for r in rows if r['regime'] in regimes], 'ridge_0.05', 'selected_entropy',
        protocol['simulator_assessment_seeds'], families) for scope, regimes in [('old_regimes', REGIMES), ('new_regimes', NEW_REGIMES)]}
    result = {'registered': registered, 'source_snapshot': snapshot, 'calibration': calibrated,
        'interval_sealed_sha256': sha256_file(directory/'interval_sealed.json'), 'development': development,
        'assessment': rows, 'summary': summaries, 'paired_intervals': intervals,
        'optimizer_failures': sum(not r['success'] for r in rows), 'seconds': time.perf_counter()-start,
        'source': evidence(), 'formal_ODME': None, 'formal_Total': None, 'uploaded': False,
        'decision': 'Research diagnostic only; original formally verified Ridge submission preserved',
        'limitations': 'Synthetic prior/likelihood assumptions. Case-mean marginal coverage excludes simultaneous and distribution-shift guarantees. New regimes use previously seen seed identities; final seeds unopened.'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('interval_multiplier', calibrated['multiplier'], 'failures', result['optimizer_failures'], 'seconds', result['seconds'], flush=True)
    print(summaries, flush=True)


if __name__ == '__main__':
    main()
