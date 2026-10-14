"""Nine-distribution zero-preserving OD MAP and visible-prior selector control.

All choice rules are sealed on development seeds, before assessment generation.
Previously inspected assessment seeds remain development evidence, not final.
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
import pandas as pd
from trafficflow.data import Release
from trafficflow.odme import operator, solve
from trafficflow.entropy_od import entropy_inverse
from trafficflow.zero_aware_od import zero_aware_inverse, prior_top_mass, apply_selector
from trafficflow.metrics import od_link_score
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot, evidence, sha256_file
from entropy_od_experiment import generated, REGIMES, macro, interval
from entropy_od_uncertainty import mismatch
from od_probability_experiment import scores


GENERATOR_OFFSETS = {**{r: i for i, r in enumerate(REGIMES)}, 'sparse_active_paths': 8, 'count_outliers': 9}
NINE = list(GENERATOR_OFFSETS)
GRID = [
    {'path_cv': .25, 'scale_mode': 'absolute', 'shrinkage': 0., 'noise_cv': .02, 'huber_delta': None},
    {'path_cv': .75, 'scale_mode': 'absolute', 'shrinkage': 0., 'noise_cv': .02, 'huber_delta': None},
    {'path_cv': .75, 'scale_mode': 'absolute', 'shrinkage': .5, 'noise_cv': .02, 'huber_delta': None},
    {'path_cv': .75, 'scale_mode': 'absolute', 'shrinkage': 1., 'noise_cv': .02, 'huber_delta': None},
    {'path_cv': .25, 'scale_mode': 'relative', 'shrinkage': 0., 'noise_cv': .02, 'huber_delta': None},
    {'path_cv': .75, 'scale_mode': 'absolute', 'shrinkage': .5, 'noise_cv': .02, 'huber_delta': 2.5}]


def synthetic(A, prior, paths, seed, regime):
    offset_seed = seed+100*GENERATOR_OFFSETS[regime]
    return (generated if regime in REGIMES else mismatch)(A, prior, paths, offset_seed, regime)


def zero_diagnostics(truth, prediction):
    actual, predicted = truth <= 1e-8, prediction <= 1e-8
    return {'true_zero_fraction': float(actual.mean()), 'predicted_zero_fraction': float(predicted.mean()),
        'zero_support_recall': float(predicted[actual].mean()) if actual.any() else None,
        'zero_support_precision': float(actual[predicted].mean()) if predicted.any() else None,
        'inactive_path_prediction_mass_over_true_total': float(prediction[actual].sum()/max(truth.sum(), 1e-9))}


def run(A, c, b, truth, paths, method, config=None):
    start = time.perf_counter()
    try:
        if method == 'ridge_0.05':
            f, info = solve(A, c, b, 'ridge', .05)
        elif method == 'fixed_entropy_D08':
            f, info = entropy_inverse(A, c, b, regularization=.05, prior_floor=20.)
        elif method.startswith('grid_'):
            f, info = zero_aware_inverse(A, c, b, **config)
        else:
            f, info = apply_selector(A, c, b, config)
        if not np.isfinite(f).all() or (f < 0).any():
            raise RuntimeError('Invalid OD point prediction')
        return {'success': True, **scores(A, c, b, truth, f, paths), **zero_diagnostics(truth, f), **info,
            'prediction_sha256': hashlib.sha256(f.tobytes()).hexdigest(), 'seconds': time.perf_counter()-start}
    except (RuntimeError, np.linalg.LinAlgError) as error:
        return {'success': False, 'error': str(error), 'simulated_S_ODME': None, 'seconds': time.perf_counter()-start}


def select(rows):
    frame = pd.DataFrame(rows)
    base = frame[frame.method.eq('ridge_0.05')].set_index(['panel', 'seed', 'regime'])
    if not base.success.all():
        raise ValueError('Ridge development comparator failed')
    base_mean = macro(base.reset_index(), 'simulated_S_ODME')
    hypotheses, standalone = [], []
    for index, settings in enumerate(GRID):
        candidate = frame[frame.method.eq(f'grid_{index}')].set_index(['panel', 'seed', 'regime'])
        if set(candidate.index) != set(base.index):
            raise ValueError('Incomplete development grid')
        if not candidate.success.all():
            hypotheses.append({'grid': index, 'valid': False, 'optimizer_failures': int((~candidate.success).sum())})
            continue
        candidate = candidate.reindex(base.index)
        for threshold in [None, .60, .75, .85, .95]:
            applied = np.ones(len(base), bool) if threshold is None else base.prior_top15_mass.to_numpy(float) >= threshold
            selected = base.copy()
            selected['simulated_S_ODME'] = np.where(applied, candidate.simulated_S_ODME, base.simulated_S_ODME)
            selected = selected.reset_index()
            gains = {r: macro(selected[selected.regime.eq(r)], 'simulated_S_ODME')-
                macro(base.reset_index()[base.reset_index().regime.eq(r)], 'simulated_S_ODME') for r in NINE}
            row = {'valid': True, 'grid': index, 'threshold': threshold,
                'config': {'kind': 'zero_aware', 'settings': settings, 'threshold': threshold},
                'mean_gain': macro(selected, 'simulated_S_ODME')-base_mean,
                'regime_gain': gains, 'worst_regime_gain': min(gains.values()),
                'application_fraction': float(applied.mean())}
            hypotheses.append(row)
            if threshold is None:
                standalone.append(row)
    valid = [r for r in hypotheses if r['valid']]
    criterion = lambda r: (r['worst_regime_gain'], r['mean_gain'], -r['application_fraction'], -r['grid'])
    best = max(valid, key=criterion) if valid else None
    fixed = max(standalone, key=criterion) if standalone else None
    accepted = best is not None and best['mean_gain'] >= .005 and best['worst_regime_gain'] >= -.002
    return {'best_development_hypothesis': best, 'best_standalone': fixed,
        'development_gate_passed': accepted,
        'deployment_config': best['config'] if accepted else {'kind': 'ridge', 'threshold': None},
        'all_grid_and_threshold_evidence': hypotheses, 'mean_development_Ridge': base_mean,
        'scope': 'Single global setting and weak-prior-only gate, chosen from development seeds; assessment never selects parameters.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path('artifacts/research')/args.output.stem
    if args.output.exists() or directory.exists():
        raise FileExistsError('Never overwrite model or failed experiment evidence')
    protocol = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())['protocol']
    registered = register_protocol(args.output.with_suffix('.registered.json'), {
        'id': 'Zero_preserving_MAP_and_visible_prior_selector_nine_regimes', 'grid': GRID,
        'generating_regimes_and_seed_offsets': GENERATOR_OFFSETS,
        'historical_demand_replica_excluded': 'D11 proved route_choice_shift was another independent_paths case',
        'development_seeds': protocol['simulator_development_seeds'], 'assessment_seeds': protocol['simulator_assessment_seeds'],
        'final_seeds_generated': False, 'selector_thresholds': [None, .60, .75, .85, .95],
        'selector_input': 'Sum of largest ceil(.15*n_paths) weak-prior values / total prior; no truth or hidden labels',
        'selection': 'Maximum worst-regime dev gain then mean; tie-break less application and grid index',
        'development_gate': 'Mean gain >=.005, every regime mean >=-.002, full optimizer success; else deploy Ridge .05',
        'solver': 'Scaled nonnegative MAP; Gaussian or original-objective-certified Huber IRLS. Continuous prior, not discrete hurdle probability.',
        'assessment_scope': 'Previously viewed seeds and data distributions; development research, not an unopened final test',
        'formal_ODME': None, 'formal_Total': None, 'uploaded': False})
    directory.mkdir(parents=True)
    snapshot = capture_source_snapshot(directory/'source_snapshot.zip')
    start = time.perf_counter()
    release = Release('data/kaggle_public')
    panels = release.panels()
    cached = {p: operator(release, p, 'train') for p in panels}
    result = {'registered': registered, 'source_snapshot': snapshot, 'development': [], 'assessment': [],
        'formal_ODME': None, 'formal_Total': None, 'uploaded': False}
    for panel in panels:
        A, _, prior, paths = cached[panel]
        for seed in protocol['simulator_development_seeds']:
            for regime in NINE:
                truth, c, b = synthetic(A, prior, paths, seed, regime)
                for method, config in [('ridge_0.05', None), ('fixed_entropy_D08', None)]+[(f'grid_{i}', settings) for i, settings in enumerate(GRID)]:
                    row = run(A, c, b, truth, paths, method, config)
                    result['development'].append({**row, 'panel': panel, 'seed': seed, 'regime': regime,
                        'method': method, 'prior_top15_mass': prior_top_mass(b)})
        (directory/'progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('development', panel, flush=True)
    selection = select(result['development'])
    result['selection'] = selection
    (directory/'selection_sealed.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    (directory/'OD_selector.json').write_text(json.dumps(selection['deployment_config'], indent=2), encoding='utf-8')
    methods = {'ridge_0.05': None, 'fixed_entropy_D08': None, 'development_winner': selection['deployment_config']}
    if selection['best_development_hypothesis']:
        methods['visible_prior_hypothesis'] = selection['best_development_hypothesis']['config']
    if selection['best_standalone']:
        methods['standalone_MAP'] = selection['best_standalone']['config']
    # Assessment generation starts only after selection and deploy config are sealed.
    for panel in panels:
        A, _, prior, paths = cached[panel]
        for seed in protocol['simulator_assessment_seeds']:
            for regime in NINE:
                truth, c, b = synthetic(A, prior, paths, seed, regime)
                for method, config in methods.items():
                    row = run(A, c, b, truth, paths, method, config)
                    result['assessment'].append({**row, 'panel': panel, 'seed': seed, 'regime': regime,
                        'method': method, 'prior_top15_mass': prior_top_mass(b)})
        (directory/'progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('assessment', panel, flush=True)
    expected = len(panels)*len(protocol['simulator_assessment_seeds'])*len(NINE)*len(methods)
    keys = {(r['panel'], r['seed'], r['regime'], r['method']) for r in result['assessment']}
    if len(keys) != expected or len(result['assessment']) != expected:
        raise ValueError('Incomplete or repeated method/case coverage')
    assessment = pd.DataFrame(result['assessment'])
    result['summary'] = {}
    for method in methods:
        sub = assessment[assessment.method.eq(method)]
        result['summary'][method] = {'optimizer_failures': int((~sub.success).sum()),
            'macro_simulated_S_ODME': macro(sub, 'simulated_S_ODME') if sub.success.all() else None,
            'regimes': {r: macro(sub[sub.regime.eq(r)], 'simulated_S_ODME') if sub[sub.regime.eq(r)].success.all() else None for r in NINE},
            'sparse_inactive_prediction_mass_over_true_total': float(sub[sub.regime.eq('sparse_active_paths')].inactive_path_prediction_mass_over_true_total.mean()) if sub.success.all() else None}
    families = sorted({p.rsplit('_', 1)[0] for p in panels})
    result['paired_intervals'] = {m: interval(result['assessment'], 'ridge_0.05', m, protocol['simulator_assessment_seeds'], families) for m in methods if m != 'ridge_0.05'}
    result['public_train_stability'] = []
    for panel in panels:
        A, c, b, _ = cached[panel]
        for role, config in [('development_winner', selection['deployment_config']),
                             ('visible_prior_hypothesis', selection['best_development_hypothesis']['config'] if selection['best_development_hypothesis'] else {'kind': 'ridge'})]:
            try:
                f, info = apply_selector(A, c, b, config)
                result['public_train_stability'].append({**info, 'panel': panel, 'role': role, 'success': True,
                    'S_link_only': od_link_score(A, f, c), 'formal_ODME': None})
            except RuntimeError as error:
                result['public_train_stability'].append({'panel': panel, 'role': role, 'success': False, 'error': str(error), 'formal_ODME': None})
    result.update(seconds=time.perf_counter()-start, source=evidence(),
        optimizer_failures=sum(not r['success'] for part in ['development', 'assessment'] for r in result[part]),
        limitations='Known synthetic truth under nine assumed distributions; no hidden path/reference truth. MAP zeros are not posterior activation probabilities. Final seeds unopened; old assessment inspected, no formal promotion or upload.')
    (directory/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('development_gate', selection['development_gate_passed'], 'best_hypothesis', selection['best_development_hypothesis'], flush=True)
    print(result['summary'], result['paired_intervals'], 'seconds', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
