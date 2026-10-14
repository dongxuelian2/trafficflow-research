"""Entropy OD inversion under broad independently generated prior mismatch.

Selection uses development seeds only. Simulator assessment seeds remain
distinct; final simulator seeds are not generated. No public_reference or
hidden path truth is read, and no predictions are uploaded.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.odme import operator, solve
from trafficflow.bayesian_od import posterior
from trafficflow.entropy_od import entropy_inverse
from trafficflow.metrics import od_link_score
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot, evidence, sha256_file
from od_probability_experiment import generate as forward_generate, scores
from od_prior_misspecification import generate as reverse_generate


REGIMES = ['independent_paths', 'hierarchical_demand', 'prior_total_bias',
    'absolute_additive_prior_noise', 'lognormal_prior_noise', 'missing_prior_paths', 'group_biased_prior']


def generated(A, public_prior, paths, seed, regime):
    if regime in REGIMES[:3]:
        truth, c = forward_generate(A, public_prior, paths, seed, regime)
        return truth, c, public_prior
    return reverse_generate(A, public_prior, paths, seed, regime)


def macro(frame, metric):
    frame = frame.copy()
    frame['family'] = frame.panel.str.rsplit('_', n=1).str[0]
    return float(frame.groupby(['family', 'seed', 'regime'])[metric].mean().groupby('family').mean().mean())


def interval(rows, baseline, candidate, seeds, families):
    frame = pd.DataFrame(rows)
    if not frame.loc[frame.method.isin([baseline, candidate]), 'success'].all():
        return {'ci95': None, 'reason': 'Optimizer failure invalidates the full paired comparison; failures are not dropped'}
    left = frame[frame.method.eq(baseline)].set_index(['panel', 'seed', 'regime']).simulated_S_ODME
    right = frame[frame.method.eq(candidate)].set_index(['panel', 'seed', 'regime']).simulated_S_ODME
    delta = (right-left).rename('gain').reset_index()
    delta['family'] = delta.panel.str.rsplit('_', n=1).str[0]
    matrix = delta.groupby(['family', 'seed']).gain.mean().unstack().reindex(index=families, columns=seeds).to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError('Incomplete family/seed paired evidence')
    rng = np.random.default_rng(20261013)
    family_indices = rng.integers(0, len(families), (4000, len(families)))
    seed_indices = rng.integers(0, len(seeds), (4000, len(seeds)))
    means = matrix[family_indices[:, :, None], seed_indices[:, None, :]].mean(axis=(1, 2))
    return {'paired_family_seed_macro_gain': float(matrix.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
        'n_families': len(families), 'n_independent_assessment_seeds': len(seeds), 'resamples': 4000,
        'scope': 'Synthetic generating distributions and a small number of seeds; no Kaggle confidence interval'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-study', type=Path, help='Preserved study for a numerical-solver repair comparison')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite OD evidence')
    directory = Path('artifacts/research')/args.output.stem
    if directory.exists():
        raise FileExistsError('Never overwrite OD artifacts')
    protocol = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())['protocol']
    fixed_bayes = json.loads(Path('reports/research/D02_probabilistic_OD_deterministic.json').read_text())
    grid = [{'regularization': r, 'prior_floor': p} for p in (2., 20.) for r in (.005, .05, 1.)]
    repair = None
    if args.reference_study:
        old = json.loads(args.reference_study.read_text())
        old_config = old['registered']['protocol']
        if old_config['grid'] != grid or old_config['generating_regimes'] != REGIMES or old_config['development_seeds'] != protocol['simulator_development_seeds'] or old_config['assessment_seeds'] != protocol['simulator_assessment_seeds']:
            raise ValueError('Numerical comparison must retain all generating/selection settings')
        repair = {'original_sha256': sha256_file(args.reference_study),
            'original_optimizer_failures': old['optimizer_failures'],
            'change': 'Roundoff-bounded residual-reducing Newton step near stationarity; tolerance remains 2e-6',
            'assessment_status': 'Same previously inspected simulator seeds; numerical regression, not a new blind test'}
    registered = register_protocol(args.output.with_suffix('.registered.json'), {
        'id': 'Measurement_space_entropy_OD_broad_prior_control', 'grid': grid,
        'generating_regimes': REGIMES, 'development_seeds': protocol['simulator_development_seeds'],
        'assessment_seeds': protocol['simulator_assessment_seeds'], 'final_seeds_generated': False,
        'generator': 'Three public-prior forward perturbations and four independent gamma-truth reverse-generated prior regimes',
        'selection': 'One global entropy setting chosen by maximum worst-regime development gain, then mean gain; no direction-specific tuning',
        'development_gate': 'At least .005 global mean gain and no regime mean loss below -.002, no optimizer failures; otherwise retain Ridge .05',
        'comparators': ['Ridge .05', 'D02 Bayesian fixed settings (formally rejected K01)', 'best development entropy setting'],
        'truth': 'Independently generated path flows only. Never reads public_reference or hidden answers.',
        'numerical_repair_reference': repair,
        'formal_ODME': None, 'formal_Total': None, 'uploaded': False})
    directory.mkdir(parents=True)
    snapshot = capture_source_snapshot(directory/'source_snapshot.zip')
    start = time.perf_counter()
    release = Release('data/kaggle_public')
    panels = release.panels()
    families = sorted({p.rsplit('_', 1)[0] for p in panels})
    cached = {p: operator(release, p, 'train') for p in panels}
    result = {'registered': registered, 'source_snapshot': snapshot, 'development': [], 'assessment': [],
        'formal_ODME': None, 'formal_Total': None, 'uploaded': False}
    def run(A, c, b, truth, paths, method, settings=None):
        tic = time.perf_counter()
        try:
            if method == 'ridge_0.05':
                f, info = solve(A, c, b, 'ridge', .05)
            elif method == 'fixed_D02_bayesian':
                f, _, info = posterior(A, c, b, paths, **settings)
            else:
                f, info = entropy_inverse(A, c, b, **settings)
            return {'success': True, **scores(A, c, b, truth, f, paths), **info, 'seconds': time.perf_counter()-tic}
        except RuntimeError as error:
            return {'success': False, 'error': str(error), 'simulated_S_ODME': None, 'seconds': time.perf_counter()-tic}
    for panel in panels:
        A, _, public_prior, paths = cached[panel]
        for seed in protocol['simulator_development_seeds']:
            for i, regime in enumerate(REGIMES):
                truth, c, b = generated(A, public_prior, paths, seed+100*i, regime)
                for method, settings in [('ridge_0.05', None)]+[(f'entropy_{i}', s) for i, s in enumerate(grid)]:
                    row = run(A, c, b, truth, paths, method, settings)
                    if method == 'ridge_0.05' and not row['success']:
                        raise RuntimeError('Development Ridge comparator failed: '+row['error'])
                    result['development'].append({'panel': panel, 'seed': seed, 'regime': regime, 'method': method, **row})
        (directory/'progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('development', panel, flush=True)
    frame = pd.DataFrame(result['development'])
    baseline_rows = frame[frame.method.eq('ridge_0.05')]
    baseline_mean = macro(baseline_rows, 'simulated_S_ODME')
    settings_summary = []
    for i, settings in enumerate(grid):
        subset = frame[frame.method.eq(f'entropy_{i}')]
        if not subset.success.all():
            settings_summary.append({'grid': i, 'valid': False, 'optimizer_failures': int((~subset.success).sum())})
            continue
        means = {r: macro(subset[subset.regime.eq(r)], 'simulated_S_ODME')-macro(baseline_rows[baseline_rows.regime.eq(r)], 'simulated_S_ODME') for r in REGIMES}
        settings_summary.append({'grid': i, 'valid': True, 'mean_gain': macro(subset, 'simulated_S_ODME')-baseline_mean,
            'regime_gain': means, 'worst_regime_gain': min(means.values()), 'settings': settings})
    valid = [r for r in settings_summary if r['valid']]
    best = max(valid, key=lambda r: (r['worst_regime_gain'], r['mean_gain'])) if valid else None
    winner = 'selected_entropy' if best and best['mean_gain'] >= .005 and best['worst_regime_gain'] >= -.002 else 'ridge_0.05'
    selection = {'best_entropy': best, 'winner': winner, 'all_grid_evidence': settings_summary,
        'mean_ridge_development': baseline_mean, 'selection_scope': 'Development seeds only; assessment choices unchanged'}
    result['selection'] = selection
    (directory/'selection_sealed.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    for panel in panels:
        A, public_counts, public_prior, paths = cached[panel]
        bayes_settings = fixed_bayes['grid'][fixed_bayes['panels'][panel]['selected_grid']]
        methods = [('ridge_0.05', None), ('fixed_D02_bayesian', bayes_settings)]
        if best:
            methods.append(('selected_entropy', best['settings']))
        for seed in protocol['simulator_assessment_seeds']:
            for i, regime in enumerate(REGIMES):
                truth, c, b = generated(A, public_prior, paths, seed+100*i, regime)
                for method, settings in methods:
                    row = run(A, c, b, truth, paths, method, settings)
                    result['assessment'].append({'panel': panel, 'seed': seed, 'regime': regime, 'method': method, **row})
        (directory/'progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('assessment', panel, flush=True)
    assessment = pd.DataFrame(result['assessment'])
    result['summary'] = {}
    for method in assessment.method.unique():
        sub = assessment[assessment.method.eq(method)]
        result['summary'][method] = {'optimizer_failures': int((~sub.success).sum()),
            'macro_simulated_S_ODME': macro(sub, 'simulated_S_ODME') if sub.success.all() else None,
            'regimes': {r: macro(sub[sub.regime.eq(r)], 'simulated_S_ODME') if sub[sub.regime.eq(r)].success.all() else None for r in REGIMES}}
    result['paired_intervals'] = {m: interval(result['assessment'], 'ridge_0.05', m, protocol['simulator_assessment_seeds'], families)
        for m in assessment.method.unique() if m != 'ridge_0.05'}
    result['public_train_stability'] = []
    if best:
        for panel in panels:
            A, c, b, paths = cached[panel]
            old, _ = solve(A, c, b, 'ridge', .05)
            try:
                f, info = entropy_inverse(A, c, b, **best['settings'])
                delta = f-old
                row = np.asarray(A.T@np.linalg.pinv((A@A.T).toarray(), rcond=1e-10)@(A@delta))
                result['public_train_stability'].append({'panel': panel, 'success': True,
                    'S_link_only': od_link_score(A, f, c), 'total_flow_ratio_vs_ridge': float(f.sum()/max(old.sum(), 1e-9)),
                    'nullspace_L2_change_fraction': float(np.dot(delta-row, delta-row)/max(np.dot(delta, delta), 1e-30)),
                    'prior_relative_L1_change': float(np.abs(f-b).sum()/max(b.sum(), 1e-9)), **info,
                    'formal_ODME': None})
            except RuntimeError as error:
                result['public_train_stability'].append({'panel': panel, 'success': False, 'error': str(error), 'formal_ODME': None})
    result.update(seconds=time.perf_counter()-start, source=evidence(),
        optimizer_failures=sum(not row['success'] for part in ('development', 'assessment') for row in result[part]),
        limitations='Synthetic path truth and mismatched weak priors, not hidden OD labels. Assessment seeds were used by earlier simulator studies; final seeds remain unopened. A successful synthetic candidate cannot reverse the verified K01 formal failure. Global setting and promotion decision are sealed on development; no assessment reselection.')
    (directory/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(selection['winner'], result['summary'], result['paired_intervals'], result['seconds'], flush=True)
