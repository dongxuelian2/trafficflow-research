"""Reuse development-only point scores, then evaluate fresh research seeds.

Nine generating distributions; no previous assessment values select the rule.
New research seeds are separate from the unopened final simulator seeds.
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
from trafficflow.od_expert import prior_expert_inverse
from trafficflow.runtime import capture_source_snapshot, evidence, sha256_file
from trafficflow.research_eval import register_protocol
from entropy_od_experiment import macro, interval
from zero_aware_od_experiment import GRID, NINE, GENERATOR_OFFSETS, synthetic, zero_diagnostics
from od_probability_experiment import scores


RESEARCH_SEEDS = [5003, 6203, 7001]


def select_expert(rows):
    frame = pd.DataFrame(rows)
    key = ['panel', 'seed', 'regime']
    base = frame[frame.method.eq('ridge_0.05')].set_index(key)
    reference = base.reset_index()
    evidence_rows = []
    for fallback in ['ridge', 'entropy']:
        fallback_rows = frame[frame.method.eq('ridge_0.05' if fallback == 'ridge' else 'fixed_entropy_D08')].set_index(key).reindex(base.index)
        if not fallback_rows.success.all():
            continue
        for index in [None, *range(len(GRID))]:
            alternatives = base if index is None else frame[frame.method.eq(f'grid_{index}')].set_index(key).reindex(base.index)
            if not alternatives.success.all():
                continue
            for threshold in ([None] if index is None else [None, .60, .75, .85, .95]):
                active = np.zeros(len(base), bool) if index is None else np.ones(len(base), bool) if threshold is None else base.prior_top15_mass.to_numpy(float) >= threshold
                selected = reference.copy()
                selected['simulated_S_ODME'] = np.where(active, alternatives.simulated_S_ODME, fallback_rows.simulated_S_ODME)
                gains = {r: macro(selected[selected.regime.eq(r)], 'simulated_S_ODME')-macro(reference[reference.regime.eq(r)], 'simulated_S_ODME') for r in NINE}
                evidence_rows.append({'config': {'fallback': fallback, 'alternative': None if index is None else GRID[index], 'threshold': threshold},
                    'grid': index, 'mean_gain': macro(selected, 'simulated_S_ODME')-macro(reference, 'simulated_S_ODME'),
                    'worst_regime_gain': min(gains.values()), 'regime_gain': gains, 'application_fraction': float(active.mean())})
    if not evidence_rows:
        raise ValueError('No complete expert development comparison')
    best = max(evidence_rows, key=lambda r: (r['worst_regime_gain'], r['mean_gain'], -r['application_fraction']))
    accepted = best['mean_gain'] >= .005 and best['worst_regime_gain'] >= -.002
    return {'best_hypothesis': best, 'development_gate_passed': accepted,
        'deployment_config': best['config'] if accepted else {'fallback': 'ridge', 'alternative': None, 'threshold': None},
        'all_development_choices': evidence_rows,
        'selection_scope': 'D12 development rows only; no assessment scores passed to this function'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path('artifacts/research')/args.output.stem
    if args.output.exists() or directory.exists():
        raise FileExistsError('Never overwrite expert evidence')
    protocol = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())['protocol']
    if set(RESEARCH_SEEDS) & set(protocol['simulator_development_seeds']+protocol['simulator_assessment_seeds']+protocol['simulator_final_seeds']):
        raise ValueError('New research seeds overlap reserved/previous seed groups')
    registered = register_protocol(args.output.with_suffix('.registered.json'), {
        'id': 'Development_cached_entropy_Gaussian_OD_expert_fresh_research_seeds',
        'development_cache_sha256': sha256_file(args.cache_study), 'generating_regimes_and_offsets': GENERATOR_OFFSETS,
        'development_seeds': protocol['simulator_development_seeds'], 'fresh_research_assessment_seeds': RESEARCH_SEEDS,
        'final_seeds_generated': False, 'fallbacks': ['ridge_0.05', 'D08_entropy_tau.05_floor20'],
        'alternative_grid': GRID, 'thresholds': [None, .60, .75, .85, .95],
        'selection': 'Global worst-regime development gain then mean, tie less alternative application',
        'development_gate': 'Mean >=.005, all regime means >=-.002, complete optimizer success',
        'routing_inputs': 'Visible weak prior top15% mass only; no truth, assessment feedback or reference paths',
        'assessment_scope': 'Fresh seed identities within the same nine researched generating distributions; not a final distribution-shift test',
        'formal_ODME': None, 'formal_Total': None, 'uploaded': False})
    directory.mkdir(parents=True)
    snapshot = capture_source_snapshot(directory/'source_snapshot.zip')
    start = time.perf_counter()
    cached = json.loads(args.cache_study.read_text())
    development = cached['development']
    if len(development) != 2160 or any(not r['success'] for r in development):
        raise ValueError('D12 development cache incomplete or failed')
    selection = select_expert(development)
    (directory/'development_only.json').write_text(json.dumps(development, indent=2), encoding='utf-8')
    (directory/'selection_sealed.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    (directory/'OD_expert.json').write_text(json.dumps(selection['deployment_config'], indent=2), encoding='utf-8')
    release = Release('data/kaggle_public')
    rows = []
    for panel in release.panels():
        A, _, prior, paths = operator(release, panel, 'train')
        for seed in RESEARCH_SEEDS:
            for regime in NINE:
                truth, c, b = synthetic(A, prior, paths, seed, regime)
                for method in ['ridge_0.05', 'development_expert']:
                    tic = time.perf_counter()
                    try:
                        if method == 'ridge_0.05':
                            f, info = solve(A, c, b, 'ridge', .05)
                        else:
                            f, info = prior_expert_inverse(A, c, b, selection['deployment_config'])
                        row = {'success': True, **scores(A, c, b, truth, f, paths), **zero_diagnostics(truth, f), **info,
                            'prediction_sha256': hashlib.sha256(f.tobytes()).hexdigest()}
                    except RuntimeError as error:
                        row = {'success': False, 'error': str(error), 'simulated_S_ODME': None}
                    rows.append({**row, 'method': method, 'panel': panel, 'seed': seed, 'regime': regime, 'seconds': time.perf_counter()-tic})
        (directory/'assessment_progress.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(panel, 'complete', flush=True)
    frame = pd.DataFrame(rows)
    families = sorted({p.rsplit('_', 1)[0] for p in release.panels()})
    if len({(r['panel'], r['seed'], r['regime'], r['method']) for r in rows}) != 540 or len(rows) != 540:
        raise ValueError('Incomplete fresh-seed method/case coverage')
    summary = {m: {regime: macro(frame[frame.method.eq(m) & frame.regime.eq(regime)], 'simulated_S_ODME')
        if frame[frame.method.eq(m) & frame.regime.eq(regime)].success.all() else None for regime in NINE} for m in ['ridge_0.05', 'development_expert']}
    result = {'registered': registered, 'source_snapshot': snapshot, 'selection': selection,
        'development_only_cache_sha256': sha256_file(directory/'development_only.json'),
        'selection_sealed_sha256': sha256_file(directory/'selection_sealed.json'), 'assessment': rows, 'summary': summary,
        'paired_interval': interval(rows, 'ridge_0.05', 'development_expert', RESEARCH_SEEDS, families),
        'optimizer_failures': sum(not r['success'] for r in rows), 'seconds': time.perf_counter()-start,
        'source': evidence(), 'formal_ODME': None, 'formal_Total': None, 'uploaded': False,
        'limitations': 'Synthetic generating families still chosen by researchers; no actual hidden OD truth. Fresh seeds improve independence but do not establish formal score, reliable posterior or final architecture completion.'}
    (directory/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(selection['best_hypothesis'], result['paired_interval'], summary, 'seconds', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
