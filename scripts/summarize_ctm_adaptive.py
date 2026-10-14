"""Inspect every frozen CTM case and verify a separate-process replay."""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.runtime import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repeat', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite diagnostics')
    original, repeat = [json.loads(p.read_text()) for p in (args.source, args.repeat)]
    for key in ['registered', 'rows', 'summary']:
        if original[key] != repeat[key]:
            raise ValueError('Independent CTM replay changed '+key)
    rows = original['rows']
    config = original['registered']['protocol']
    expected = {(seed, scenario, rate) for seed in config['seeds'] for scenario in config['cases'] for rate in config['mask_rates']}
    actual = [(r['seed'], r['scenario'], r['mask_rate']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Incomplete or repeated CTM cases')
    methods = list(original['summary'])
    if any(set(r['candidates']) != set(methods) or r['n_targets'] <= 0 for r in rows):
        raise ValueError('Incomplete candidate coverage')
    comparison = {}
    rng = np.random.default_rng(20261013)
    indices = rng.integers(0, len(config['seeds']), (4000, len(config['seeds'])))
    for method in methods:
        gains = np.array([np.mean([r['candidates'][method]['simulated_observation_State']-r['candidates']['interpolation']['simulated_observation_State']
                                  for r in rows if r['seed'] == seed]) for seed in config['seeds']])
        comparison[method] = {'seed_macro_gain': float(gains.mean()), 'individual_seed_macro_gains': gains.tolist(),
            'nominal_seed_bootstrap_ci95': np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist(),
            'n_seed_clusters': len(gains),
            'scope': 'Only three seeds and fixed six scenarios; masks share truth. Limited synthetic evidence, not 54 independent samples or a Kaggle CI.'}
    slices = {}
    for field in ['scenario', 'mask_rate']:
        slices[field] = {str(value): {method: float(np.mean([r['candidates'][method]['simulated_observation_State'] for r in rows if r[field] == value]))
                                    for method in methods} for value in sorted({r[field] for r in rows})}
    latent = {}
    for group in ['target', 'detector_failure', 'five_minute_drop_target', 'recovery_target']:
        latent[group] = {}
        for method in methods:
            sufficient = [r['candidates'][method]['latent_diagnostics'][group] for r in rows]
            n = sum(r['n'] for r in sufficient)
            latent[group][method] = {'sum_target_counts_over_shared_masks': n,
                **{metric: float(np.sqrt(sum(r['n']*r[metric]**2 for r in sufficient if r['n'])/n)) if n else None
                   for metric in ['speed_rmse', 'lane_flow_rmse', 'density_rmse']}}
    result = {'source_sha256': sha256_file(args.source), 'repeat_sha256': sha256_file(args.repeat),
        'independent_process_registered_rows_summary_exact': True, 'n_scenarios': len(rows),
        'max_conservation_error_vehicles': max(r['conservation_error'] for r in rows),
        'comparison': comparison, 'slices': slices, 'latent_diagnostics': latent,
        'formal_Total': None,
        'decision': 'No synthetic-assessment model selection. Positive average mixes lose mild-case State and true CTM-flux diagnostic; require real-train and latent/noise controls before any production change.'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('replay_exact', True, 'cases', len(rows), 'comparison', comparison, flush=True)


if __name__ == '__main__':
    main()
