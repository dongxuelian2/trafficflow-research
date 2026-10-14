"""Verify D07 closes preserved D06 failures without changing the data or gate."""
import json
import argparse
from pathlib import Path
import numpy as np
from trafficflow.runtime import sha256_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, default=Path('reports/research/D06_entropy_OD_broad_prior.json'))
    parser.add_argument('--after', type=Path, default=Path('reports/research/D07_entropy_OD_numeric_repair.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/research/D06_D07_numeric_repair_verification.json'))
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError('Never overwrite numerical repair verification')
    paths = [args.before, args.after]
    old, new = [json.loads(p.read_text()) for p in paths]
    for key in ('grid', 'generating_regimes', 'development_seeds', 'assessment_seeds', 'development_gate', 'selection'):
        if old['registered']['protocol'][key] != new['registered']['protocol'][key]:
            raise ValueError('Numerical repair changed generating or decision settings')
    key = lambda r: (r['panel'], r['seed'], r['regime'], r['method'])
    repaired = {key(r): r for r in new['development']}
    if set(repaired) != {key(r) for r in old['development']}:
        raise ValueError('Development scene/method identities changed')
    failures = [r for r in old['development'] if not r['success']]
    closed = [r for r in failures if repaired[key(r)]['success']]
    if len(closed) != len(failures):
        raise ValueError('A preserved optimizer failure remains unresolved')
    baseline_differences = [abs(r['simulated_S_ODME']-repaired[key(r)]['simulated_S_ODME'])
        for r in old['development'] if r['method'] == 'ridge_0.05']
    newly_failed = [r for r in old['development'] if r['success'] and not repaired[key(r)]['success']]
    successful_entropy_differences = [abs(r['simulated_S_ODME']-repaired[key(r)]['simulated_S_ODME'])
        for r in old['development'] if r['success'] and repaired[key(r)]['success'] and r['method'].startswith('entropy_')]
    certificates = [r['primal_stationarity_max_abs'] for r in new['development'] if r['method'].startswith('entropy_') and r['success']]
    if max(baseline_differences) != 0 or max(certificates) > 2e-6:
        raise ValueError('Unchanged baseline or original stationarity tolerance failed')
    result = {'source_study_sha256': {p.name: sha256_file(p) for p in paths},
        'generators_grid_seeds_and_development_gate_unchanged': True,
        'preserved_failed_development_solves': len(failures), 'closed_failed_solves': len(closed),
        'remaining_total_optimizer_failures': new['optimizer_failures'],
        'newly_failed_cases': [{k: r[k] for k in ('panel', 'seed', 'regime', 'method')} for r in newly_failed],
        'Ridge_baseline_max_score_difference': max(baseline_differences),
        'previous_successful_entropy_max_score_difference': max(successful_entropy_differences),
        'max_development_primal_stationarity': max(certificates),
        'precision_fallback_total': sum(r.get('precision_fallbacks', 0) for r in new['development']),
        'seconds_before': old['seconds'], 'seconds_after': new['seconds'],
        'old_winner': old['selection']['winner'], 'new_winner': new['selection']['winner'],
        'new_selected_setting': new['selection']['best_entropy'], 'formal_ODME': None,
        'scope': 'Numerical regression on existing synthetic cases, not an independent final benchmark or a formal gain'}
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)
