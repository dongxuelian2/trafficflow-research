"""Compare frozen-prior assimilation studies, without selecting new models."""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.research_eval import paired_block_interval
from trafficflow.runtime import sha256_file


PAIRS = [
    ('A58_residual_assimilation_I210_W_autumn_repaired','A60_counterfactual_assimilation_I210_W_autumn'),
    ('A59_residual_assimilation_I5_N_winter','A61_counterfactual_assimilation_I5_N_winter'),
    ('A62_residual_assimilation_I405_N_winter','A63_counterfactual_assimilation_I405_N_winter')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite comparison evidence')
    reports, rows, pairs = {}, [], []
    for original, counterfactual in PAIRS:
        for name in [original,counterfactual]:
            path = Path('reports/research')/(name+'.json')
            run = json.loads(path.read_text())
            reports[path.name] = sha256_file(path)
            base = run['assessment']['frozen_prior']['metrics']
            for role in ['primary','joint']:
                choice = run['selected'][role]
                metric = run['assessment'][choice['key']]['metrics']
                rows.append({'study':name,'panel':run['registered']['protocol']['panel'],'role':role,
                    'selected_key':choice['key'],'strength':choice['strength'],
                    'cal_State':choice['metrics']['S_state'],'assessment_State':metric['S_state'],
                    'control_State':base['S_state'],'pooled_State_gain':metric['S_state']-base['S_state'],
                    'day_interval_vs_prior':run['paired_day_intervals'].get(choice['key']),
                    'target_touching_proxy_L1':metric['density_target_pair_relative_L1'],
                    'control_proxy_L1':base['density_target_pair_relative_L1'],'seconds':run['seconds'],
                    'n_solver_channel_calls':2*len(run['solver_records']),
                    'max_certified_linear_residual':max(c['relative_linear_residual'] for r in run['solver_records'] for c in r['certificates']),
                    'State_promoted':False,'formal_State':None,'formal_Physics':None})
        a = json.loads((Path('reports/research')/(original+'.json')).read_text())
        b = json.loads((Path('reports/research')/(counterfactual+'.json')).read_text())
        if a['registered']['protocol']['prior_sha256'] != b['registered']['protocol']['prior_sha256']:
            raise ValueError('Comparison prior weights differ')
        if a['registered']['protocol']['configurations'] != b['registered']['protocol']['configurations']:
            raise ValueError('Counterfactual solver grid changed')
        sse, count = np.zeros((2,2)), 0
        phase_counts = {}
        for phase in ['selection','assessment']:
            paths = sorted((Path('artifacts/research')/original/phase).glob('*.npz'))
            for p in paths:
                before = dict(np.load(p,allow_pickle=False))
                after = dict(np.load(Path('artifacts/research')/counterfactual/phase/p.name,allow_pickle=False))
                for key in before:
                    if not np.array_equal(before[key],after[key],equal_nan=True):
                        raise ValueError('Original prior/observation input changed: '+key)
                if phase == 'selection':
                    good = np.isfinite(before['strict']).all(axis=-1)
                    for i,center in enumerate([before['prior'],after['anchor_prior']]):
                        sse[i] += np.sum((before['strict'][good]-center[good])**2,axis=0)
                    count += int(good.sum())
            phase_counts[phase] = len(paths)
        contrasts = {}
        for role in ['primary','joint']:
            x = a['assessment'][a['selected'][role]['key']]['metrics']
            y = b['assessment'][b['selected'][role]['key']]['metrics']
            contrasts[role] = {'pooled_State_gain_CF_vs_original':y['S_state']-x['S_state'],
                'paired_day_interval':paired_block_interval([r['daily_State'] for r in x['days']],[r['daily_State'] for r in y['days']]),
                'proxy_L1_change_CF_minus_original':y['density_target_pair_relative_L1']-x['density_target_pair_relative_L1']}
        pairs.append({'original':original,'counterfactual':counterfactual,
            'prior_observations_geometry_exact':True,'matched_days':phase_counts,
            'same_prior_weight_sha256':a['registered']['protocol']['prior_sha256'],
            'cal_strict_anchor_count':count,'normalized_anchor_innovation_RMSE_original_CF':np.sqrt(sse/count).tolist(),
            'selected_strategy_contrasts':contrasts,
            'scope':'Same prior and solver/strength grid; different cal choices are part of each strategy. Counterfactual centering increases missingness; anchor observation RMSE is not latent truth or target accuracy.'})
    result = {'source_reports_sha256':reports,'rows':rows,'paired_strategies':pairs,
        'decision':'No State promotion: all gains below .002, only three panel/fold cases, one prior fit seed; no formal Physics evidence.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limitations':'Day-block intervals are for day mean, not pooled/Kaggle scores. These are old public-train development dates; no final simulator seeds or hidden labels.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print([(r['study'],r['role'],r['pooled_State_gain']) for r in rows],flush=True)


if __name__ == '__main__':
    main()
