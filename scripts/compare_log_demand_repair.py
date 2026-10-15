"""Compare solver repairs without dropping failures or changing model choice."""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.runtime import sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite repair comparisons')
    before,after=[json.loads(p.read_text()) for p in [args.before,args.after]]
    fields=['grid','development_seeds','fresh_assessment_seeds','generating_regimes']
    if any(before['registered']['protocol'][k]!=after['registered']['protocol'][k] for k in fields):raise ValueError('Repair data/settings changed')
    original=before['selection']['best_hypothesis']['settings'];repaired=after['selection']['best_hypothesis']['settings']
    if original!=repaired or not after['selection']['original_choice_retained']:raise ValueError('Original model choice not retained')
    rows=[];recovered=[]
    for phase in ['development','assessment']:
        key=lambda r:(r['panel'],r['seed'],r['regime'],r['method'])
        old={key(r):r for r in before[phase]};new={key(r):r for r in after[phase]}
        if set(old)!=set(new):raise ValueError('Incomplete repair cases')
        for identity,left in old.items():
            right=new[identity]
            if not left['success'] and right['success']:
                recovered.append({'phase':phase,'identity':identity})
            if left['success'] and right['success']:
                drift=max(abs(left[k]-right[k]) for k in ['simulated_S_od','simulated_S_link','simulated_S_dev','simulated_S_attr','simulated_S_ODME'])
                rows.append({'phase':phase,'identity':identity,'maximum_metric_drift':drift,
                    'prediction_hash_identical':left.get('prediction_sha256')==right.get('prediction_sha256')})
            if left['success'] and not right['success']:raise ValueError('New numerical failure introduced')
    fixed=[r for r in rows if r['identity'][-1] in ['ridge_0.05','fixed_entropy_D08']]
    if not all(r['maximum_metric_drift']==0 and r['prediction_hash_identical'] for r in fixed):raise ValueError('Reference solver changed')
    result={'before_sha256':sha256_file(args.before),'after_sha256':sha256_file(args.after),
        'same_grid_generators_seeds':True,'same_original_selection':True,'unchanged_KKT_tolerance':2e-4,
        'n_recovered_recorded_cases':len(recovered),'recovered_cases':recovered,
        'all_reference_predictions_and_scores_exact':True,'n_reference_records':len(fixed),
        'n_old_successful_records_replayed':len(rows),'max_successful_metric_drift':max(r['maximum_metric_drift'] for r in rows),
        'old_successful_drift_records':sorted(rows,key=lambda r:r['maximum_metric_drift'],reverse=True)[:20],
        'formal_ODME':None,'formal_Total':None,
        'scope':'Same previously generated simulated cases; numerical repair regression,not new blind assessment. Deployment alias repeats hypothesis failures and is not independent evidence.'}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print('recovered_recorded',len(recovered),'references_exact',len(fixed),'max_metric_drift',result['max_successful_metric_drift'],flush=True)


if __name__=='__main__':main()
