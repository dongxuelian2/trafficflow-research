"""Regenerate every synthetic case,recalculate metrics and replay direct API."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from trafficflow.data import Release
from trafficflow.odme import operator,solve
from trafficflow.entropy_od import entropy_inverse
from trafficflow.log_demand_od import log_demand_inverse
from trafficflow.demand_residual_od import demand_residual_inverse,GROUP_SETTINGS
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from zero_aware_od_experiment import synthetic,NINE
from od_shift_generators import shifted_synthetic
from entropy_od_experiment import macro


def independent_scores(A,c,b,truth,f,paths):
    total=truth.sum();path_error=np.abs(f-truth).sum()/total
    od=max(0.,1-path_error);link=max(0.,1-np.abs(A@f-c).sum()/max(np.abs(c).sum(),1e-9))
    deviation=np.exp(-abs(np.abs(f-b).sum()/max(np.abs(truth-b).sum(),1e-9)-1))
    d=paths.destination_zone.to_numpy(str)
    t=np.array([truth[d==label].sum() for label in sorted(set(d))])/total
    p=np.array([f[d==label].sum() for label in sorted(set(d))])/max(f.sum(),1e-9)
    attribution=max(0.,1-.5*np.abs(t-p).sum())
    return dict(simulated_S_od=od,simulated_S_link=link,simulated_S_dev=deviation,
        simulated_S_attr=attribution,simulated_S_ODME=.45*od+.25*link+.15*deviation+.15*attribution,
        relative_path_L1=path_error)


def selection_stability(development):
    frame=pd.DataFrame(development);frame['family']=frame.panel.str.rsplit('_',n=1).str[0]
    results=[]
    for heldout in sorted(frame.family.unique()):
        fit=frame[frame.family!=heldout];reference=fit[fit.method=='fixed_entropy_D08'];options=[]
        for method in sorted(m for m in fit.method.unique() if m.startswith('grid_')):
            candidate=fit[fit.method==method]
            if not candidate.success.all():continue
            gains={r:macro(candidate[candidate.regime==r],'simulated_S_ODME')-
                macro(reference[reference.regime==r],'simulated_S_ODME') for r in NINE}
            options.append({'method':method,'worst_regime_gain':min(gains.values()),
                'mean_gain':macro(candidate,'simulated_S_ODME')-macro(reference,'simulated_S_ODME')})
        best=max(options,key=lambda r:(r['worst_regime_gain'],r['mean_gain'],-int(r['method'].split('_')[1]))) if options else None
        accepted=best is not None and best['worst_regime_gain']>=-.002 and best['mean_gain']>=.005
        results.append({'excluded_family':heldout,'development_only_best':best,'gate_passed':accepted})
    return {'rows':results,'scope':'Residual hyperparameter selection sensitivity only;upstream group setting already saw all families. '
        'Not full unseen-corridor validation;never changes the sealed primary choice.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite independent verification')
    study=json.loads(args.study.read_text());source_folder=Path('artifacts/research')/args.study.stem
    if json.loads((source_folder/'selection_sealed.json').read_text())!=study['selection']:
        raise ValueError('Sealed model choice differs')
    if sha256_file(source_folder/'selection_sealed.json')!=study['selection_seal_sha256']:raise ValueError('Selection seal changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Demand_entropy_full_case_regeneration_and_direct_API_replay','source_study_sha256':sha256_file(args.study),
        'scope':'Every case data and score;direct end-to-end API on two panels,5assessment laws and one development law',
        'formal_ODME':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');cache={p:operator(release,p,'train') for p in release.panels()}
    grouped={};case_checks=[];fresh_checks=[];max_score_error=0.;score_records=0
    for phase in ['development','assessment']:
        for row in study[phase]:grouped.setdefault((phase,row['panel'],row['seed'],row['regime']),[]).append(row)
    identities={(r['phase'],r['panel'],r['seed'],r['regime']) for r in study['case_files']}
    if identities!=set(grouped) or len(identities)!=720 or len(study['case_files'])!=720:raise ValueError('Case coverage differs')
    fresh_seed=study['registered']['protocol']['fresh_assessment_seeds'][0]
    dev_seed=study['registered']['protocol']['development_seeds'][0]
    with threadpool_limits(limits=4):
        for item in study['case_files']:
            identity=(item['phase'],item['panel'],item['seed'],item['regime']);A,_,prior,paths=cache[item['panel']]
            if sha256_file(item['path'])!=item['sha256']:raise ValueError('Saved case bytes changed')
            truth,c,b=(synthetic if item['regime'] in NINE else shifted_synthetic)(A,prior,paths,item['seed'],item['regime'])
            fresh=item['panel'] in ['D12_I5_N','D7_I210_W'] and (
                item['phase']=='assessment' and item['seed']==fresh_seed and item['regime'] in [
                    'hierarchical_demand','missing_prior_paths','distance_gravity','gamma_poisson_counts','sparse_zone_shutdown'] or
                item['phase']=='development' and item['seed']==dev_seed and item['regime']=='group_biased_prior')
            with np.load(item['path']) as arrays:
                for key,value in [('truth',truth),('counts',c),('weak_prior',b)]:
                    if not np.array_equal(arrays[key],value):raise ValueError('Fresh generated inputs differ')
                for row in grouped[identity]:
                    method=row.get('alias_of',row['method'])
                    if not row['success']:
                        if method in arrays.files:raise ValueError('Failed case unexpectedly has predictions')
                        continue
                    f=arrays[method]
                    if hashlib.sha256(f.tobytes()).hexdigest()!=row['prediction_sha256']:raise ValueError('Saved prediction hash differs')
                    values=independent_scores(A,c,b,truth,f,paths)
                    error=max(abs(float(v)-row[k]) for k,v in values.items());max_score_error=max(max_score_error,error)
                    if error>1e-12:raise ValueError('Independent score differs')
                    score_records+=1
                    if not fresh or row['method']=='deployment_winner':continue
                    if method=='ridge_0.05':other,_=solve(A,c,b,'ridge',.05)
                    elif method=='fixed_entropy_D08':other,_=entropy_inverse(A,c,b,regularization=.05,prior_floor=20.)
                    elif method=='fixed_log_demand_D17':other,_=log_demand_inverse(A,c,b,paths,**GROUP_SETTINGS)
                    else:
                        settings=(study['registered']['protocol']['grid'][int(method.split('_')[1])] if method.startswith('grid_')
                            else study['selection']['best_development_hypothesis']['settings'])
                        other,_=demand_residual_inverse(A,c,b,paths,**settings)
                    if not np.array_equal(other,f):raise ValueError('Direct uncached end-to-end API prediction differs')
                    fresh_checks.append({'phase':item['phase'],'panel':item['panel'],'seed':item['seed'],'regime':item['regime'],
                        'method':method,'exact_prediction_sha256':row['prediction_sha256']})
            case_checks.append({**item,'fresh_generated_data_exact':True,'all_successful_prediction_hashes_and_scores_checked':True})
    result={'registered':registered,'source_snapshot':snapshot,'n_cases_regenerated':len(case_checks),
        'case_checks':case_checks,'n_successful_score_records_recalculated':score_records,
        'max_independent_score_error':max_score_error,'n_direct_API_method_case_replays':len(fresh_checks),
        'direct_API_replays':fresh_checks,'all_data_exact':True,'all_direct_API_predictions_exact':True,
        'development_only_family_selection_stability':selection_stability(study['development']),
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_ODME':None,'formal_Total':None,
        'limitations':'Regeneration verifies implementation,not generator realism. Direct API cases are12unique simulated '
        'problems with multiple methods,not independent performance samples. No new model selection or hidden data.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'verification.json').write_text(text,encoding='utf-8')
    print('cases_exact',len(case_checks),'scores_checked',score_records,'score_max_error',max_score_error,
        'direct_API_calls_exact',len(fresh_checks),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
