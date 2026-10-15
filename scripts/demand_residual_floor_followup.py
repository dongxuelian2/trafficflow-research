"""Preregistered center-floor ablation;choices use D21 development only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from trafficflow.data import Release
from trafficflow.odme import operator
from trafficflow.demand_residual_od import residual_from_demand
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from zero_aware_od_experiment import NINE,synthetic,zero_diagnostics
from od_shift_generators import NEW_REGIMES,shifted_synthetic
from od_probability_experiment import scores
from entropy_od_experiment import macro,interval
from demand_residual_od_experiment import method_result


def select(rows,grid):
    frame=pd.DataFrame(rows);reference=frame[frame.method=='fixed_entropy_D08'];settings=[]
    if len(reference)!=270 or not reference.success.all():raise ValueError('Strong development comparator incomplete')
    for i,configuration in enumerate(grid):
        candidate=frame[frame.method==f'grid_{i}'];valid=len(candidate)==270 and bool(candidate.success.all())
        gains={r:macro(candidate[candidate.regime==r],'simulated_S_ODME')-
            macro(reference[reference.regime==r],'simulated_S_ODME') for r in NINE} if valid else None
        settings.append({'index':i,'settings':configuration,'valid':valid,'failures':int((~candidate.success).sum()),
            'regime_gains_vs_entropy':gains,'worst_regime_gain':min(gains.values()) if valid else None,
            'mean_gain_vs_entropy':macro(candidate,'simulated_S_ODME')-macro(reference,'simulated_S_ODME') if valid else None})
    valid=[r for r in settings if r['valid']]
    best=max(valid,key=lambda r:(r['worst_regime_gain'],r['mean_gain_vs_entropy'],-r['index'])) if valid else None
    accepted=best is not None and best['worst_regime_gain']>=-.002 and best['mean_gain_vs_entropy']>=.005
    return {'best_development_hypothesis':best,'all_settings':settings,'development_gate_passed':accepted,
        'deployment':'demand_residual' if accepted else 'fixed_entropy_D08',
        'scope':'D21 development arrays only;D21 assessment not read or used for model choice'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite floor followup')
    registered=json.loads(args.output.with_suffix('.registered.json').read_text());protocol=registered['protocol']
    source=Path('artifacts/research/D21_demand_entropy_path_residual');dev_path=source/'development_progress.json'
    if sha256_file(dev_path)!=protocol['source_development_sha256'] or sha256_file(source/'selection_sealed.json')!=protocol['source_selection_seal_sha256']:
        raise ValueError('Registered source development or choice changed')
    grid=protocol['grid'];dev=protocol['development_seeds'];assessment_seeds=protocol['fresh_assessment_seeds']
    expected=[{'center_blend':a,'regularization':.05,'residual_floor':f} for a,f in [(.25,2.),(.5,2.),(.25,20.),(.5,20.)]]
    if grid!=expected or set(assessment_seeds)&{411,829,1201,2909,3253,4001,7907,8017,9109,931003,931013,931031,932003,932011,932023}:
        raise ValueError('Registered settings or fresh seed separation changed')
    old_rows=json.loads(dev_path.read_text());old={(r['panel'],r['seed'],r['regime'],r['method']):r for r in old_rows}
    folder.mkdir(parents=True);(folder/'cases').mkdir();snapshot=capture_source_snapshot(folder/'source_snapshot.zip')
    start=time.perf_counter();release=Release('data/kaggle_public');panels=release.panels()
    cache={p:operator(release,p,'train') for p in panels};families=sorted({p.rsplit('_',1)[0] for p in panels})
    development=[];assessment=[];case_files=[]
    def write_case(phase,panel,seed,regime,arrays):
        path=folder/'cases'/(phase+'_'+panel+'_'+str(seed)+'_'+regime+'.npz');np.savez_compressed(path,**arrays)
        case_files.append({'phase':phase,'panel':panel,'seed':seed,'regime':regime,'path':path.as_posix(),'sha256':sha256_file(path)})
    with threadpool_limits(limits=4):
        for panel in panels:
            A,_,public_prior,paths=cache[panel]
            for seed in dev:
                for regime in NINE:
                    path=source/'cases'/('development_'+panel+'_'+str(seed)+'_'+regime+'.npz')
                    with np.load(path) as saved:arrays={key:saved[key].copy() for key in saved.files}
                    truth,c,b=synthetic(A,public_prior,paths,seed,regime)
                    if any(not np.array_equal(arrays[k],value) for k,value in [('truth',truth),('counts',c),('weak_prior',b)]):
                        raise ValueError('Cached development inputs differ from registered generator')
                    # Controls retain exact original bytes;not new solver calls.
                    for method in ['ridge_0.05','fixed_entropy_D08','fixed_log_demand_D17','grid_0','grid_1']:
                        row=dict(old[(panel,seed,regime,method)])
                        if row['success'] and hashlib.sha256(arrays[method].tobytes()).hexdigest()!=row['prediction_sha256']:
                            raise ValueError('Reused development prediction changed')
                        row['original_method_seconds']=row.pop('seconds');row['seconds']=0.
                        row['reused_D21_development_prediction']=True;development.append(row)
                    arrays.pop('grid_2',None);arrays.pop('grid_3',None)
                    group=arrays.get('fixed_log_demand_D17')
                    for i in [2,3]:
                        row,f=method_result(A,c,b,truth,paths,f'grid_{i}',group,grid[i])
                        development.append({**row,'panel':panel,'seed':seed,'regime':regime,'method':f'grid_{i}',
                            'reused_D21_development_prediction':False})
                        if f is not None:arrays[f'grid_{i}']=f
                    write_case('development',panel,seed,regime,arrays)
            (folder/'development_progress.json').write_text(json.dumps(development,indent=2,allow_nan=False),encoding='utf-8')
            print('floor development',panel,'complete',flush=True)
        selection=select(development,grid);seal=folder/'selection_sealed.json'
        seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8');print('floor sealed selection',selection,flush=True)
        best=selection['best_development_hypothesis']
        for panel in panels:
            A,_,public_prior,paths=cache[panel]
            for seed in assessment_seeds:
                for regime in NINE+NEW_REGIMES:
                    truth,c,b=(synthetic if regime in NINE else shifted_synthetic)(A,public_prior,paths,seed,regime)
                    arrays={'truth':truth,'counts':c,'weak_prior':b};group=None
                    methods=[('ridge_0.05',None),('fixed_entropy_D08',None),('fixed_log_demand_D17',None)]
                    if best is not None:methods.append(('development_hypothesis',best['settings']))
                    for method,settings in methods:
                        row,f=method_result(A,c,b,truth,paths,method,group,settings)
                        assessment.append({**row,'panel':panel,'seed':seed,'regime':regime,'method':method})
                        if f is not None:arrays[method]=f
                        if method=='fixed_log_demand_D17':group=f
                    write_case('assessment',panel,seed,regime,arrays)
            (folder/'assessment_progress.json').write_text(json.dumps(assessment,indent=2,allow_nan=False),encoding='utf-8')
            print('floor assessment',panel,'complete',flush=True)
    primary='development_hypothesis' if selection['development_gate_passed'] else 'fixed_entropy_D08'
    assessment += [{**r,'method':'deployment_winner','alias_of':primary} for r in assessment if r['method']==primary]
    comparisons={};summary={};methods=sorted({r['method'] for r in assessment})
    for population,regimes in [('same_laws',NINE),('new_shift_laws',NEW_REGIMES)]:
        rows=[r for r in assessment if r['regime'] in regimes];frame=pd.DataFrame(rows)
        comparisons[population]={m:interval(rows,'fixed_entropy_D08',m,assessment_seeds,families) for m in methods if m!='fixed_entropy_D08'}
        summary[population]={m:{r:macro(frame[(frame.method==m)&(frame.regime==r)],'simulated_S_ODME')
            if frame[(frame.method==m)&(frame.regime==r)].success.all() else None for r in regimes} for m in methods}
    result={'registered':registered,'source_snapshot':snapshot,'development':development,'selection':selection,
        'selection_seal_sha256':sha256_file(seal),'assessment':assessment,'case_files':case_files,'n_unique_simulated_cases':len(case_files),
        'comparisons':comparisons,'summary':summary,'optimizer_case_failure_records':sum(not r['success'] for r in development+assessment),
        'n_reused_development_predictions':sum(r.get('reused_D21_development_prediction',False) for r in development),
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_ODME':None,'formal_Total':None,'uploaded':False,
        'limitations':'Floor hypothesis preregistered from D21 development only;D21 assessment not read by selection code. '
        '1350development controls reused,not independent optimization evidence. Three new research seeds on own laws,not final/private accuracy. '
        'Same counts reused in both model stages;point estimates only.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'experiment.json').write_text(text,encoding='utf-8')
    print('floor complete',comparisons,'failure_records',result['optimizer_case_failure_records'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
