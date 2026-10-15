"""Sealed two-stage OD controls and fresh same-law/distribution-shift assessment."""
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
from trafficflow.odme import operator,solve
from trafficflow.entropy_od import entropy_inverse
from trafficflow.log_demand_od import log_demand_inverse
from trafficflow.demand_residual_od import GROUP_SETTINGS,residual_from_demand
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from zero_aware_od_experiment import NINE,synthetic,zero_diagnostics
from od_shift_generators import NEW_REGIMES,shifted_synthetic
from od_probability_experiment import scores
from entropy_od_experiment import macro,interval


GRID=[{'center_blend':a,'regularization':r} for a,r in [(.25,.05),(.5,.05),(1.,.05),(.5,.5)]]
ASSESSMENT=[932003,932011,932023]


def method_result(A,c,b,truth,paths,method,group=None,settings=None):
    start=time.perf_counter()
    try:
        if method=='ridge_0.05':f,info=solve(A,c,b,'ridge',.05)
        elif method=='fixed_entropy_D08':f,info=entropy_inverse(A,c,b,regularization=.05,prior_floor=20.)
        elif method=='fixed_log_demand_D17':f,info=log_demand_inverse(A,c,b,paths,**GROUP_SETTINGS)
        else:
            if group is None:raise RuntimeError('Shared-demand stage failed;residual case cannot be scored')
            f,info=residual_from_demand(A,c,b,group,**settings)
        if not np.isfinite(f).all() or (f<0).any():raise RuntimeError('Invalid path predictions')
        return {'success':True,**scores(A,c,b,truth,f,paths),**zero_diagnostics(truth,f),
            'solver_info':info,'prediction_sha256':hashlib.sha256(f.tobytes()).hexdigest(),
            'seconds':time.perf_counter()-start},f
    except (RuntimeError,np.linalg.LinAlgError) as error:
        return {'success':False,'error':str(error),'simulated_S_ODME':None,'seconds':time.perf_counter()-start},None


def choose(development):
    frame=pd.DataFrame(development);reference=frame[frame.method=='fixed_entropy_D08']
    if len(reference)!=270 or not reference.success.all():raise ValueError('Strong development reference incomplete')
    rows=[]
    for index,settings in enumerate(GRID):
        candidate=frame[frame.method==f'grid_{index}'];valid=len(candidate)==270 and bool(candidate.success.all())
        gains={r:macro(candidate[candidate.regime==r],'simulated_S_ODME')-
            macro(reference[reference.regime==r],'simulated_S_ODME') for r in NINE} if valid else None
        rows.append({'index':index,'settings':settings,'valid':valid,'failures':int((~candidate.success).sum()),
            'regime_gains_vs_entropy':gains,'worst_regime_gain':min(gains.values()) if valid else None,
            'mean_gain_vs_entropy':macro(candidate,'simulated_S_ODME')-macro(reference,'simulated_S_ODME') if valid else None})
    valid=[r for r in rows if r['valid']]
    best=max(valid,key=lambda r:(r['worst_regime_gain'],r['mean_gain_vs_entropy'],-r['index'])) if valid else None
    accepted=best is not None and best['mean_gain_vs_entropy']>=.005 and best['worst_regime_gain']>=-.002
    return {'best_development_hypothesis':best,'all_settings':rows,'development_gate_passed':accepted,
        'deployment':'demand_residual' if accepted else 'fixed_entropy_D08',
        'scope':'Development only. Compare against strong entropy,not only the original Ridge;assessment never changes choice.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite completed or failed research')
    protocol=json.loads(Path('reports/research/protocol_v2_registered.json').read_text())['protocol']
    dev=protocol['simulator_development_seeds']
    if set(ASSESSMENT)&set(dev+protocol['simulator_assessment_seeds']+protocol['simulator_final_seeds']+[931003,931013,931031]):
        raise ValueError('Assessment research seeds overlap previous/final seeds')
    source_study=Path('reports/research/D17_log_demand_OD_numeric_polish.json')
    if json.loads(source_study.read_text())['selection']['best_hypothesis']['settings']!=GROUP_SETTINGS:
        raise ValueError('Frozen shared-demand setting changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Shared_demand_center_then_entropy_path_residual','grid':GRID,'fixed_group_settings':GROUP_SETTINGS,
        'fixed_group_study_sha256':sha256_file(source_study),'development_seeds':dev,'development_regimes':NINE,
        'fresh_assessment_seeds':ASSESSMENT,'same_law_assessment_regimes':NINE,'new_shift_assessment_regimes':NEW_REGIMES,
        'new_generator_source_sha256':sha256_file('scripts/od_shift_generators.py'),
        'selection':'Max worst-law development gain vs fixed entropy,then mean;gate mean>=.005,worst>=-.002,all cases succeed',
        'assessment':'After choice seal only;separately report same-laws and six newly registered stress laws,three fresh seeds',
        'truth':'Own generators,public train incidence/weakprior/metadata only;no reference or hidden answers',
        'uncertainty':'Paired five fixed families x three seeds,4000 resamples;not posterior/Kaggle/private coverage',
        'limits':'Counts reused in prior-center and correction stages,not a joint posterior. Positive model has no exact zero support.',
        'final_simulator_seeds_generated':False,'formal_ODME':None,'formal_Total':None,'uploaded':False})
    folder.mkdir(parents=True);(folder/'cases').mkdir();snapshot=capture_source_snapshot(folder/'source_snapshot.zip')
    start=time.perf_counter();release=Release('data/kaggle_public');panels=release.panels()
    families=sorted({p.rsplit('_',1)[0] for p in panels});cache={p:operator(release,p,'train') for p in panels}
    development=[];assessment=[];case_files=[]
    def one_case(panel,seed,regime,phase,selected=None):
        A,_,public_prior,paths=cache[panel]
        truth,c,b=(synthetic if regime in NINE else shifted_synthetic)(A,public_prior,paths,seed,regime)
        arrays={'truth':truth,'counts':c,'weak_prior':b};rows=[];group=None
        methods=[('ridge_0.05',None),('fixed_entropy_D08',None),('fixed_log_demand_D17',None)]
        if phase=='development':methods += [(f'grid_{i}',s) for i,s in enumerate(GRID)]
        elif selected is not None:methods += [('development_hypothesis',selected)]
        for method,settings in methods:
            row,f=method_result(A,c,b,truth,paths,method,group,settings)
            rows.append({**row,'panel':panel,'seed':seed,'regime':regime,'method':method})
            if f is not None:arrays[method]=f
            if method=='fixed_log_demand_D17':group=f
        path=folder/'cases'/(phase+'_'+panel+'_'+str(seed)+'_'+regime+'.npz')
        np.savez_compressed(path,**arrays)
        case_files.append({'path':path.as_posix(),'sha256':sha256_file(path),'phase':phase,'panel':panel,'seed':seed,'regime':regime})
        return rows
    with threadpool_limits(limits=4):
        for panel in panels:
            for seed in dev:
                for regime in NINE:development.extend(one_case(panel,seed,regime,'development'))
            (folder/'development_progress.json').write_text(json.dumps(development,indent=2,allow_nan=False),encoding='utf-8')
            print('development',panel,'complete',flush=True)
        selection=choose(development);seal=folder/'selection_sealed.json'
        seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8')
        print('sealed selection',selection,flush=True)
        chosen=selection['best_development_hypothesis'];settings=None if chosen is None else chosen['settings']
        for panel in panels:
            for seed in ASSESSMENT:
                for regime in NINE+NEW_REGIMES:assessment.extend(one_case(panel,seed,regime,'assessment',settings))
            (folder/'assessment_progress.json').write_text(json.dumps(assessment,indent=2,allow_nan=False),encoding='utf-8')
            print('assessment',panel,'complete',flush=True)
    primary_method='development_hypothesis' if selection['development_gate_passed'] else 'fixed_entropy_D08'
    assessment += [{**r,'method':'deployment_winner','alias_of':primary_method} for r in assessment if r['method']==primary_method]
    methods=sorted({r['method'] for r in assessment});summary={};comparisons={}
    for population,regimes in [('same_laws',NINE),('new_shift_laws',NEW_REGIMES)]:
        rows=[r for r in assessment if r['regime'] in regimes];frame=pd.DataFrame(rows)
        summary[population]={m:{r:macro(frame[(frame.method==m)&(frame.regime==r)],'simulated_S_ODME')
            if frame[(frame.method==m)&(frame.regime==r)].success.all() else None for r in regimes} for m in methods}
        comparisons[population]={m:interval(rows,'fixed_entropy_D08',m,ASSESSMENT,families) for m in methods if m!='fixed_entropy_D08'}
    result={'registered':registered,'source_snapshot':snapshot,'development':development,'selection':selection,
        'selection_seal_sha256':sha256_file(seal),'assessment':assessment,'summary':summary,'comparisons':comparisons,
        'case_files':case_files,'n_unique_simulated_cases':len(case_files),
        'optimizer_case_failure_records':sum(not r['success'] for r in development+assessment),
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_ODME':None,'formal_Total':None,'uploaded':False,
        'limitations':'First fresh research assessment for this model;not final or hidden-test accuracy. '
        'Same-laws/shift-laws share fixed public networks and three seeds;no universal statistical guarantee. '
        'Deployment alias duplicates an existing method,not independent evidence. Counts reused in two stages;point estimates only.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'experiment.json').write_text(text,encoding='utf-8')
    print('complete',comparisons,'failure_records',result['optimizer_case_failure_records'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
