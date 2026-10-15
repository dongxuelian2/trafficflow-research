"""Broad known-truth OD demand-factor control,selection before fresh seeds."""
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
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from zero_aware_od_experiment import synthetic,NINE,zero_diagnostics
from entropy_od_experiment import macro,interval
from od_probability_experiment import scores


GRID=[{'group_sd':sd,'noise_cv':cv,'prior_floor':2.,'huber_delta':huber} for sd,cv,huber in
      [(.2,.02,None),(.6,.02,None),(.6,.1,None),(.6,.02,2.5)]]
ASSESSMENT=[931003,931013,931031]


def run(A,c,b,truth,paths,method,config=None):
    start=time.perf_counter()
    try:
        if method=='ridge_0.05':f,info=solve(A,c,b,'ridge',.05)
        elif method=='fixed_entropy_D08':f,info=entropy_inverse(A,c,b,regularization=.05,prior_floor=20.)
        else:f,info=log_demand_inverse(A,c,b,paths,**config)
        if not np.isfinite(f).all() or (f<0).any():raise RuntimeError('Invalid group demand output')
        return {'success':True,**scores(A,c,b,truth,f,paths),**zero_diagnostics(truth,f),
            'solver_info':info,'prediction_sha256':hashlib.sha256(f.tobytes()).hexdigest(),'seconds':time.perf_counter()-start}
    except (RuntimeError,np.linalg.LinAlgError) as error:
        return {'success':False,'error':str(error),'simulated_S_ODME':None,'seconds':time.perf_counter()-start}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-study',type=Path);args=parser.parse_args()
    directory=Path('artifacts/research')/args.output.stem
    if args.output.exists() or directory.exists():raise FileExistsError('Never overwrite OD controls')
    protocol=json.loads(Path('reports/research/protocol_v2_registered.json').read_text())['protocol'];dev=protocol['simulator_development_seeds']
    if set(ASSESSMENT)&set(dev+protocol['simulator_assessment_seeds']+protocol['simulator_final_seeds']):raise ValueError('Research/final seed overlap')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Positive_origin_destination_log_demand_broad_prior_control','grid':GRID,'development_seeds':dev,
        'fresh_assessment_seeds':ASSESSMENT,'generating_regimes':NINE,'final_seeds_generated':False,
        'model':'f=max(b,2)*exp(origin effect+destination effect),Gaussian log-effect prior;countGaussian or Huber',
        'scope':'Low-dimensional positive demand-factor MAP,not old additive Gaussian covariance or alternative path choice',
        'optimization':'Two starts,bounds+-3,L-BFGS-B plus exact-Hessian/roundoff-budget polish;unchanged projected-gradient<=2e-4,local only',
        'numerical_repair_reference':None if args.reference_study is None else {
            'path':args.reference_study.as_posix(),'sha256':sha256_file(args.reference_study),
            'policy':'Same grid/data/seeds/objective/tolerance;retain original sealed selection;assessment is numerical regression'},
        'selection':'One setting max worst-regime dev gain then mean;accept only mean>=.005 and worst>=-.002 and all successful',
        'assessment':'Generate only after choice seal;evaluate ridge,entropy,selected deployment policy and fixed dev hypothesis',
        'truth':'Own forward/reverse known-truth generators,public network/weakprior only;no reference or hidden path truth',
        'formal_ODME':None,'formal_Total':None,'uploaded':False})
    directory.mkdir(parents=True);snapshot=capture_source_snapshot(directory/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');panels=release.panels();families=sorted({p.rsplit('_',1)[0] for p in panels})
    caches={p:operator(release,p,'train') for p in panels};development=[];assessment=[]
    with threadpool_limits(limits=4):
        for panel in panels:
            A,_,public_prior,paths=caches[panel]
            for seed in dev:
                for regime in NINE:
                    truth,c,b=synthetic(A,public_prior,paths,seed,regime)
                    for method,settings in [('ridge_0.05',None),('fixed_entropy_D08',None)]+[(f'grid_{i}',s) for i,s in enumerate(GRID)]:
                        row=run(A,c,b,truth,paths,method,settings)
                        development.append({**row,'panel':panel,'seed':seed,'regime':regime,'method':method})
            (directory/'development_progress.json').write_text(json.dumps(development,indent=2,allow_nan=False),encoding='utf-8')
            print('development',panel,'complete',flush=True)
        frame=pd.DataFrame(development);baseline=frame[frame.method=='ridge_0.05']
        if not baseline.success.all():raise ValueError('Reference Ridge development failed')
        rows=[]
        for i,setting in enumerate(GRID):
            candidate=frame[frame.method==f'grid_{i}'];valid=bool(candidate.success.all())
            gains={r:macro(candidate[candidate.regime==r],'simulated_S_ODME')-macro(baseline[baseline.regime==r],'simulated_S_ODME') for r in NINE} if valid else None
            rows.append({'index':i,'settings':setting,'valid':valid,'failures':int((~candidate.success).sum()),
                'regime_gains':gains,'mean_gain':macro(candidate,'simulated_S_ODME')-macro(baseline,'simulated_S_ODME') if valid else None,
                'worst_regime_gain':min(gains.values()) if valid else None})
        valid=[r for r in rows if r['valid']];best=max(valid,key=lambda r:(r['worst_regime_gain'],r['mean_gain'],-r['index'])) if valid else None
        accepted=best is not None and best['mean_gain']>=.005 and best['worst_regime_gain']>=-.002
        selection={'best_hypothesis':best,'all_settings':rows,'development_gate_passed':accepted,
            'deployment':'log_demand' if accepted else 'ridge_0.05','scope':'Development only;no assessment score used'}
        if args.reference_study:
            previous=json.loads(args.reference_study.read_text());old_protocol=previous['registered']['protocol']
            if old_protocol['grid']!=GRID or old_protocol['development_seeds']!=dev or old_protocol['fresh_assessment_seeds']!=ASSESSMENT:
                raise ValueError('Numerical repair changed generating/selection settings')
            selection['recomputed_dev_choice']=selection['best_hypothesis']
            selection['best_hypothesis']=previous['selection']['best_hypothesis']
            selection['development_gate_passed']=previous['selection']['development_gate_passed']
            selection['deployment']=previous['selection']['deployment']
            selection['original_choice_retained']=True;selection['reference_report_sha256']=sha256_file(args.reference_study)
            best=selection['best_hypothesis'];accepted=selection['development_gate_passed']
        seal=directory/'selection_sealed.json';seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8')
        print('development selection',selection,flush=True)
        for panel in panels:
            A,_,public_prior,paths=caches[panel]
            for seed in ASSESSMENT:
                for regime in NINE:
                    truth,c,b=synthetic(A,public_prior,paths,seed,regime)
                    methods=[('ridge_0.05',None),('fixed_entropy_D08',None)]
                    if best is not None:methods.append(('development_hypothesis',best['settings']))
                    for method,settings in methods:
                        row=run(A,c,b,truth,paths,method,settings)
                        assessment.append({**row,'panel':panel,'seed':seed,'regime':regime,'method':method})
            (directory/'assessment_progress.json').write_text(json.dumps(assessment,indent=2,allow_nan=False),encoding='utf-8')
            print('assessment',panel,'complete',flush=True)
    # Retained Ridge primary is explicit,even if a secondary case looks better.
    if not accepted:
        primary=[{**r,'method':'deployment_winner'} for r in assessment if r['method']=='ridge_0.05']
    else:primary=[{**r,'method':'deployment_winner'} for r in assessment if r['method']=='development_hypothesis']
    assessment+=primary
    methods=sorted(set(r['method'] for r in assessment));af=pd.DataFrame(assessment)
    summaries={m:{r:macro(af[(af.method==m)&(af.regime==r)],'simulated_S_ODME') if af[(af.method==m)&(af.regime==r)].success.all() else None for r in NINE} for m in methods}
    comparisons={m:interval(assessment,'ridge_0.05',m,ASSESSMENT,families) for m in methods if m!='ridge_0.05'}
    result={'registered':registered,'source_snapshot':snapshot,'development':development,'selection':selection,
        'selection_seal_sha256':sha256_file(seal),'assessment':assessment,'summary':summaries,'comparisons':comparisons,
        'optimizer_case_failures':sum(not r['success'] for r in development+assessment),
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_ODME':None,'formal_Total':None,'uploaded':False,
        'limitations':'Own generator laws and three fresh research seeds only. Positive factor model cannot represent true sparse zeros;local stationarity not global optimum. Old Bayes K01 formal failure remains valid. No real path truth or posterior coverage,final seeds unopened.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(directory/'experiment.json').write_text(text,encoding='utf-8')
    print('complete comparisons',comparisons,'failures',result['optimizer_case_failures'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
