from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from .data import Release,REPO,SPLITS,csv

def dump(obj,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def validate(release,output,full_hash=False):
    result=release.inventory(output,full_hash)
    for panel in release.panels():
        release.network(panel)
        for split in SPLITS:
            paths=release.files(panel,split)
            if not paths:
                raise ValueError(f'{panel}/{split}: missing masked partitions')
            for task in [1,4]:
                release.template(task,panel,split)
            if not panel.startswith('D12_I405'):
                release.template(2,panel,split)
            target=release.template(1,panel,split)
            # One day in one regime; never combine alternative mask views.
            seen={}
            for path in paths:
                frame=release.read(path)
                for date,regimes in frame.groupby(frame.timestamp.str[:10]).mask_regime.unique().items():
                    if len(regimes)!=1 or date in seen:
                        raise ValueError('Day appears under multiple mask views or partitions')
                    seen[date]=regimes[0]
                needed=target[target.timestamp.isin(frame.timestamp.unique())]
                joined=needed.merge(frame[['timestamp','station_id','link_id','mask_regime','speed_kmh','flow_vph']],
                         on=['timestamp','station_id','link_id','mask_regime'],how='left',suffixes=('','_input'),indicator=True)
                if not joined._merge.eq('both').all() or joined[['speed_kmh_input','flow_vph_input']].notna().any().any():
                    raise ValueError('Template targets absent or exposed in masked inputs')
            if not target.timestamp.str[:10].isin(seen).all():
                raise ValueError('Template contains an absent date')
    result['validation']='passed';dump(result,output)
    print('Release validation passed')

def official_baselines(release,output,panels,split):
    src=REPO/'vendor'/'official'/'src';output.mkdir(parents=True,exist_ok=True)
    commands=[['task1/build_task1_baseline_submission.py','--release-root',str(release.root),'--split',split,'--output',str(output/'state.csv')]]
    for p in panels:
        commands[0]+=['--panel',p]
    commands.append(['task2/build_task2_persistence_submission.py','--release-root',str(release.root),'--split',split,'--output',str(output/'queue.csv')])
    task4=['task4/build_task4_odme_artifacts.py','--release-root',str(release.root),'--split',split,'--output-root',str(output/'odme')]
    for p in panels:
        task4+=['--panel',p]
    commands.append(task4)
    if split=='train':
        score=['task1/score_task1.py','--release-root',str(release.root),'--split','train','--submission',str(output/'state.csv'),'--output',str(output/'state_official_score.csv')]
        for p in panels:
            score+=['--panel',p]
        commands.append(score)
    score4=['task4/score_task4.py','--release-root',str(release.root),'--split',split,'--submission',str(output/'odme'/'baseline_submission.csv'),'--output',str(output/'odme_link_score.csv')]
    for p in panels:
        score4+=['--panel',p]
    commands.append(score4)
    physics=['task3/score_task3.py','--release-root',str(release.root),'--split',split,
             '--state-submission',str(output/'state.csv'),'--output',str(output/'physics_diagnostic_NOT_FORMAL.csv')]
    for p in panels:
        physics+=['--panel',p]
    commands.append(physics)
    report=[]
    for cmd in commands:
        run=[sys.executable,str(src/cmd[0])]+cmd[1:]
        result=subprocess.run(run,capture_output=True,text=True)
        report.append({'command':run,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        print(result.stdout,flush=True)
        if result.returncode:
            dump(report,output/'official_runs.json')
            raise RuntimeError(result.stderr)
    report.append({'task3':'uses the same state.csv; formal score unknown, public diagnostic is not a leaderboard score',
                   'task2_score':None,'task4_total':None})
    dump(report,output/'official_runs.json')

def fit_final(release,panel,models_dir,iterations=250,max_rows=500000,seed=2026):
    from .experiments import groups,raw_for,training_rows
    from .state import Profile,StateModel
    from .queue import QueueModel,proxy_windows
    selection=json.loads((models_dir/panel/'experiment.json').read_text())['selection']
    partitions,_=groups(release,panel)
    profile=Profile.fit(raw_for(release,panel,partitions['profile']))
    paths=partitions['fit']+partitions['calibration']+partitions['test']
    method=selection['state_base']
    original=StateModel.load(models_dir/panel/selection.get('state_artifact',f'state_{method}.pkl'))
    model=StateModel(profile,method,seed,original.iterations,original.threads,
                     getattr(original,'feature_version',1),getattr(original,'num_leaves',31))
    model.density_blend=getattr(original,'density_blend',1.)
    if method in ['lightgbm','catboost','joint_lightgbm']:
        model.fit(*training_rows(release,panel,paths,profile,max_rows,seed,model.feature_version))
    model.save(models_dir/panel/'state_final.pkl')
    if not panel.startswith('D12_I405'):
        queue_method=selection['queue']
        original=QueueModel.load(models_dir/panel/selection.get('queue_artifact',f'queue_{queue_method}.pkl'))
        proxy=proxy_windows(raw_for(release,panel,paths),release.network(panel),release.topology(panel),panel)
        from copy import deepcopy
        qm=deepcopy(original);qm.model=None;qm.fit(proxy)
        qm.save(models_dir/panel/'queue_final.pkl')

def score_public(release,output,panels,split):
    from .experiments import raw_for,day_template,target_truth
    from .metrics import state_score,total_score,od_link_score
    from .odme import operator
    from .submission import validate_task
    state=None
    if split=='train':
        pred=csv(output/'state.csv')
        for c in ['speed_kmh','flow_vph']:
            pred[c]=pd.to_numeric(pred[c],errors='raise')
        truths,lanes=[],[]
        for panel in panels:
            template=release.template(1,panel,'train')
            for raw in raw_for(release,panel,release.files(panel,'train')):
                target=day_template(release,panel,'train',raw,template)
                if len(target):
                    truths.append(target_truth(release,panel,target,raw));lanes.append(release.station_lanes(panel,target))
        state=state_score(pd.concat(truths,ignore_index=True),pred,np.concatenate(lanes))
    path_pred=csv(output/'odme.csv');validate_task(path_pred,'odme')
    link_scores={}
    for panel in panels:
        A,c,b,paths=operator(release,panel,split)
        template=release.template(4,panel,split)
        sub=path_pred[(path_pred.panel==panel)&path_pred.departure_time.isin(template.departure_time)]
        if len(sub)!=len(template) or set(sub.path_id)!=set(paths.path_id):
            raise ValueError('Missing/extra OD paths')
        expected=template.set_index('path_id').reindex(sub.path_id)
        for zone in ['origin_zone','destination_zone']:
            if not np.array_equal(expected[zone].to_numpy(),sub[zone].to_numpy()):
                raise ValueError('Invalid OD zones in scored file')
        f=pd.to_numeric(sub.set_index('path_id').reindex(paths.path_id).path_flow).to_numpy(float)
        link_scores[panel]=od_link_score(A,f,c)
    values=pd.Series(link_scores)
    link=float(values.groupby(values.index.map(lambda s:s.rsplit('_',1)[0])).mean().mean())
    result={'split':split,'dataset':'fixture' if (release.root/'FIXTURE_ONLY.json').exists() else 'official_public',
            'Task1_train':state,'Task4_S_link':link,'Task4_panels':link_scores,
            'formal_score_contract':total_score({k:None for k in ['state','queue','physics','odme']}),
            'note':'Local train metric and link fit do not determine the leaderboard total.'}
    dump(result,output/f'public_score_{split}.json');return result

def predict(release,models_dir,output,panels,splits,baseline=False):
    from .state import StateModel,StateFeatures,Profile
    from .queue import QueueModel,predict_windows
    from .experiments import day_template
    from .physics import reconstruct,grids,refine,diagnose
    from .odme import predict as od_predict
    output.mkdir(parents=True,exist_ok=True)
    state_path=output/'state.csv';queue_path=output/'queue.csv';od_path=output/'odme.csv'
    if any(p.exists() for p in [state_path,queue_path,od_path]):
        raise FileExistsError('Prediction output must be a fresh directory')
    first={'state':True,'queue':True,'odme':True};report=[]
    # The master key decides which split/task combinations must be emitted.
    # A public package may also ship train templates that are not upload rows.
    required=set()
    for chunk in pd.read_csv(release.root/'submission_key.csv',usecols=['panel','split','task'],dtype=str,chunksize=100000,keep_default_na=False):
        required.update(chunk[['panel','split','task']].drop_duplicates().itertuples(index=False,name=None))
    for panel in panels:
        net,topo=release.network(panel),release.topology(panel)
        if baseline:
            profile=Profile.fit(release.read(p) for p in release.files(panel,'train',False))
            model=StateModel(profile,'historical');qm=QueueModel('persistence')
            selection={'physics_strength':0.,'fd_bounds':False,'odme':{'method':'ridge','regularization':.05}}
        else:
            model=StateModel.load(models_dir/panel/'state_final.pkl')
            qm=QueueModel.load(models_dir/panel/'queue_final.pkl') if not panel.startswith('D12_I405') else None
            selection=json.loads((models_dir/panel/'experiment.json').read_text())['selection']
        for split in splits:
            if not release.files(panel,split):
                continue
            if not any((panel,split,t) in required for t in ['state','queue','odme']):
                continue
            template=release.template(1,panel,split)
            written=0
            for path in (release.files(panel,split) if (panel,split,'state') in required else []):
                frame=release.read(path);target=day_template(release,panel,split,frame,template)
                if not len(target):
                    continue
                features=StateFeatures(frame,net,topo,model.profile,version=getattr(model,'feature_version',1))
                pred=model.predict(features,target)
                full,_=reconstruct(frame,pred);times,links,_,_=grids(full,net)
                ramps=release.ramps(panel,split,times,links)
                pred=refine(frame,pred,net,topo,ramps,selection['physics_strength'],selection['fd_bounds'])
                report.append({'panel':panel,'split':split,'date':frame.timestamp.iloc[0][:10],
                               'physics':diagnose(frame,pred,net,topo,ramps)})
                pred.to_csv(state_path,index=False,mode='w' if first['state'] else 'a',header=first['state']);first['state']=False;written+=len(pred)
            if (panel,split,'state') in required and written!=len(template):
                raise ValueError('Incomplete state prediction coverage')
            if not panel.startswith('D12_I405') and (panel,split,'queue') in required:
                q=predict_windows(release,panel,split,qm)
                q.to_csv(queue_path,index=False,mode='w' if first['queue'] else 'a',header=first['queue']);first['queue']=False
            if (panel,split,'odme') in required:
                o,info=od_predict(release,panel,split,method=selection['odme']['method'],regularization=selection['odme']['regularization'])
                o.to_csv(od_path,index=False,mode='w' if first['odme'] else 'a',header=first['odme']);first['odme']=False
                report.append({'panel':panel,'split':split,'odme':info})
    from .runtime import evidence,sha256_file
    model_files={str(p.relative_to(models_dir)):sha256_file(p) for p in models_dir.glob('*/**/*final.pkl')} if not baseline else {}
    dump({'diagnostics':report,'formal_scores':{'state':None,'queue':None,'physics':None,'odme':None},
          'runtime':evidence(),'model_sha256':model_files},output/'prediction_report.json')
    return {'state':state_path,'queue':queue_path,'odme':od_path}

def main():
    ap=argparse.ArgumentParser(description='Independent TrafficFlowBench research pipeline')
    ap.add_argument('command',choices=['validate','score','experiment','fit-final','predict','baseline','official-baselines','merge','fixture','run'])
    ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/run'))
    ap.add_argument('--models-dir',type=Path,default=Path('artifacts/experiments'))
    ap.add_argument('--panel',action='append')
    ap.add_argument('--split',choices=list(SPLITS)+['all'],default='all')
    ap.add_argument('--iterations',type=int,default=250)
    ap.add_argument('--max-rows',type=int,default=300000)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--days',type=int,default=42)
    ap.add_argument('--full-hash',action='store_true')
    args=ap.parse_args()
    if args.command=='fixture':
        from .fixture import make_fixture
        print(json.dumps(make_fixture(args.release_root,args.days,args.seed,args.panel)))
        return
    release=Release(args.release_root)
    panels=args.panel or release.panels()
    if not set(panels).issubset(release.panels()):
        raise ValueError('Unknown panel')
    splits=list(SPLITS) if args.split=='all' else [args.split]
    if args.command in ['validate','run']:
        validate(release,args.output/'release_manifest.json',args.full_hash)
    if args.command in ['experiment','run']:
        from .experiments import run_experiment
        for p in panels:
            run_experiment(release,p,args.models_dir/p,args.seed,args.iterations,args.max_rows)
    if args.command in ['fit-final','run']:
        for p in panels:
            fit_final(release,p,args.models_dir,args.iterations,args.max_rows,args.seed)
    if args.command in ['predict','baseline','run']:
        files=predict(release,args.models_dir,args.output,panels,splits,args.command=='baseline')
        if args.command=='run':
            from .submission import merge_files
            print(merge_files(files,release.root/'submission_key.csv',args.output/'submission.csv'))
    if args.command=='official-baselines':
        official_baselines(release,args.output,panels,'train' if args.split=='all' else args.split)
    if args.command=='score':
        print(score_public(release,args.output,panels,'train' if args.split=='all' else args.split))
    if args.command=='merge':
        from .submission import merge_files
        files={k:args.output/f'{k}.csv' for k in ['state','queue','odme']}
        print(merge_files(files,release.root/'submission_key.csv',args.output/'submission.csv'))

if __name__=='__main__':
    main()
