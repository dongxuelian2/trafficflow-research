"""Frozen first-round candidate suite, final refit and complete release predictions.

Independent child processes bound CPU and memory; no agent delegation or hidden
labels. Calibration selects candidates. Test is evaluated after that selection.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from trafficflow.data import Release
from trafficflow.experiments import groups,raw_for,training_rows,evaluate_state
from trafficflow.state import Profile,StateModel
from trafficflow.queue import QueueModel,proxy_windows
from trafficflow.metrics import grouped_iou,total_score
from trafficflow.odme import predict as od_predict,operator,count_holdout
from trafficflow.runtime import evidence
from trafficflow.cli import fit_final,predict

def write(obj,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def experiment(release,panel,folder,max_rows):
    started=time.perf_counter();initial=evidence();folder.mkdir(parents=True,exist_ok=True)
    parts,dates=groups(release,panel)
    profile=Profile.fit(raw_for(release,panel,parts['profile']))
    result={'panel':panel,'dataset':'official_public_train','dates':dates,'seed':2026,
      'validation':'chronological; calibration selects; test evaluated after selection',
      'state':{},'queue':{},'odme':{},'selection':{},'source_at_start':initial['source_sha256']}
    def save():write(result,folder/'experiment.json')
    models={}
    # Candidate families/parameters fixed using the D12 development experiment.
    for method in ['historical','interpolation','lightgbm']:
        t=time.perf_counter();model=StateModel(profile,method,iterations=400,feature_version=2,num_leaves=63)
        model.density_blend=0.
        if method=='lightgbm':
            model.fit(*training_rows(release,panel,parts['fit'],profile,max_rows,2026,2))
        model.save(folder/f'state_{method}.pkl');models[method]=model
        result['state'][method]={'calibration':evaluate_state(release,panel,parts['calibration'],model),
                                'seconds':time.perf_counter()-t}
        save();print(panel,method,result['state'][method]['calibration']['S_state'],flush=True)
    best=max(models,key=lambda name:result['state'][name]['calibration']['S_state'])
    # Geometry bounds are tested, without treating public physics as a formal score.
    t=time.perf_counter();tag=best+'_bounds'
    result['state'][tag]={'calibration':evaluate_state(release,panel,parts['calibration'],models[best],0.,True),
                          'seconds':time.perf_counter()-t}
    selected=max([best,tag],key=lambda name:result['state'][name]['calibration']['S_state'])
    result['state'][selected]['test']=evaluate_state(release,panel,parts['test'],models[best],0.,selected==tag)
    if best!='historical':
        result['state']['historical']['test']=evaluate_state(release,panel,parts['test'],models['historical'])
    result['selection'].update({'state_base':best,'state':selected,'state_artifact':f'state_{best}.pkl',
                               'physics_strength':0.,'fd_bounds':selected==tag})
    save();print(panel,'selected state',selected,'test',result['state'][selected]['test']['S_state'],flush=True)
    if not panel.startswith('D12_I405'):
        net,topo=release.network(panel),release.topology(panel)
        fit=proxy_windows(raw_for(release,panel,parts['fit']),net,topo,panel)
        cal=proxy_windows(raw_for(release,panel,parts['calibration']),net,topo,panel)
        qm={}
        for method in ['persistence','wave','lightgbm']:
            t=time.perf_counter();model=QueueModel(method,iterations=400).fit(fit)
            tuning=model.tune(cal) if method!='persistence' else {'selected_threshold':.5}
            result['queue'][method]={'calibration':grouped_iou(cal.y,model.probability(cal.x)>=model.threshold,cal.window,cal.condition),
              'tuning':tuning,'seconds':time.perf_counter()-t,'n_training_windows':len(np.unique(fit.window)),
              'n_calibration_windows':len(np.unique(cal.window))}
            model.save(folder/f'queue_{method}.pkl');qm[method]=model;save()
        selected_queue=max(qm,key=lambda name:result['queue'][name]['calibration']['proxy_IoU'])
        test=proxy_windows(raw_for(release,panel,parts['test']),net,topo,panel)
        for method in {selected_queue,'persistence'}:
            result['queue'][method]['test']=grouped_iou(test.y,qm[method].probability(test.x)>=qm[method].threshold,test.window,test.condition)
            result['queue'][method]['n_test_windows']=len(np.unique(test.window))
        result['selection'].update({'queue':selected_queue,'queue_artifact':f'queue_{selected_queue}.pkl'})
        print(panel,'selected queue',selected_queue,'test proxy',result['queue'][selected_queue]['test']['proxy_IoU'],flush=True)
        del fit,cal,test,qm
        save()
    A,c,b,_=operator(release,panel,'train')
    for method,reg in [('prior',0.),('ridge',.05),('ridge',.005),('relative',5.),('poisson',.05)]:
        t=time.perf_counter()
        try:
            _,info=od_predict(release,panel,'train',method,reg)
            info['sensor_holdout']=count_holdout(A,c,b,reg,2026,method)
        except RuntimeError as exc:
            info={'success':False,'error':str(exc),'S_ODME':None}
        info['seconds']=time.perf_counter()-t;result['odme'][f'{method}:{reg}']=info;save()
    result['selection']['odme']={'method':'ridge','regularization':.05,
         'reason':'Conservative prior regularization; link fit is not hidden OD accuracy.'}
    result['formal_score_contract']=total_score({k:None for k in ['state','queue','physics','odme']})
    result['local_score_contract']=total_score({'state':result['state'][selected]['test']['S_state'],'queue':None,'physics':None,'odme':None})
    result['seconds']=time.perf_counter()-started;result['runtime']=evidence();result['complete']=True;save()

def child(args):
    release=Release(args.release_root);folder=args.output/args.child_panel
    completed=folder/'experiment.json'
    if not completed.exists() or not json.loads(completed.read_text()).get('complete'):
        experiment(release,args.child_panel,folder,args.max_rows)
    if not (folder/'refit.json').exists():
        started=time.perf_counter();fit_final(release,args.child_panel,args.output,max_rows=args.max_rows)
        write({'seconds':time.perf_counter()-started,'runtime':evidence()},folder/'refit.json')
        print(args.child_panel,'final refit complete',flush=True)
    output=folder/'predictions'
    if not (output/'prediction_report.json').exists():
        started=time.perf_counter();predict(release,args.output,output,[args.child_panel],['validation','private'])
        write({'seconds':time.perf_counter()-started,'runtime':evidence()},folder/'prediction_runtime.json')
        print(args.child_panel,'prediction complete',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/campaign'))
    ap.add_argument('--workers',type=int,default=2);ap.add_argument('--max-rows',type=int,default=300000)
    ap.add_argument('--panel',action='append');ap.add_argument('--child-panel')
    args=ap.parse_args()
    if args.child_panel:child(args);return
    if not 1<=args.workers<=4:raise ValueError('Use 1 to 4 workers')
    panels=args.panel or Release(args.release_root).panels();args.output.mkdir(parents=True,exist_ok=True)
    def run(panel):
        folder=args.output/panel;folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
        command=[sys.executable,str(Path(__file__).resolve()),'--release-root',str(args.release_root.resolve()),
                 '--output',str(args.output.resolve()),'--child-panel',panel,'--max-rows',str(args.max_rows)]
        env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
        with (folder/'campaign.log').open('a',encoding='utf-8') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        return {'panel':panel,'returncode':proc.returncode,'seconds':time.perf_counter()-started,'command':command}
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,p) for p in panels]):
            record=future.result();records.append(record);write(records,args.output/'campaign.json');print(record,flush=True)
    if any(r['returncode'] for r in records):raise RuntimeError('Campaign failure; inspect panel logs')

if __name__=='__main__':main()
