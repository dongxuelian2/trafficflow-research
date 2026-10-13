"""A second, explicitly experimental candidate with weak conservation.

Calibration bounds State loss to 0.004 and minimizes observable conservation
RMS. This is a Pareto heuristic, NOT optimization of a formal physics/total
score. Original State predictions remain available beside this candidate.
"""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from trafficflow.data import Release,VALUES
from trafficflow.experiments import groups,raw_for,day_template,target_truth
from trafficflow.state import StateModel,StateFeatures
from trafficflow.physics import reconstruct,grids,refine,diagnose
from trafficflow.runtime import evidence

def save(d,path):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,indent=2,allow_nan=False))

class Stats:
    def __init__(self):
        self.regimes={r:np.zeros(3) for r in ['R1','R2','R3']};self.diagnostics=[]
    def add(self,truth,pred,lanes,diagnostic):
        if not truth[['panel','timestamp','station_id','link_id','mask_regime']].reset_index(drop=True).equals(pred[['panel','timestamp','station_id','link_id','mask_regime']].reset_index(drop=True)):
            raise ValueError('Physics candidate lost ordered target keys')
        y=truth[VALUES].to_numpy(float);p=pred[VALUES].to_numpy(float)
        if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid candidate values')
        errors=(p-y)**2;errors[:,1]/=lanes**2
        for r in self.regimes:
            sel=truth.mask_regime.eq(r).to_numpy();self.regimes[r]+=np.r_[sel.sum(),errors[sel].sum(axis=0)]
        self.diagnostics.append(diagnostic)
    def result(self):
        rows=[]
        for r,(n,ev,eq) in self.regimes.items():
            rv,rq=np.sqrt([ev,eq]/n) if n else [25.,600.]
            rows.append({'regime':r,'n':int(n),'rmse_speed_kmh':float(rv),'rmse_flow_per_lane':float(rq),
                         'S_state':.54*max(0,1-rv/25)+.46*max(0,1-rq/600)})
        n=sum(d['observable_transitions'] for d in self.diagnostics)
        rms=np.sqrt(sum(d['observable_transitions']*d['conservation_RMS_vehicles']**2 for d in self.diagnostics if d['conservation_RMS_vehicles'] is not None)/n) if n else None
        return {'S_state':float(np.mean([r['S_state'] for r in rows])),'details':rows,
          'observable_conservation_RMS_vehicles':float(rms) if rms is not None else None,'observable_transitions':n,
          'possible_transitions':sum(d['possible_transitions'] for d in self.diagnostics),
          'FD_absolute_residual_per_lane_mean_days':float(np.mean([d['FD_mean_absolute_residual_per_lane'] for d in self.diagnostics if d['FD_mean_absolute_residual_per_lane'] is not None])),
          'formal_S_physics':None,'formal_S_total':None}

def evaluate(r,p,paths,model,strengths,bounds):
    net,topo=r.network(p),r.topology(p);template=r.template(1,p,'train');stats={s:Stats() for s in strengths};errors={}
    for path,raw in zip(paths,raw_for(r,p,paths)):
        frame=r.read(path);target=day_template(r,p,'train',frame,template)
        if not len(target):continue
        features=StateFeatures(frame,net,topo,model.profile,version=model.feature_version);original=model.predict(features,target)
        full,_=reconstruct(frame,original);times,links,_,_=grids(full,net);ramps=r.ramps(p,'train',times,links)
        truth=target_truth(r,p,target,raw);lanes=r.station_lanes(p,target)
        for s in strengths:
            if s in errors:continue
            try:
                pred=refine(frame,original,net,topo,ramps,s,bounds)
                stats[s].add(truth,pred,lanes,diagnose(frame,pred,net,topo,ramps))
            except RuntimeError as exc:errors[s]=str(exc)
    return {str(s):({'success':False,'error':errors[s]} if s in errors else {'success':True,**stats[s].result()}) for s in strengths}

def child(args):
    started=time.perf_counter();r=Release(args.release_root);p=args.child_panel;folder=args.output/p;folder.mkdir(parents=True,exist_ok=True)
    src=args.campaign/p;d=json.loads((src/'experiment.json').read_text());sel=d['selection'];parts,dates=groups(r,p)
    checkpoint=folder/'research.json'
    if checkpoint.exists():out=json.loads(checkpoint.read_text())
    else:
        model=StateModel.load(src/sel['state_artifact'])
        for tree in model.models:tree.set_params(n_jobs=2)
        cal=evaluate(r,p,parts['calibration'],model,[0.,.00001,.0001,.001],sel['fd_bounds']);base=cal['0.0']
        expected=d['state'][sel['state']]['calibration']['S_state']
        if abs(base['S_state']-expected)>1e-10:raise ValueError('Streaming State formula differs from verified reference')
        candidates=[s for s,row in cal.items() if row['success'] and row['S_state']>=base['S_state']-.004 and row['observable_conservation_RMS_vehicles'] is not None]
        selected=min(candidates,key=lambda s:cal[s]['observable_conservation_RMS_vehicles']) if candidates else '0.0'
        test=evaluate(r,p,parts['test'],model,sorted({0.,float(selected)}),sel['fd_bounds'])
        out={'panel':p,'calibration':cal,'test':test,'selected_strength':float(selected),'dates':dates,
             'selection':'min observable RMS subject to calibration State loss <=0.004; incomplete physics heuristic',
             'formal_S_physics':None,'formal_S_total':None,'seconds_research':time.perf_counter()-started,'runtime':evidence()}
        out['runtime']['inference_threads']=2;save(out,checkpoint)
        print(p,'physics strength',selected,'test',test[selected]['S_state'],flush=True)
    if (folder/'production.json').exists():return
    if not (src/'predictions'/'prediction_report.json').exists():raise RuntimeError('Base production predictions not yet complete; rerun after campaign')
    original=pd.read_csv(src/'predictions'/'state.csv',dtype={c:str for c in ['panel','timestamp','station_id','link_id','mask_regime']})
    net,topo=r.network(p),r.topology(p);first=True;written=0;diagnostics=[];production_start=time.perf_counter()
    for split in ['validation','private']:
        for path in r.files(p,split):
            frame=r.read(path);pred=original[original.timestamp.isin(frame.timestamp.unique())].copy()
            if not len(pred):continue
            full,_=reconstruct(frame,pred);times,links,_,_=grids(full,net);ramps=r.ramps(p,split,times,links)
            try:
                adjusted=refine(frame,pred,net,topo,ramps,out['selected_strength'],sel['fd_bounds']);failed=None
            except RuntimeError as exc:adjusted=pred;failed=str(exc)
            diagnostics.append({'split':split,'date':frame.timestamp.iloc[0][:10],'solver_fallback':failed,
                                 'physics':diagnose(frame,adjusted,net,topo,ramps)})
            adjusted.to_csv(folder/'state.csv',index=False,mode='w' if first else 'a',header=first);first=False;written+=len(adjusted)
    if written!=len(original):raise ValueError('Incomplete candidate production')
    save({'rows':written,'selected_strength':out['selected_strength'],'diagnostics':diagnostics,
          'seconds':time.perf_counter()-production_start,'formal_S_physics':None},folder/'production.json')
    print(p,'physics candidate prediction complete',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--campaign',type=Path,default=Path('artifacts/campaign_v1'));ap.add_argument('--output',type=Path,default=Path('artifacts/physics_tradeoff'))
    ap.add_argument('--panel',action='append');ap.add_argument('--child-panel');ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    if args.child_panel:child(args);return
    panels=args.panel or Release(args.release_root).panels();args.output.mkdir(parents=True,exist_ok=True)
    def run(p):
        folder=args.output/p;folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
        cmd=[sys.executable,str(Path(__file__).resolve()),'--release-root',str(args.release_root.resolve()),'--campaign',str(args.campaign.resolve()),
             '--output',str(args.output.resolve()),'--child-panel',p]
        env=os.environ.copy();env.update({'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        with (folder/'run.log').open('a',encoding='utf-8') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        return {'panel':p,'returncode':proc.returncode,'seconds':time.perf_counter()-started,'command':cmd}
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,p) for p in panels]):
            record=future.result();records.append(record);save(records,args.output/'suite.json');print(record,flush=True)
    if any(r['returncode'] for r in records):raise RuntimeError('Physics candidate incomplete; inspect logs and rerun after campaign if necessary')

if __name__=='__main__':main()
