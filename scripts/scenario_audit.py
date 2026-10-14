"""Public-train speed-change strata; these are not true incident annotations."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from trafficflow.data import Release,eligible
from trafficflow.experiments import groups,raw_for,day_template,target_truth
from trafficflow.state import StateModel,StateFeatures
from trafficflow.physics import refine,reconstruct,grids
from trafficflow.metrics import state_score
from trafficflow.runtime import evidence

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--campaign',type=Path,default=Path('artifacts/campaign_v1'))
    ap.add_argument('--panel',default='D12_I5_N');ap.add_argument('--output',type=Path,default=Path('reports/official/scenario_D12_I5_N.json'))
    args=ap.parse_args();started=time.perf_counter();r=Release(args.release_root);p=args.panel
    folder=args.campaign/p;selection=json.loads((folder/'experiment.json').read_text())['selection']
    model=StateModel.load(folder/selection['state_artifact']);parts,dates=groups(r,p)
    net,topo=r.network(p),r.topology(p);template=r.template(1,p,'train')
    truths=[];preds=[];lanes=[];changes=[]
    for path,raw in zip(parts['test'],raw_for(r,p,parts['test'])):
        frame=r.read(path);target=day_template(r,p,'train',frame,template)
        if not len(target):continue
        f=StateFeatures(frame,net,topo,model.profile,version=model.feature_version);pred=model.predict(f,target)
        if selection['fd_bounds'] or selection['physics_strength']:
            full,_=reconstruct(frame,pred);times,links,_,_=grids(full,net)
            pred=refine(frame,pred,net,topo,r.ramps(p,'train',times,links),selection['physics_strength'],selection['fd_bounds'])
        truth=target_truth(r,p,target,raw)
        raw=raw.sort_values(['station_id','link_id','timestamp']).copy();raw.loc[~eligible(raw),'speed_kmh']=np.nan
        raw['drop_kmh']=raw.groupby(['station_id','link_id']).speed_kmh.shift()-raw.speed_kmh
        gap=pd.to_datetime(raw.timestamp,utc=True).groupby([raw.station_id,raw.link_id]).diff().dt.total_seconds()
        raw.loc[gap.ne(300),'drop_kmh']=np.nan
        delta=target[['timestamp','station_id','link_id']].merge(raw[['timestamp','station_id','link_id','drop_kmh']],
                   on=['timestamp','station_id','link_id'],validate='one_to_one').drop_kmh.to_numpy(float)
        truths.append(truth);preds.append(pred);lanes.append(r.station_lanes(p,target));changes.append(delta)
    truth=pd.concat(truths,ignore_index=True);pred=pd.concat(preds,ignore_index=True);lane=np.concatenate(lanes);delta=np.concatenate(changes)
    strata={'rapid_drop_15_to_30_kmh':(delta>=15)&(delta<30),'severe_drop_at_least_30_kmh':delta>=30,
            'recovery_at_least_15_kmh':delta<=-15,'stable_change_below_5_kmh':np.abs(delta)<5,
            'previous_observation_unavailable':~np.isfinite(delta)}
    out={'panel':p,'dates':dates['test'],'label_source':'public train observations, not true incidents',
         'selected':selection,'strata':{},'formal_scores':None,'seconds':None}
    for name,sel in strata.items():
        out['strata'][name]={'n':int(sel.sum()),'metrics':state_score(truth.loc[sel],pred,lane[sel]) if sel.any() else None}
    out['seconds']=time.perf_counter()-started;out['runtime']=evidence()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2))
    print({name:(row['n'],row['metrics']['S_state'] if row['metrics'] else None) for name,row in out['strata'].items()})

if __name__=='__main__':main()
