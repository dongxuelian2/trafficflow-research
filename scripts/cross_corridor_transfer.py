"""Zero-shot graph transfer with no target-family train labels in model input.

Target labels are read only for the fixed assessment. Target historical profiles
are replaced by geometry-only defaults; no target-family calibration is used.
"""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release, VALUES, eligible
from trafficflow.experiments import partition_date, raw_for, day_template, target_truth
from trafficflow.state import Profile, StateFeatures, neighbors
from trafficflow.graph_state import GraphState
from trafficflow.research_eval import Fold, state_from_sufficient_stats, paired_block_interval, register_protocol
from train_graph_experiment import predict, stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('artifacts/research/A01_graph_run01'))
    parser.add_argument('--target',default='D7_I10_W')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Transfer evidence must not be overwritten')
    source=json.loads((args.source/'experiment.json').read_text())
    source_panel=source['prepared']['panel']
    if args.target.rsplit('_',1)[0]==source_panel.rsplit('_',1)[0]:
        raise ValueError('Transfer target must be an excluded corridor family')
    registered=register_protocol(args.output.with_suffix('.registered.json'),
        {'source_model':str(args.source),'source_panel':source_panel,'target_family_excluded':args.target.rsplit('_',1)[0],
        'fold':source['prepared']['fold'],'target_supervised_labels_used':False,
        'target_profile':'geometry-only free speed/.4 capacity',
        'models':['interpolation_geometry_profile','graph_zero_shot','fixed_half_graph_half_interpolation'],
        'target_calibration':False,'ensemble_alpha':.5})
    started=time.perf_counter()
    release=Release('data/kaggle_public')
    protocol=json.loads(Path('reports/research/protocol_v2_registered.json').read_text())
    fold=Fold(**next(f for f in protocol['protocol']['folds'] if f['name']==source['prepared']['fold']))
    paths=release.files(args.target,'train')
    dates=fold.assign([partition_date(release,p) for p in paths])
    paths=[p for p in paths if partition_date(release,p) in dates['assessment']]
    net,topo=release.network(args.target),release.topology(args.target)
    empty=pd.DataFrame(columns=VALUES,index=pd.MultiIndex.from_arrays([[],[]],names=['link_id','slot']),dtype=float)
    defaults=pd.DataFrame({'speed_kmh':net.free_speed_kmh,'flow_vph':.4*net.capacity_vph})
    profile=Profile(empty,defaults)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model=GraphState(graph=source['selected']['name']!='temporal').to(device)
    model.load_state_dict(torch.load(args.source/(source['selected']['name']+'.pt'),map_location=device,weights_only=True))
    model.eval()
    rows={name:[] for name in registered['protocol']['models']}
    template=release.template(1,args.target,'train')
    for path,raw in zip(paths,raw_for(release,args.target,paths)):
        frame=release.read(path)
        f=StateFeatures(frame,net,topo,profile,version=2)
        shape=(len(f.times),len(f.links))
        raw_channels=f.raw/np.array([25.,600.])
        known=frame.copy()
        known.loc[~eligible(frame),'occupancy']=np.nan
        occupancy=known.pivot_table(index='timestamp',columns='link_id',values='occupancy',aggfunc='mean').reindex(index=f.times,columns=f.links).to_numpy(float)
        dt=pd.to_datetime(f.times,utc=True)
        angle=2*np.pi*(dt.hour*60+dt.minute)/1440
        sin=np.broadcast_to(np.asarray(np.sin(angle))[:,None,None],(*shape,1))
        cos=np.broadcast_to(np.asarray(np.cos(angle))[:,None,None],(*shape,1))
        x=np.concatenate([np.nan_to_num(raw_channels),np.isfinite(raw_channels),f.prior/np.array([25.,600.]),
            f.base/np.array([25.,600.]),np.nan_to_num(occupancy[...,None])/.25,np.isfinite(occupancy[...,None]),sin,cos],axis=-1).astype(np.float32)
        adjacency=[]
        for direction in ('incoming','outgoing'):
            a=np.zeros((len(f.links),len(f.links)),np.float32)
            for j,connected in enumerate(neighbors(topo,f.links,direction)):
                if connected:a[j,connected]=1/len(connected)
            adjacency.append(torch.tensor(a,device=device))
        lanes=net.lanes.reindex(f.links).to_numpy(float)
        fd=np.stack([net.free_speed_kmh.reindex(f.links),net.capacity_vph.reindex(f.links)/lanes,
                     net.k_jam.reindex(f.links)/lanes]).astype(np.float32)
        graph=adjacency+[torch.tensor(fd,device=device)]
        target=day_template(release,args.target,'train',frame,template)
        truth=target_truth(release,args.target,target,raw)[VALUES].to_numpy(float)
        truth[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
        indices=f.lookup(target)
        interpolation=f.base.reshape(-1,2)[indices].copy()
        interpolation[:,0]=np.minimum(interpolation[:,0],target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
        interpolation[:,1]=np.minimum(interpolation[:,1],target.link_id.map(net.capacity_vph/net.lanes).to_numpy(float)*1.05)
        day={'x':x,'target_indices':indices,'station_truth':truth,
             'target_link_lanes':target.link_id.map(net.lanes).to_numpy(float),
             'station_lanes':release.station_lanes(args.target,target),'regime':target.mask_regime.iloc[0]}
        neural=predict(model,day,graph,device)
        predictions=[interpolation,neural,.5*(interpolation+neural)]
        for name,prediction in zip(rows,predictions):
            rows[name].append(stats(day,prediction))
    result={'source_family':source_panel.rsplit('_',1)[0],'excluded_target_family':args.target.rsplit('_',1)[0],
        'target_panel':args.target,'target_train_labels_in_model_inputs':False,'dates':dates['assessment'],
        'registered':registered,'models':{name:state_from_sufficient_stats(r) for name,r in rows.items()},
        'paired_daily_gain':paired_block_interval([r['daily_State'] for r in rows['interpolation_geometry_profile']],
             [r['daily_State'] for r in rows['fixed_half_graph_half_interpolation']]),
        'seconds':time.perf_counter()-started,'formal_State':None,
        'limitation':'Single source family and target direction; target network and current masked observations are allowed. No leave-one-family-out multi-family training yet.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({name:r['S_state'] for name,r in result['models'].items()},flush=True)


if __name__=='__main__':main()
