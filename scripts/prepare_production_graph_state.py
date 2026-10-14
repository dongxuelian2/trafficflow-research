"""Prepare a full-public-train State candidate without touching formal best.

This preparation has no holdout performance claim. Its future architecture and
blend must come from independently sealed development studies.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release,VALUES
from trafficflow.state import StateModel,StateFeatures,neighbors
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.experiments import groups,raw_for,training_rows,day_template,target_truth,partition_date
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot,evidence,sha256_file


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261013)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite production preparation')
    baseline_path=Path('artifacts/best_models')/args.panel/'state_final.pkl'
    original=StateModel.load(baseline_path)
    report=Path('reports/research')/(args.output.name+'.json')
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'Production_State_full_public_train_preparation','panel':args.panel,'seed':args.seed,
        'baseline_file':baseline_path.as_posix(),'baseline_sha256':sha256_file(baseline_path),
        'profile':'Preserve the frozen formal model profile; public-train prefix only',
        'fit_dates':'All public train dates after the preserved profile prefix',
        'large_max_labels':10000000,'model':'LGB 400/63, same version-2 features; public train labels only',
        'neural_architecture_selection':'Not selected here. Future architecture/alpha must come from sealed development studies.',
        'inference_scope':'Offline State only; no future information supplied to Queue',
        'formal_Total':None,'formal_State':None,'formal_Physics':None,'uploaded':False})
    args.output.mkdir(parents=True)
    snapshot=capture_source_snapshot(args.output/'source_snapshot.zip')
    start=time.perf_counter()
    release=Release('data/kaggle_public')
    partitions,dates=groups(release,args.panel)
    paths=partitions['fit']+partitions['calibration']+partitions['test']
    all_dates=[partition_date(release,p) for p in paths]
    profile=original.profile
    model=StateModel(profile,'lightgbm',seed=args.seed,iterations=400,feature_version=2,num_leaves=63)
    x,y,base=training_rows(release,args.panel,paths,profile,10000000,args.seed,2)
    n=len(y)
    # Per-day cap must cover all required targets for a true full-label control.
    train_template=release.template(1,args.panel,'train')
    required=int(train_template.timestamp.str[:10].isin(all_dates).sum())
    if n!=required:raise ValueError('Large budget failed to cover every public-train fitting target')
    model.fit(x,y,base);model.save(args.output/'lightgbm_full_public_train.pkl')
    del x,y,base
    net,topo=release.network(args.panel),release.topology(args.panel)
    links=None;sources=[]
    for path,raw in zip(paths,raw_for(release,args.panel,paths)):
        frame=release.read(path)
        f=StateFeatures(frame,net,topo,profile,version=2)
        if links is None:links=f.links
        elif links!=f.links:raise ValueError('Production graph order changed')
        target=day_template(release,args.panel,'train',frame,train_template)
        truth=target_truth(release,args.panel,target,raw)
        idx=f.lookup(target)
        value=truth[VALUES].to_numpy(float,copy=True)
        lane=target.link_id.map(net.lanes).to_numpy(float)
        value[:,1]/=lane
        shape=f.raw.shape[:2]
        y=np.zeros((*shape,2),np.float32);mask=np.zeros(shape,bool)
        averaged=pd.DataFrame({'i':idx,'v':value[:,0],'q':value[:,1]}).groupby('i')[['v','q']].mean()
        y.reshape(-1,2)[averaged.index]=averaged.to_numpy()/np.array([25.,600.])
        mask.ravel()[idx]=True
        output=args.output/'fit'/(f.times[0][:10]+'.npz')
        output.parent.mkdir(exist_ok=True)
        np.savez_compressed(output,x=reconstruction_input(f),y=y,mask=mask,
            target_indices=idx,station_truth=value,station_lanes=release.station_lanes(args.panel,target),
            target_link_lanes=lane,regime=target.mask_regime.iloc[0],links=np.array(links),times=np.array(f.times))
        sources.append({'date':f.times[0][:10],'masked_path':path.as_posix(),'cache_sha256':sha256_file(output)})
    adjacency=[]
    for direction in ('incoming','outgoing'):
        a=np.zeros((len(links),len(links)),np.float32)
        for j,connected in enumerate(neighbors(topo,links,direction)):
            if connected:a[j,connected]=1/len(connected)
        adjacency.append(a)
    lane=net.lanes.reindex(links).to_numpy(float)
    fd=np.stack([net.free_speed_kmh.reindex(links),net.capacity_vph.reindex(links)/lane,
        net.k_jam.reindex(links)/lane]).astype(np.float32)
    np.savez(args.output/'graph.npz',incoming=adjacency[0],outgoing=adjacency[1],fd=fd)
    result={'registered':registered,'source_snapshot':snapshot,'panel':args.panel,'fit_dates':all_dates,
        'profile_dates':dates['profile'],'n_fitting_labels':n,'n_grid_fit_dates':len(sources),'sources':sources,
        'large_State_model_sha256':sha256_file(args.output/'lightgbm_full_public_train.pkl'),
        'graph_sha256':sha256_file(args.output/'graph.npz'),'seconds':time.perf_counter()-start,
        'source':evidence(),'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'training_fit_score':None,'uploaded':False,
        'limitations':'Production preparation reuses all public train labels, including earlier development assessments. No holdout score is claimed. Neural grid targets average colocated stations; scoring retains station-level truth.'}
    (args.output/'prepared.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(args.panel,n,len(sources),result['seconds'],flush=True)
