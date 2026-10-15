"""All policies frozen before the registered new public calendar is opened."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,VALUES
from trafficflow.state import StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.experiments import partition_date,day_template,target_truth
from trafficflow.research_eval import register_protocol,paired_block_interval
from trafficflow.runtime import capture_source_snapshot,sha256_file,evidence
from native_remask_roles import load_roles
from recurrent_state_experiment import transition_masks,summarize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite native forward evidence')
    study=json.loads(args.study.read_text());config=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256'] or json.loads((source/'selection_sealed.json').read_text())!=study['selection']:
        raise ValueError('Choices changed or not sealed')
    dates=config['future_dates'];original=json.loads(Path(config['source_study']).read_text())
    if set(dates)&{d for values in original['prepared']['dates'].values() for d in values}:raise ValueError('Forward dates overlap original model calendar')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Native_remask_frozen_policy_forward_public_calendar','source_study':args.study.as_posix(),
        'source_study_sha256':sha256_file(args.study),'selection_seal_sha256':study['selection_seal_sha256'],
        'dates':dates,'source_of_dates':'Training preregistration,before new model fit','selection':'No checkpoint,alpha,primary or date reselection',
        'scope':'Official masked TRAIN/template targets,new calendar for these models;previously viewed public dates by other studies possible',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    paths,models=load_roles(args.study,folder/'State_primary.pkl');roles=list(models)
    release=Release('data/kaggle_public');panel=config['panel'];net,topology=release.network(panel),release.topology(panel)
    profile=models['incumbent'].profile;version=models['incumbent'].feature_version
    if any(m.feature_version!=version or not m.profile.slots.equals(profile.slots) or not m.profile.fallback.equals(profile.fallback) for m in models.values()):
        raise ValueError('Policies have different feature profiles')
    files={partition_date(release,p):p for p in release.files(panel,'train')};raw_files={p.name:p for p in release.files(panel,'train',False)}
    template=release.template(1,panel,'train');days=[];predictions={r:[] for r in roles};source_hashes={};prediction_files=[]
    for date in dates:
        path=files[date];raw_path=raw_files[path.name];frame=release.read(path);raw=release.read(raw_path)
        source_hashes[path.as_posix()]=sha256_file(path);source_hashes[raw_path.as_posix()]=sha256_file(raw_path)
        features=StateFeatures(frame,net,topology,profile,version);target=day_template(release,panel,'train',frame,template)
        teacher=target_truth(release,panel,target,raw)[VALUES].to_numpy(float);lane=target.link_id.map(net.lanes).to_numpy(float);teacher[:,1]/=lane
        indices=features.lookup(target);shape=(len(features.times),len(features.links));y=np.zeros((*shape,2),np.float32);mask=np.zeros(shape,bool)
        grouped=pd.DataFrame({'i':indices,'v':teacher[:,0],'q':teacher[:,1]}).groupby('i')[['v','q']].mean()
        y.reshape(-1,2)[grouped.index]=grouped.to_numpy()/np.array([25.,600.]);mask.ravel()[indices]=True
        day={'date':date,'regime':frame.mask_regime.iloc[0],'x':reconstruction_input(features),'y':y,'mask':mask,
            'station_truth':teacher,'target_indices':indices,'target_link_lanes':lane,'station_lanes':release.station_lanes(panel,target)}
        day['diagnostic_masks']=transition_masks(day,target,raw);days.append(day);values={}
        for role,model in models.items():
            p=model.predict(features,target)[VALUES].to_numpy(float);p[:,1]/=lane;predictions[role].append(p);values[role]=p
        output=folder/(date+'.npz');np.savez_compressed(output,truth_link_lane_units=teacher,station_lanes=day['station_lanes'],target_link_lanes=lane,
            target_grid_indices=indices,**values,**{'mask_'+k:v for k,v in day['diagnostic_masks'].items()})
        prediction_files.append({'date':date,'path':output.as_posix(),'sha256':sha256_file(output)})
        print('native_forward',date,'complete',flush=True)
    factor=(net.lanes*net.length_km).reindex(features.links).to_numpy(np.float32)
    metrics={r:summarize(days,p,factor) for r,p in predictions.items()}
    pairs=[('prior_GRU','point_seed_mean'),('prior_GRU','queue_seed_mean'),('point_seed_mean','queue_seed_mean'),('prior_GRU','primary_export')]+[
        (f'seed{s}_control',f'seed{s}_{v}') for s in config['seeds'] for v in ['point','queue']]
    comparisons={b+'_vs_'+a:{'pooled_State_gain':metrics[b]['S_state']-metrics[a]['S_state'],
        'paired_day_diagnostic':paired_block_interval([r['daily_State'] for r in metrics[a]['days']],[r['daily_State'] for r in metrics[b]['days']])} for a,b in pairs}
    result={'registered':registered,'source_snapshot':snapshot,'metrics':metrics,'comparisons':comparisons,
        'exports':{n:{'path':p.as_posix(),'sha256':sha256_file(p)} for n,p in paths.items()},'prediction_files':prediction_files,
        'source_files_sha256':source_hashes,'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
        'selection_changed':False,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Public research holdout for these policies,not final/private. Two optimizer seeds share data. '
        'Diagnostics use raw observations,e.g.speed changes,not latent accident or Queue truth. Legacy increment proxy is not formal Physics.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('native_forward_COMPLETE',{r:metrics[r]['S_state'] for r in ['prior_GRU','point_seed_mean','queue_seed_mean','primary_export']},'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
