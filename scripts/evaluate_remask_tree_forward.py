"""Independent new calendar for frozen trees and cal-selected tree/NN hybrid."""
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
from recurrent_state_experiment import summarize,transition_masks
from remask_tree_roles import context,load_tree_roles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite tree forward evidence')
    study,cfg,source,views,original=context(args.study)
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Remask_trees_frozen_new_calendar',
        'source_study':args.study.as_posix(),'source_study_sha256':sha256_file(args.study),'dates':cfg['future_dates'],
        'calendar_origin':'Fixed before tree fitting;no model/alpha reselection','primary':study['selection']['primary'],
        'scope':'Public research calendar,not final/private. All reference/negative policies reported.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    paths,models=load_tree_roles(args.study,folder/'exports');primary=models.pop('primary_export')
    release=Release('data/kaggle_public');panel=cfg['panel'];net,topology=release.network(panel),release.topology(panel)
    profile=models['nn_primary'].profile;version=models['nn_primary'].feature_version
    files={partition_date(release,p):p for p in release.files(panel,'train')};raw_files={p.name:p for p in release.files(panel,'train',False)};template=release.template(1,panel,'train')
    days=[];predictions={};outputs=[];hashes={};direct_primary=[]
    for date in cfg['future_dates']:
        path=files[date];raw_path=raw_files[path.name];frame=release.read(path);raw=release.read(raw_path)
        hashes[path.as_posix()]=sha256_file(path);hashes[raw_path.as_posix()]=sha256_file(raw_path)
        f=StateFeatures(frame,net,topology,profile,version);target=day_template(release,panel,'train',frame,template)
        truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float);lanes=target.link_id.map(net.lanes).to_numpy(float);truth[:,1]/=lanes
        indices=f.lookup(target);shape=(len(f.times),len(f.links));y=np.zeros((*shape,2),np.float32);mask=np.zeros(shape,bool)
        grouped=pd.DataFrame({'i':indices,'v':truth[:,0],'q':truth[:,1]}).groupby('i')[['v','q']].mean();y.reshape(-1,2)[grouped.index]=grouped.to_numpy()/np.array([25.,600.]);mask.ravel()[indices]=True
        day={'date':date,'regime':frame.mask_regime.iloc[0],'x':reconstruction_input(f),'y':y,'mask':mask,'station_truth':truth,
            'target_indices':indices,'target_link_lanes':lanes,'station_lanes':release.station_lanes(panel,target)}
        day['diagnostic_masks']=transition_masks(day,target,raw);days.append(day);values={}
        for role,model in models.items():
            p=model.predict(f,target)[VALUES].to_numpy(float);p[:,1]/=lanes;values[role]=p
        for variant in ['point','queue']:
            for alpha in [.25,.5,.75]:values[f'{variant}_hybrid_{alpha}']=alpha*values[variant+'_seed_mean']+(1-alpha)*values['nn_primary']
        values['primary_export']=values[study['selection']['primary']]
        if date in [cfg['future_dates'][0],cfg['future_dates'][-1]]:
            p=primary.predict(f,target)[VALUES].to_numpy(float);p[:,1]/=lanes;error=float(np.abs(p-values['primary_export']).max())
            if error>1e-8:raise ValueError('Serialized selected policy differs from derived forward prediction')
            direct_primary.append({'date':date,'maximum_lane_unit_prediction_error':error})
        for role,p in values.items():predictions.setdefault(role,[]).append(p)
        out=folder/(date+'.npz');np.savez_compressed(out,truth_link_lane_units=truth,target_grid_indices=indices,
            target_link_lanes=lanes,station_lanes=day['station_lanes'],**values,**{'mask_'+k:v for k,v in day['diagnostic_masks'].items()})
        outputs.append({'date':date,'path':out.as_posix(),'sha256':sha256_file(out)});print('remask_tree_forward',date,'complete',flush=True)
    factor=(net.lanes*net.length_km).reindex(f.links).to_numpy(np.float32);metrics={r:summarize(days,p,factor) for r,p in predictions.items()}
    comparisons={r:{'pooled_State_gain_vs_NN':m['S_state']-metrics['nn_primary']['S_state'],
        'paired_day_diagnostic':paired_block_interval([v['daily_State'] for v in metrics['nn_primary']['days']],[v['daily_State'] for v in m['days']])} for r,m in metrics.items()}
    result={'registered':registered,'source_snapshot':snapshot,'metrics':metrics,'comparisons':comparisons,'prediction_files':outputs,
        'exports':{r:{'path':p.as_posix(),'sha256':sha256_file(p)} for r,p in paths.items()},'source_files_sha256':hashes,
        'direct_primary_replays':direct_primary,'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'choices_changed':False,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Candidate hybrids derived from frozen predictor arrays;selected serialized hybrid directly replayed on2fixed dates. '
        'Public research calendar may have appeared in older studies. Diagnostic increments are incomplete Physics proxies,not formal score.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('remask_tree_forward_COMPLETE',study['selection']['primary'],'selected',metrics['primary_export']['S_state'],'NN',metrics['nn_primary']['S_state'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
