"""Frozen channel policy on its preregistered new public date block."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,VALUES
from trafficflow.state import StateModel,StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.experiments import partition_date,day_template,target_truth
from trafficflow.research_eval import register_protocol,paired_block_interval
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from recurrent_state_experiment import transition_masks,summarize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite channel forward evidence')
    study=json.loads(args.study.read_text());cfg=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256']:raise ValueError('Channel choice changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Frozen_channel_policy_forward_calendar',
        'source_study':args.study.as_posix(),'source_study_sha256':sha256_file(args.study),'dates':cfg['future_dates'],
        'date_origin':'Fixed before channel choice;all weights fixed before opening these outcomes',
        'scope':'Public research dates,possibly viewed by prior unrelated policies;not final/private',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    exports={**cfg['components'],'primary_export':study['primary_export']};models={}
    for role,row in exports.items():
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Frozen channel component changed')
        models[role]=StateModel.load(Path(row['path']))
    release=Release('data/kaggle_public');panel=cfg['panel'];net,topology=release.network(panel),release.topology(panel)
    anchor=models['nn_primary'];profile,version=anchor.profile,anchor.feature_version
    files={partition_date(release,p):p for p in release.files(panel,'train')};raw_files={p.name:p for p in release.files(panel,'train',False)};template=release.template(1,panel,'train')
    days=[];predictions={r:[] for r in models};outputs=[];source_hashes={};maximum_component_error=0.
    for date in cfg['future_dates']:
        path=files[date];raw_path=raw_files[path.name];frame=release.read(path);raw=release.read(raw_path)
        source_hashes[path.as_posix()]=sha256_file(path);source_hashes[raw_path.as_posix()]=sha256_file(raw_path)
        f=StateFeatures(frame,net,topology,profile,version);target=day_template(release,panel,'train',frame,template)
        teacher=target_truth(release,panel,target,raw)[VALUES].to_numpy(float);lanes=target.link_id.map(net.lanes).to_numpy(float);teacher[:,1]/=lanes
        indices=f.lookup(target);shape=(len(f.times),len(f.links));y=np.zeros((*shape,2),np.float32);mask=np.zeros(shape,bool)
        grouped=pd.DataFrame({'i':indices,'v':teacher[:,0],'q':teacher[:,1]}).groupby('i')[['v','q']].mean()
        y.reshape(-1,2)[grouped.index]=grouped.to_numpy()/np.array([25.,600.]);mask.ravel()[indices]=True
        day={'date':date,'regime':frame.mask_regime.iloc[0],'x':reconstruction_input(f),'y':y,'mask':mask,
            'station_truth':teacher,'target_indices':indices,'target_link_lanes':lanes,'station_lanes':release.station_lanes(panel,target)}
        day['diagnostic_masks']=transition_masks(day,target,raw);days.append(day);values={}
        for role,model in models.items():
            p=model.predict(f,target)[VALUES].to_numpy(float);p[:,1]/=lanes;values[role]=p;predictions[role].append(p)
        chosen=study['selection']['choice']
        if chosen:
            weight=np.array([chosen['speed_tree_weight'],chosen['flow_tree_weight']]);expected=(1-weight)*values['nn_primary']+weight*values[chosen['variant']+'_seed_mean']
        else:expected=values['nn_primary']
        error=float(np.abs(expected-values['primary_export']).max());maximum_component_error=max(maximum_component_error,error)
        if error>1e-8:raise ValueError('Serialized channel output differs from fixed component weights')
        output=folder/(date+'.npz');np.savez_compressed(output,truth_link_lane_units=teacher,target_grid_indices=indices,
            target_link_lanes=lanes,station_lanes=day['station_lanes'],**values,**{'mask_'+k:v for k,v in day['diagnostic_masks'].items()})
        outputs.append({'date':date,'path':output.as_posix(),'sha256':sha256_file(output)});print('channel_forward',date,'complete',flush=True)
    factor=(net.lanes*net.length_km).reindex(f.links).to_numpy(np.float32);metrics={r:summarize(days,p,factor) for r,p in predictions.items()}
    comparisons={r:{'pooled_State_gain':m['S_state']-metrics['nn_primary']['S_state'],
        'paired_day_diagnostic':paired_block_interval([v['daily_State'] for v in metrics['nn_primary']['days']],[v['daily_State'] for v in m['days']])} for r,m in metrics.items() if r!='nn_primary'}
    result={'registered':registered,'source_snapshot':snapshot,'metrics':metrics,'comparisons':comparisons,'prediction_files':outputs,
        'source_files_sha256':source_hashes,'exports':exports,'n_direct_role_day_predictor_calls':len(days)*len(models),
        'maximum_serialized_vs_component_error_lane_units':maximum_component_error,'selection_changed':False,
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Fixed State-only channel selection;all4roles directly inferred on14public research days. Per-channel dynamics may affect Physics,whose formal value is unknown.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('channel_forward_COMPLETE','NN',metrics['nn_primary']['S_state'],'selected',metrics['primary_export']['S_state'],
        'gain',comparisons['primary_export']['pooled_State_gain'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
