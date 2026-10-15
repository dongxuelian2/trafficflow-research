"""Preregistered forward-calendar evaluation,with every original choice frozen."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,VALUES,STATE_KEY
from trafficflow.state import StateModel,StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.experiments import partition_date,day_template,raw_for,target_truth
from trafficflow.research_eval import paired_block_interval
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from recurrent_state_experiment import transition_masks,summarize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite forward assessment')
    registered=json.loads(args.output.with_suffix('.registered.json').read_text());protocol=registered['protocol']
    dates=pd.date_range(protocol['start_date'],protocol['end_date'],freq='D').strftime('%Y-%m-%d').tolist()
    sources=[]
    # Validate both original experiments before reading any extension labels.
    for name,replay_name in zip(protocol['studies'],protocol['export_reports']):
        study_path=Path('reports/research')/(name+'.json');study=json.loads(study_path.read_text())
        replay_path=Path('reports/research')/(replay_name+'.json');replay=json.loads(replay_path.read_text())
        directory=Path('artifacts/research')/name
        if sha256_file(directory/'selection_sealed.json')!=study['selection_seal_sha256']:raise ValueError('Original seal changed')
        if replay['registered']['protocol']['source_study_sha256']!=sha256_file(study_path):raise ValueError('Wrong exported source')
        if set(dates)&{d for group in study['prepared']['dates'].values() for d in group}:raise ValueError('Extension overlaps model calendar')
        if not replay['all_predictions_within_absolute_tolerance']:raise ValueError('Original model deployment not verified')
        for role,row in replay['exports'].items():
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Frozen role changed')
        sources.append((name,study,replay,sha256_file(study_path),sha256_file(replay_path)))
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    release=Release('data/kaggle_public');studies=[];file_hashes={}
    for name,study,replay,study_sha,replay_sha in sources:
        config=study['registered']['protocol'];panel=config['panel'];baseline=StateModel.load(Path(config['baseline_file']))
        net,topology=release.network(panel),release.topology(panel);template=release.template(1,panel,'train')
        files={partition_date(release,p):p for p in release.files(panel,'train')}
        if not set(dates)<=set(files):raise ValueError('Missing registered forward dates')
        models={role:StateModel.load(Path(row['path'])) for role,row in replay['exports'].items()};predictions={role:[] for role in models};days=[]
        out=folder/name;out.mkdir();factor=None
        for date in dates:
            frame=release.read(files[date]);features=StateFeatures(frame,net,topology,baseline.profile,version=baseline.feature_version)
            target=day_template(release,panel,'train',frame,template);raw=next(raw_for(release,panel,[files[date]]))
            truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float);lane=target.link_id.map(net.lanes).to_numpy(float);truth[:,1]/=lane
            indices=features.lookup(target);shape=(len(features.times),len(features.links));y=np.zeros((*shape,2),np.float32);mask=np.zeros(shape,bool)
            grouped=pd.DataFrame({'i':indices,'v':truth[:,0],'q':truth[:,1]}).groupby('i')[['v','q']].mean()
            y.reshape(-1,2)[grouped.index]=grouped.to_numpy()/np.array([25.,600.]);mask.ravel()[indices]=True
            day={'date':date,'x':reconstruction_input(features),'y':y,'mask':mask,'target_indices':indices,'station_truth':truth,
                'target_link_lanes':lane,'station_lanes':release.station_lanes(panel,target),'regime':target.mask_regime.iloc[0]}
            day['diagnostic_masks']=transition_masks(day,target,raw)
            if factor is None:factor=(net.lanes*net.length_km).reindex(features.links).to_numpy(np.float32)
            saved={}
            for role,model in models.items():
                result=model.predict(features,target)
                if not result[STATE_KEY].reset_index(drop=True).equals(target[STATE_KEY].reset_index(drop=True)):
                    raise ValueError('Frozen policy changed prediction keys')
                values=result[VALUES].to_numpy(float);values[:,1]/=lane
                predictions[role].append(values);saved[role]=values
            np.savez_compressed(out/(date+'.npz'),**saved,**{'mask_'+k:v for k,v in day['diagnostic_masks'].items()})
            days.append(day);file_hashes[files[date].as_posix()]=sha256_file(files[date])
        metrics={role:summarize(days,values,factor) for role,values in predictions.items()}
        comparisons={role:{'gain_vs_incumbent':values['S_state']-metrics['incumbent']['S_state'],
            'paired_days_vs_incumbent':paired_block_interval([d['daily_State'] for d in metrics['incumbent']['days']],
                [d['daily_State'] for d in values['days']])} for role,values in metrics.items() if role!='incumbent'}
        studies.append({'source_study':name,'source_study_sha256':study_sha,'source_replay_sha256':replay_sha,
            'panel':panel,'dates':dates,'original_primary':study['selection']['primary'],'metrics':metrics,'comparisons':comparisons,
            'all_original_choices_frozen':True,'calendar_disjoint_from_all_original_phases':True})
        print('forward',panel,'incumbent',metrics['incumbent']['S_state'],'original_primary',metrics['primary_export']['S_state'],flush=True)
        del models,days,predictions
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result={'registered':registered,'source_snapshot':snapshot,'studies':studies,'masked_input_files_sha256':file_hashes,
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'selection_changed':False,
        'limitations':'Forward public-train development,not untouched final/hidden truth. All predictors and alpha values frozen from original cal. '
        'Daily block intervals condition on the selected models;multiple policy comparisons not multiplicity corrected. '
        'Observed transitions and legacy target-pair increments are diagnostics only.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'extension.json').write_text(text,encoding='utf-8')
    print('forward_complete','studies',len(studies),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
