"""Registered future ordinary and synthetic-stress evaluation,frozen policies."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import Release,VALUES,STATE_KEY,eligible
from trafficflow.state import StateModel,StateFeatures
from trafficflow.outage_stress import outage_geometry
from trafficflow.state_augmentation import rebuild_after_outage
from trafficflow.experiments import partition_date,day_template,raw_for,target_truth
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats,paired_block_interval
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from public_outage_stress import metric


def policy_paths(study_path):
    study=json.loads(study_path.read_text());config=study['registered']['protocol'];folder=Path('artifacts/research')/study_path.stem
    origin=json.loads(Path(config['source_study']).read_text())['registered']['protocol']
    replay=json.loads(Path('reports/research/A74b_recurrent_I210_W_independent_replay.json').read_text());exports=replay['exports']
    if config['panel']!='D7_I210_W':raise ValueError('This registered reference export is I210_W only')
    if replay['registered']['protocol']['source_study_sha256']!=sha256_file(config['source_study']):raise ValueError('Wrong prior export')
    for name in ['lgb','tcn_gru_seed_mean']:
        if sha256_file(exports[name]['path'])!=exports[name]['sha256']:raise ValueError('Prior exported policy changed')
    if sha256_file(origin['incumbent_file'])!=origin['incumbent_sha256']:raise ValueError('Original incumbent changed')
    paths={'lgb':Path(exports['lgb']['path']),'incumbent':Path(origin['incumbent_file']),
        'prior_GRU':Path(exports['tcn_gru_seed_mean']['path'])}
    for seed in config['seeds']:
        for arm in ['control','augmented']:
            name=f'seed{seed}_{arm}';path=folder/('State_'+name+'.pkl')
            if sha256_file(path)!=study['training'][name]['model_sha256']:raise ValueError('Trained predictor changed')
            paths[name]=path
    return paths


def predict_roles(models,features,target,seeds):
    values={name:model.predict(features,target)[VALUES].to_numpy(float,copy=True) for name,model in models.items()}
    for arm in ['control','augmented']:
        values[arm+'_seed_mean']=np.mean([values[f'seed{s}_{arm}'] for s in seeds],axis=0)
    return values


def compare_daily(rows,dates,left,right):
    a=[np.mean([r['synthetic_reconstruction_score'] for r in rows if r['date']==date and r['role']==left]) for date in dates]
    b=[np.mean([r['synthetic_reconstruction_score'] for r in rows if r['date']==date and r['role']==right]) for date in dates]
    return paired_block_interval(a,b)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Use a new future evaluation')
    study=json.loads(args.study.read_text());config=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256'] or json.loads((source/'selection_sealed.json').read_text())!=study['selection']:
        raise ValueError('Selection changed or incomplete')
    paths=policy_paths(args.study);seeds=config['seeds'];dates=config['future_evaluation_dates'];cases=config['future_cases'];mask_seeds=config['future_mask_seeds']
    roles=list(paths)+['control_seed_mean','augmented_seed_mean']
    primary=study['selection']['primary']
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Frozen_augmentation_future_calendar_and_public_outages','source_study':args.study.as_posix(),
        'source_study_sha256':sha256_file(args.study),'selection_seal_sha256':study['selection_seal_sha256'],
        'dates':dates,'mask_seeds':mask_seeds,'cases':cases,'roles':roles,'primary':primary,
        'exports':{n:{'path':p.as_posix(),'sha256':sha256_file(p)} for n,p in paths.items()},
        'calendar_origin':'Fixed in training preregistration before any new model fit',
        'ordinary':'Official masked train/template target population,raw train evaluator only',
        'stress':'Own removed sensor regions,all raw-public-train eligible teachers;different target population',
        'comparisons':'Fixed augmented-v-control,augmented-v-prior,perseed pairs;all outcomes retained',
        'selection':'No weight,alpha,mask,date or primary selection from these outcomes',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    folder.mkdir(parents=True);(folder/'cases').mkdir();(folder/'ordinary').mkdir()
    snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    models={n:StateModel.load(p) for n,p in paths.items()}
    release=Release('data/kaggle_public');panel=config['panel'];net,topology=release.network(panel),release.topology(panel)
    profile=models['incumbent'].profile;version=models['incumbent'].feature_version
    if any(m.feature_version!=version or not m.profile.slots.equals(profile.slots) or not m.profile.fallback.equals(profile.fallback) for m in models.values()):
        raise ValueError('Policies use different profiles/features')
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train')
    rows=[];ordinary=[];case_records=[];ordinary_files=[];file_hashes={}
    for date in dates:
        frame=release.read(files[date])
        raw=next(raw_for(release,panel,[files[date]]));file_hashes[files[date].as_posix()]=sha256_file(files[date])
        features=StateFeatures(frame,net,topology,profile,version=version);target=day_template(release,panel,'train',frame,template)
        truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float,copy=True);lanes=release.station_lanes(panel,target)
        values=predict_roles(models,features,target,seeds)
        path=folder/'ordinary'/(date+'.npz');np.savez_compressed(path,truth_total_units=truth,station_lanes=lanes,
            target_grid_indices=features.lookup(target),**values)
        ordinary_files.append({'date':date,'path':path.as_posix(),'sha256':sha256_file(path)})
        for role,p in values.items():
            m=metric(truth,p,lanes);ordinary.append({'date':date,'role':role,'regime':frame.mask_regime.iloc[0],
                'n':m['n'],'speed_sse':m['speed_sse'],'lane_flow_sse':m['lane_flow_sse'],'daily_State':m['synthetic_reconstruction_score']})
        for mask_seed in mask_seeds:
            for duration,fraction in cases:
                selected,geometry=outage_geometry(frame,panel,date,mask_seed,duration,fraction)
                inside=raw.timestamp.between(geometry['start_timestamp'],geometry['last_removed_timestamp'])&raw.link_id.isin(geometry['selected_links'])
                wanted=raw.loc[inside&eligible(raw),['timestamp','station_id','link_id']].copy()
                wanted['panel']=panel;wanted['mask_regime']=frame.mask_regime.iloc[0];wanted=wanted[STATE_KEY]
                key=f'{date}_{mask_seed}_{duration}_{int(100*fraction)}';record={'key':key,'date':date,'mask_seed':mask_seed,
                    'duration_steps':duration,'spatial_fraction':fraction,'geometry':geometry,'n':len(wanted)}
                if len(wanted):
                    hidden=rebuild_after_outage(frame,selected,net,topology,profile,version)
                    teacher=target_truth(release,panel,wanted,raw)[VALUES].to_numpy(float,copy=True);lane=release.station_lanes(panel,wanted)
                    values=predict_roles(models,hidden,wanted,seeds);path=folder/'cases'/(key+'.npz')
                    np.savez_compressed(path,truth_total_units=teacher,station_lanes=lane,target_grid_indices=hidden.lookup(wanted),**values)
                    record.update(path=path.as_posix(),sha256=sha256_file(path))
                    for role,p in values.items():rows.append({'key':key,'date':date,'mask_seed':mask_seed,'duration_steps':duration,
                        'spatial_fraction':fraction,'role':role,**metric(teacher,p,lane)})
                else:
                    for role in roles:rows.append({'key':key,'date':date,'mask_seed':mask_seed,'duration_steps':duration,
                        'spatial_fraction':fraction,'role':role,'n':0,'synthetic_reconstruction_score':None})
                case_records.append(record)
        (folder/'progress.json').write_text(json.dumps({'completed_date':date,'ordinary':ordinary,'rows':rows},indent=2,allow_nan=False),encoding='utf-8')
        print('augmentation_future',date,'complete',flush=True)
    ordinary_metrics={n:state_from_sufficient_stats([r for r in ordinary if r['role']==n]) for n in roles}
    ordinary_comparisons={}
    pairs=[('control_seed_mean','augmented_seed_mean'),('prior_GRU','augmented_seed_mean'),('incumbent',primary)]+[
        (f'seed{s}_control',f'seed{s}_augmented') for s in seeds]
    for left,right in pairs:
        ordinary_comparisons[right+'_vs_'+left]={'pooled_State_gain':ordinary_metrics[right]['S_state']-ordinary_metrics[left]['S_state'],
            'paired_day_diagnostic':paired_block_interval([r['daily_State'] for r in ordinary if r['role']==left],
                [r['daily_State'] for r in ordinary if r['role']==right])}
    stress_summary={};stress_comparisons={}
    for duration,fraction in cases:
        key=f'{duration*5}min_{int(100*fraction)}pct';subset=[r for r in rows if r['duration_steps']==duration and r['spatial_fraction']==fraction]
        if any(r['n']==0 for r in subset):stress_summary[key]={'all_comparisons_unknown':'Zero-support case retained'};continue
        stress_summary[key]={n:float(np.mean([r['synthetic_reconstruction_score'] for r in subset if r['role']==n])) for n in roles}
        stress_comparisons[key]={right+'_vs_'+left:compare_daily(subset,dates,left,right) for left,right in pairs}
    result={'registered':registered,'source_snapshot':snapshot,'ordinary':ordinary_metrics,'ordinary_daily_stats':ordinary,
        'ordinary_comparisons':ordinary_comparisons,'ordinary_files':ordinary_files,'stress_summary':stress_summary,
        'stress_comparisons':stress_comparisons,'stress_rows':rows,'case_records':case_records,'masked_files_sha256':file_hashes,
        'n_nested_stress_cases':len(case_records),'n_center_date_seed_draws':len(dates)*len(mask_seeds),
        'seconds':time.perf_counter()-start,'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
        'source':evidence(gpu_used=torch.cuda.is_available()),'selection_changed':False,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Fixed public research dates were not fitted by these weights,but other historical studies may have viewed them. '
        'Nested sizes share42centers and mask seeds share14days;unadjusted intervals resample chronological date means. '
        'Ordinary and synthetic stress populations differ;neither is a formal Kaggle component.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('augmentation_future_COMPLETE',{n:ordinary_metrics[n]['S_state'] for n in ['prior_GRU','control_seed_mean','augmented_seed_mean']},
        'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
