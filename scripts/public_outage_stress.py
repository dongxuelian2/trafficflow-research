"""Frozen first-fold models under new,metadata-selected public-train outages."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,VALUES,STATE_KEY,eligible
from trafficflow.state import StateModel,StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.outage_stress import outage_geometry,remove_measurements,MEASUREMENTS
from trafficflow.experiments import partition_date,day_template,raw_for,target_truth
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats,paired_block_interval
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence


def metric(truth,prediction,lanes):
    n=len(truth)
    if not n:return {'n':0,'synthetic_reconstruction_score':None,'speed_sse':0.,'lane_flow_sse':0.}
    v=prediction[:,0]-truth[:,0];q=(prediction[:,1]-truth[:,1])/lanes
    rv,rq=float(np.sqrt(np.mean(v*v))),float(np.sqrt(np.mean(q*q)))
    return {'n':n,'speed_sse':float(v@v),'lane_flow_sse':float(q@q),'rmse_speed':rv,'rmse_lane_flow':rq,
        'synthetic_reconstruction_score':float(.54*max(0,1-rv/25)+.46*max(0,1-rq/600))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite outage stress evidence')
    source=Path('reports/research/A72b_recurrent_I210_W_autumn_full_day.json')
    replay_path=Path('reports/research/A74b_recurrent_I210_W_independent_replay.json')
    study=json.loads(source.read_text());replay=json.loads(replay_path.read_text())
    dates=pd.date_range('2030-11-13','2030-11-26').strftime('%Y-%m-%d').tolist();seeds=[20261031,20261103,20261107]
    roles=['lgb','incumbent','tcn_seed_mean','tcn_gru_seed_mean'];cases=[(d,f) for d in [36,72] for f in [.3,.7]]
    if set(dates)&{d for group in study['prepared']['dates'].values() for d in group}:raise ValueError('Stress dates overlap model calendar')
    if replay['registered']['protocol']['source_study_sha256']!=sha256_file(source):raise ValueError('Wrong frozen export')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Metadata_selected_long_public_train_sensor_outages','source_study_sha256':sha256_file(source),
        'source_replay_sha256':sha256_file(replay_path),'dates':dates,'mask_seeds':seeds,'roles':roles,
        'cases':[{'duration_steps':d,'spatial_fraction':f} for d,f in cases],
        'geometry':'Timestamp center09-15UTC,random center in public milepost order;clip contiguous30/70pct sensor segment;3/6h nested cases',
        'masked_fields':MEASUREMENTS,'mask_flags':{'is_observed':False,'is_imputed':False,'is_score_eligible':False,'is_missing':True},
        'model_policy':'All original weights/alphas fixed;no choice from this stress test',
        'target_population':'All raw-public-train eligible station rows inside outage;separate from released Task1 control targets',
        'zero_target_policy':'Retain zero-support cases as unknown;never resample by observed labels',
        'scores':'Conditional channel-scaled reconstruction diagnostic,not official R1/R2/R3 State or Kaggle Total',
        'data_scope':'Dates not used by these weights,previously studied public calendars possible;no hidden split/reference access',
        'geometry_source_sha256':sha256_file('src/trafficflow/outage_stress.py'),
        'independent_scenarios':'42nested center-date-seed draws,168correlated outage cases;not168independent events',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    folder.mkdir(parents=True);(folder/'cases').mkdir();snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    release=Release('data/kaggle_public');panel=study['registered']['protocol']['panel'];net,topology=release.network(panel),release.topology(panel)
    models={r:StateModel.load(Path(replay['exports'][r]['path'])) for r in roles}
    for r in roles:
        if sha256_file(replay['exports'][r]['path'])!=replay['exports'][r]['sha256']:raise ValueError('Frozen model changed')
    profile=models['incumbent'].profile;version=models['incumbent'].feature_version
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train')
    rows=[];controls=[];source_files={};case_files=[]
    for date in dates:
        frame=release.read(files[date]);raw=next(raw_for(release,panel,[files[date]]));source_files[files[date].as_posix()]=sha256_file(files[date])
        original_features=StateFeatures(frame,net,topology,profile,version=version)
        target=day_template(release,panel,'train',frame,template);truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float,copy=True)
        lanes=release.station_lanes(panel,target)
        for role,model in models.items():
            p=model.predict(original_features,target)[VALUES].to_numpy(float,copy=True);m=metric(truth,p,lanes)
            controls.append({'date':date,'role':role,'regime':target.mask_regime.iloc[0],
                'n':m['n'],'speed_sse':m['speed_sse'],'lane_flow_sse':m['lane_flow_sse'],'daily_State':m['synthetic_reconstruction_score']})
        for seed in seeds:
            for duration,fraction in cases:
                selected,geometry=outage_geometry(frame,panel,date,seed,duration,fraction)
                hidden=remove_measurements(frame,selected)
                if not hidden.loc[selected,[c for c in MEASUREMENTS if c in hidden]].isna().all().all():raise ValueError('Incomplete outage removal')
                region=raw.timestamp.between(geometry['start_timestamp'],geometry['last_removed_timestamp'])&raw.link_id.isin(geometry['selected_links'])
                wanted=raw.loc[region&eligible(raw),['timestamp','station_id','link_id']].copy()
                wanted['panel']=panel;wanted['mask_regime']=frame.mask_regime.iloc[0];wanted=wanted[STATE_KEY]
                source_observed_fraction=float(eligible(frame)[selected].mean())
                key=f'{date}_{seed}_{duration}_{int(100*fraction)}';case=folder/'cases'/(key+'.npz')
                case_rows=[]
                if len(wanted):
                    teacher=target_truth(release,panel,wanted,raw)[VALUES].to_numpy(float,copy=True);lane=release.station_lanes(panel,wanted)
                    features=StateFeatures(hidden,net,topology,profile,version=version);x=reconstruction_input(features);indices=features.lookup(wanted)
                    if np.any(x.reshape(-1,12)[indices,2:4]) or np.any(x.reshape(-1,12)[indices,9]):
                        raise ValueError('Removed speed/flow/occupancy still visible to neural predictor')
                    arrays={'truth_total_units':teacher,'station_lanes':lane,'target_grid_indices':indices}
                    for role,model in models.items():
                        result=model.predict(features,wanted)
                        if not result[STATE_KEY].reset_index(drop=True).equals(wanted.reset_index(drop=True)):raise ValueError('Stress keys differ')
                        p=result[VALUES].to_numpy(float,copy=True);arrays[role]=p
                        case_rows.append({'date':date,'mask_seed':seed,'duration_steps':duration,'spatial_fraction':fraction,'role':role,
                            'source_mask_regime':frame.mask_regime.iloc[0],'geometry':geometry,
                            'original_observed_fraction_in_region':source_observed_fraction,**metric(teacher,p,lane)})
                    np.savez_compressed(case,**arrays)
                    case_files.append({'path':case.as_posix(),'sha256':sha256_file(case)})
                else:
                    case_rows=[{'date':date,'mask_seed':seed,'duration_steps':duration,'spatial_fraction':fraction,'role':r,
                        'geometry':geometry,'n':0,'synthetic_reconstruction_score':None,'reason':'No eligible raw target;not resampled'} for r in roles]
                rows.extend(case_rows)
        (folder/'progress.json').write_text(json.dumps({'completed_dates':date,'rows':rows},indent=2,allow_nan=False),encoding='utf-8')
        print('outage_stress',date,'complete',flush=True)
    summary={};comparisons={}
    for duration,fraction in cases:
        name=f'{duration*5}min_{int(100*fraction)}pct';subset=[r for r in rows if r['duration_steps']==duration and r['spatial_fraction']==fraction]
        summary[name]={};by_role={}
        for role in roles:
            values=[r for r in subset if r['role']==role]
            if any(r['n']==0 for r in values):
                summary[name][role]={'mean_synthetic_score':None,'reason':'Zero-support case invalidates whole comparison'};continue
            summary[name][role]={'mean_synthetic_score':float(np.mean([r['synthetic_reconstruction_score'] for r in values])),
                'n_case_rows':len(values),'pooled_RMSE_speed':float(np.sqrt(sum(r['speed_sse'] for r in values)/sum(r['n'] for r in values))),
                'pooled_RMSE_lane_flow':float(np.sqrt(sum(r['lane_flow_sse'] for r in values)/sum(r['n'] for r in values)))}
            by_role[role]=[float(np.mean([r['synthetic_reconstruction_score'] for r in values if r['date']==date])) for date in dates]
        comparisons[name]={r:paired_block_interval(by_role['incumbent'],by_role[r]) for r in roles if r!='incumbent'} if len(by_role)==len(roles) else None
    control={role:state_from_sufficient_stats([r for r in controls if r['role']==role]) for role in roles}
    result={'registered':registered,'source_snapshot':snapshot,'rows':rows,'summary':summary,'paired_day_diagnostics':comparisons,
        'official_template_control':control,'official_control_daily_stats':controls,'case_files':case_files,
        'masked_input_files_sha256':source_files,'n_nested_stress_cases':len(rows)//len(roles),'n_center_date_seed_draws':len(dates)*len(seeds),
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'selection_changed':False,
        'limits':'Own additional masks on released train only. Stress/control target populations differ and scores are not directly compared. '
        'Four outage geometries share nested centers;three mask seeds share each date. Intervals resample chronological date means,not cells/events. '
        'Known complete teacher labels stay evaluator-only;no stress outcome selects a model. Runs concurrently with winter training,not isolated timing.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('outage_complete',summary,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
