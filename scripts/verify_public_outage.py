"""Rebuild all stress labels/masks,rescore arrays,and replay selected cases."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,VALUES,STATE_KEY,eligible
from trafficflow.state import StateModel,StateFeatures
from trafficflow.outage_stress import outage_geometry,remove_measurements,MEASUREMENTS
from trafficflow.experiments import partition_date,day_template,raw_for,target_truth
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats


def independent_statistics(truth,prediction,lanes):
    error=prediction-truth;speed=float(np.square(error[:,0]).sum());flow=float(np.square(error[:,1]/lanes).sum());n=len(truth)
    score=.54*max(0,1-np.sqrt(speed/n)/25)+.46*max(0,1-np.sqrt(flow/n)/600)
    return {'n':n,'speed_sse':speed,'lane_flow_sse':flow,'synthetic_reconstruction_score':float(score)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite stress verification')
    study=json.loads(args.study.read_text());config=study['registered']['protocol']
    original=json.loads(Path('reports/research/A72b_recurrent_I210_W_autumn_full_day.json').read_text())
    replay=json.loads(Path('reports/research/A74b_recurrent_I210_W_independent_replay.json').read_text())
    if config['source_study_sha256']!=sha256_file('reports/research/A72b_recurrent_I210_W_autumn_full_day.json'):
        raise ValueError('Stress source model changed')
    if config['geometry_source_sha256']!=sha256_file('src/trafficflow/outage_stress.py'):raise ValueError('Outage generator changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Independent_public_outage_label_metric_and_predictor_replay','source_study_sha256':sha256_file(args.study),
        'scope':'Regenerate all168masks/teacher arrays;independent score formulas;four stress cases x four policies and all14control dates',
        'formal_State':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    release=Release('data/kaggle_public');panel=original['registered']['protocol']['panel'];net,topology=release.network(panel),release.topology(panel)
    roles=config['roles'];models={r:StateModel.load(Path(replay['exports'][r]['path'])) for r in roles}
    for r in roles:
        if sha256_file(replay['exports'][r]['path'])!=replay['exports'][r]['sha256']:raise ValueError('Frozen role changed')
    profile=models['incumbent'].profile;version=models['incumbent'].feature_version
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train')
    case_hashes={r['path']:r['sha256'] for r in study['case_files']};source_folder=Path('artifacts/research')/args.study.stem
    grouped={}
    for row in study['rows']:grouped.setdefault((row['date'],row['mask_seed'],row['duration_steps'],row['spatial_fraction']),[]).append(row)
    checks=[];predictor_checks=[];control_rows=[];maximum_score_error=0.;maximum_prediction_error=0.;scored=0
    for date in config['dates']:
        frame=release.read(files[date]);raw=next(raw_for(release,panel,[files[date]]))
        if sha256_file(files[date])!=study['masked_input_files_sha256'][files[date].as_posix()]:raise ValueError('Public masked input changed')
        control_target=day_template(release,panel,'train',frame,template)
        teacher=target_truth(release,panel,control_target,raw)[VALUES].to_numpy(float,copy=True);lanes=release.station_lanes(panel,control_target)
        original_features=StateFeatures(frame,net,topology,profile,version=version)
        for role,model in models.items():
            values=model.predict(original_features,control_target)[VALUES].to_numpy(float,copy=True)
            stats=independent_statistics(teacher,values,lanes)
            control_rows.append({**stats,'daily_State':stats['synthetic_reconstruction_score'],'role':role,'regime':control_target.mask_regime.iloc[0]})
        for seed in config['mask_seeds']:
            for case in config['cases']:
                duration,fraction=case['duration_steps'],case['spatial_fraction'];key=(date,seed,duration,fraction);rows=grouped[key]
                selected,geometry=outage_geometry(frame,panel,date,seed,duration,fraction)
                if any(r['geometry']!=geometry for r in rows):raise ValueError('Metadata-only geometry differs')
                masked=remove_measurements(frame,selected)
                if not masked.loc[selected,[c for c in MEASUREMENTS if c in masked]].isna().all().all():raise ValueError('Observed field remains inside outage')
                region=raw.timestamp.between(geometry['start_timestamp'],geometry['last_removed_timestamp'])&raw.link_id.isin(geometry['selected_links'])
                wanted=raw.loc[region&eligible(raw),['timestamp','station_id','link_id']].copy();wanted['panel']=panel;wanted['mask_regime']=frame.mask_regime.iloc[0];wanted=wanted[STATE_KEY]
                if not len(wanted):raise ValueError('Zero-support case requires unknown comparison;cannot issue complete verification')
                truth=target_truth(release,panel,wanted,raw)[VALUES].to_numpy(float,copy=True);lane=release.station_lanes(panel,wanted)
                path=source_folder/'cases'/(f'{date}_{seed}_{duration}_{int(100*fraction)}.npz')
                if sha256_file(path)!=case_hashes[path.as_posix()]:raise ValueError('Saved prediction bytes changed')
                fresh=date==config['dates'][0] and seed==config['mask_seeds'][0]
                features=StateFeatures(masked,net,topology,profile,version=version) if fresh else None
                with np.load(path) as arrays:
                    if not np.array_equal(arrays['truth_total_units'],truth) or not np.array_equal(arrays['station_lanes'],lane):
                        raise ValueError('Public teacher/normalization differs')
                    for row in rows:
                        role=row['role'];prediction=arrays[role]
                        if not np.isfinite(prediction).all() or (prediction<0).any():raise ValueError('Invalid prediction')
                        measured=independent_statistics(truth,prediction,lane)
                        error=abs(measured['synthetic_reconstruction_score']-row['synthetic_reconstruction_score'])
                        maximum_score_error=max(maximum_score_error,error);scored+=1
                        if error>1e-12 or measured['n']!=row['n']:raise ValueError('Independent score differs')
                        if fresh:
                            result=models[role].predict(features,wanted)
                            if not result[STATE_KEY].reset_index(drop=True).equals(wanted.reset_index(drop=True)):raise ValueError('Prediction keys differ')
                            difference=float(np.abs(result[VALUES].to_numpy(float)-prediction).max())
                            maximum_prediction_error=max(maximum_prediction_error,difference)
                            if difference>1e-8:raise ValueError('Independent stress predictor differs')
                            predictor_checks.append({'date':date,'mask_seed':seed,'duration_steps':duration,'spatial_fraction':fraction,
                                'role':role,'maximum_absolute_total_unit_difference':difference})
                checks.append({'date':date,'mask_seed':seed,'duration_steps':duration,'spatial_fraction':fraction,'mask_and_teacher_exact':True})
        print('stress_verify',date,'complete',flush=True)
    controls={role:state_from_sufficient_stats([r for r in control_rows if r['role']==role]) for role in roles}
    for role,metrics in controls.items():
        if abs(metrics['S_state']-study['official_template_control'][role]['S_state'])>1e-10:raise ValueError('Control metric differs')
    result={'registered':registered,'source_snapshot':snapshot,'n_masks_and_teachers_regenerated':len(checks),
        'n_saved_scores_independently_recalculated':scored,'maximum_score_error':maximum_score_error,
        'n_direct_stress_policy_calls':len(predictor_checks),'direct_stress_policy_checks':predictor_checks,
        'maximum_direct_prediction_error_total_units':maximum_prediction_error,'controls_recomputed_all_dates':True,
        'control_metrics':controls,'case_checks':checks,'seconds':time.perf_counter()-start,
        'source':evidence(gpu_used=torch.cuda.is_available()),'formal_State':None,'formal_Total':None,
        'scope':'Implementation verification only;no new model selection or independent traffic events. '
        'Masks/teacher regenerated from allowed public train,not hidden split truth.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('stress_verified',len(checks),'scores',scored,'max_error',maximum_score_error,'fresh_calls',len(predictor_checks),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
