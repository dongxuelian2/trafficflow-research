"""Regenerate future teachers/rescore all arrays and sample direct deployment."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import Release,VALUES,STATE_KEY,eligible
from trafficflow.state import StateModel,StateFeatures
from trafficflow.outage_stress import outage_geometry,remove_measurements
from trafficflow.experiments import partition_date,day_template,raw_for,target_truth
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from trafficflow.runtime import capture_source_snapshot,sha256_file


def independent_score(teacher,prediction,lanes):
    if not np.isfinite(prediction).all() or (prediction<0).any():raise ValueError('Invalid saved prediction')
    errors=prediction-teacher;n=len(teacher);sv=float((errors[:,0]**2).sum());sq=float(((errors[:,1]/lanes)**2).sum())
    value=.54*max(0,1-(sv/n)**.5/25)+.46*max(0,1-(sq/n)**.5/600)
    return {'n':n,'speed_sse':sv,'lane_flow_sse':sq,'score':value}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Use new future verification outputs')
    study=json.loads(args.study.read_text());config=study['registered']['protocol'];trained=json.loads(Path(config['source_study']).read_text())
    if sha256_file(config['source_study'])!=config['source_study_sha256']:raise ValueError('Trained source changed')
    trained_config=trained['registered']['protocol'];trained_folder=Path('artifacts/research')/Path(config['source_study']).stem
    if sha256_file(trained_folder/'selection_sealed.json')!=config['selection_seal_sha256']:raise ValueError('Model selection changed')
    if config['dates']!=trained_config['future_evaluation_dates'] or config['mask_seeds']!=trained_config['future_mask_seeds'] or config['cases']!=trained_config['future_cases']:
        raise ValueError('Future evaluation diverged from pre-fit registration')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Independent_future_augmentation_metrics_teacher_and_deployed_samples','source_study_sha256':sha256_file(args.study),
        'scope':'All ordinary/stress teachers,indices,saved9rolemetrics;direct2ordinarydaysx9roles+4stresscasesx9roles',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    models={}
    for role,row in config['exports'].items():
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Export changed')
        models[role]=StateModel.load(Path(row['path']))
    for arm in ['control','augmented']:models[arm+'_seed_mean']=StateModel.load(trained_folder/('State_'+arm+'_seed_mean.pkl'))
    release=Release('data/kaggle_public');panel=trained_config['panel'];net,topology=release.network(panel),release.topology(panel)
    profile=models['incumbent'].profile;version=models['incumbent'].feature_version
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train')
    ordinary_records={r['date']:r for r in study['ordinary_files']};ordinary_rows={(r['date'],r['role']):r for r in study['ordinary_daily_stats']}
    stress_rows={(r['key'],r['role']):r for r in study['stress_rows']};case_records={r['key']:r for r in study['case_records']}
    ordinary_stats=[];maximum_score=0.;maximum_prediction=0.;ordinary_count=stress_count=case_count=0;direct=[]
    for date in config['dates']:
        frame=release.read(files[date]);raw=next(raw_for(release,panel,[files[date]]))
        if sha256_file(files[date])!=study['masked_files_sha256'][files[date].as_posix()]:raise ValueError('Masked public source changed')
        features=StateFeatures(frame,net,topology,profile,version=version);target=day_template(release,panel,'train',frame,template)
        teacher=target_truth(release,panel,target,raw)[VALUES].to_numpy(float,copy=True);lanes=release.station_lanes(panel,target)
        record=ordinary_records[date]
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Ordinary predictions changed')
        with np.load(record['path']) as a:
            if not np.array_equal(teacher,a['truth_total_units']) or not np.array_equal(lanes,a['station_lanes']) or not np.array_equal(features.lookup(target),a['target_grid_indices']):
                raise ValueError('Ordinary evaluator teacher/indices differ')
            for role,model in models.items():
                value=independent_score(teacher,a[role],lanes);wanted=ordinary_rows[(date,role)]
                error=abs(value['score']-wanted['daily_State']);maximum_score=max(maximum_score,error)
                if error>1e-12 or value['n']!=wanted['n'] or any(not np.isclose(value[k],wanted[k],rtol=1e-12,atol=1e-6) for k in ['speed_sse','lane_flow_sse']):
                    raise ValueError('Ordinary sufficient statistics/score differ')
                ordinary_stats.append({**value,'date':date,'role':role,'regime':frame.mask_regime.iloc[0]});ordinary_count+=1
                if date in [config['dates'][0],config['dates'][-1]]:
                    prediction=model.predict(features,target)[VALUES].to_numpy(float);difference=float(np.abs(prediction-a[role]).max())
                    maximum_prediction=max(maximum_prediction,difference)
                    if difference>1e-8:raise ValueError('Ordinary deployed prediction differs')
                    direct.append({'population':'ordinary','date':date,'role':role,'maximum_total_unit_error':difference})
        for seed in config['mask_seeds']:
            for duration,fraction in config['cases']:
                key=f'{date}_{seed}_{duration}_{int(100*fraction)}';record=case_records[key]
                selected,geometry=outage_geometry(frame,panel,date,seed,duration,fraction)
                if geometry!=record['geometry']:raise ValueError('Metadata-only future geometry differs')
                region=raw.timestamp.between(geometry['start_timestamp'],geometry['last_removed_timestamp'])&raw.link_id.isin(geometry['selected_links'])
                wanted=raw.loc[region&eligible(raw),['timestamp','station_id','link_id']].copy()
                wanted['panel']=panel;wanted['mask_regime']=frame.mask_regime.iloc[0];wanted=wanted[STATE_KEY]
                if not len(wanted):raise ValueError('Zero-support future case requires unknown rather than complete verification')
                teacher=target_truth(release,panel,wanted,raw)[VALUES].to_numpy(float,copy=True);lane=release.station_lanes(panel,wanted)
                fresh=date==config['dates'][0] and seed==config['mask_seeds'][0]
                hidden=StateFeatures(remove_measurements(frame,selected),net,topology,profile,version=version) if fresh else None
                time_index={t:i for i,t in enumerate(features.times)};link_index={link:i for i,link in enumerate(features.links)}
                index=np.array([time_index[t]*len(features.links)+link_index[link] for t,link in zip(wanted.timestamp,wanted.link_id)])
                if sha256_file(record['path'])!=record['sha256']:raise ValueError('Stress prediction bytes changed')
                with np.load(record['path']) as a:
                    if not np.array_equal(teacher,a['truth_total_units']) or not np.array_equal(lane,a['station_lanes']) or not np.array_equal(index,a['target_grid_indices']):
                        raise ValueError('Stress teacher/normalization/indices differ')
                    for role,model in models.items():
                        value=independent_score(teacher,a[role],lane);error=abs(value['score']-stress_rows[(key,role)]['synthetic_reconstruction_score'])
                        maximum_score=max(maximum_score,error);stress_count+=1
                        if error>1e-12 or value['n']!=record['n']:raise ValueError('Synthetic score differs')
                        if fresh:
                            prediction=model.predict(hidden,wanted)[VALUES].to_numpy(float);difference=float(np.abs(prediction-a[role]).max())
                            maximum_prediction=max(maximum_prediction,difference)
                            if difference>1e-8:raise ValueError('Stress deployed prediction differs')
                            direct.append({'population':'synthetic_stress','key':key,'role':role,'maximum_total_unit_error':difference})
                case_count+=1
        print('augmentation_future_verify',date,'complete',flush=True)
    for role in models:
        measured=state_from_sufficient_stats([r for r in ordinary_stats if r['role']==role])['S_state']
        if abs(measured-study['ordinary'][role]['S_state'])>1e-12:raise ValueError('Regime pooled State differs')
    result={'registered':registered,'source_snapshot':snapshot,'n_ordinary_teacher_days_regenerated':len(config['dates']),
        'n_ordinary_saved_role_day_metrics_verified':ordinary_count,'n_stress_masks_and_teachers_regenerated':case_count,
        'n_stress_saved_scores_independently_recalculated':stress_count,'maximum_score_error':maximum_score,
        'n_direct_role_predictor_calls':len(direct),'direct_predictor_checks':direct,'maximum_prediction_error_total_units':maximum_prediction,
        'seconds':time.perf_counter()-start,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'All saved metrics/teacher arrays verified;54directrolecalls only,not every stress prediction freshly re-inferred. Not independent retraining.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('augmentation_future_verified',ordinary_count,stress_count,'direct',len(direct),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
