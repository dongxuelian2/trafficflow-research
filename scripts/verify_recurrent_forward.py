"""Rebuild every February teacher,score all saved policies,replay fixed dates."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import Release,VALUES
from trafficflow.state import StateModel,StateFeatures
from trafficflow.experiments import partition_date,day_template,raw_for,target_truth
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite forward verification')
    forward=json.loads(args.study.read_text());source_folder=Path('artifacts/research')/args.study.stem
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Recurrent_forward_all_teacher_metric_and_fixed_date_policy_replay','source_forward_sha256':sha256_file(args.study),
        'scope':'All56panel-days/560saved policy metrics;direct first/last date primary+incumbent only,chosen without looking at errors',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    release=Release('data/kaggle_public');rows=[];raw_hashes={};direct=[];max_score_error=0.;max_prediction_error=0.
    for run in forward['studies']:
        original_path=Path('reports/research')/(run['source_study']+'.json');original=json.loads(original_path.read_text())
        if sha256_file(original_path)!=run['source_study_sha256']:raise ValueError('Original model study changed')
        config=original['registered']['protocol'];panel=run['panel'];net,topology=release.network(panel),release.topology(panel)
        base=StateModel.load(Path(config['baseline_file']))
        replay_name=forward['registered']['protocol']['export_reports'][forward['registered']['protocol']['studies'].index(run['source_study'])]
        replay_path=Path('reports/research')/(replay_name+'.json');replay=json.loads(replay_path.read_text())
        if sha256_file(replay_path)!=run['source_replay_sha256']:raise ValueError('Forward export source changed')
        models={r:StateModel.load(Path(replay['exports'][r]['path'])) for r in ['primary_export','incumbent']}
        for r in models:
            if sha256_file(replay['exports'][r]['path'])!=replay['exports'][r]['sha256']:raise ValueError('Frozen role changed')
        files={partition_date(release,p):p for p in release.files(panel,'train')};raw_files={p.name:p for p in release.files(panel,'train',False)}
        template=release.template(1,panel,'train');statistics={role:[] for role in run['metrics']}
        for date in run['dates']:
            masked_path=files[date]
            if sha256_file(masked_path)!=forward['masked_input_files_sha256'][masked_path.as_posix()]:raise ValueError('Masked input changed')
            frame=release.read(masked_path);raw=next(raw_for(release,panel,[masked_path]));raw_hashes[raw_files[masked_path.name].as_posix()]=sha256_file(raw_files[masked_path.name])
            target=day_template(release,panel,'train',frame,template);truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float,copy=True)
            link_lane=target.link_id.map(net.lanes).to_numpy(float);station_lane=release.station_lanes(panel,target)
            replay_date=date in [run['dates'][0],run['dates'][-1]]
            features=StateFeatures(frame,net,topology,base.profile,version=base.feature_version) if replay_date else None
            with np.load(source_folder/run['source_study']/(date+'.npz')) as saved:
                for role in run['metrics']:
                    lane_prediction=saved[role]
                    v=lane_prediction[:,0]-truth[:,0];q=(lane_prediction[:,1]*link_lane-truth[:,1])/station_lane
                    stats={'date':date,'regime':target.mask_regime.iloc[0],'n':len(target),'speed_sse':float((v*v).sum()),
                        'lane_flow_sse':float((q*q).sum()),'daily_State':float(.54*max(0,1-np.sqrt(np.mean(v*v))/25)+.46*max(0,1-np.sqrt(np.mean(q*q))/600))}
                    expected=next(d for d in run['metrics'][role]['days'] if d['date']==date)
                    error=abs(stats['daily_State']-expected['daily_State']);max_score_error=max(max_score_error,error)
                    if error>1e-10 or stats['n']!=expected['n']:raise ValueError('Forward independent teacher/score differs')
                    statistics[role].append(stats);rows.append({'panel':panel,'date':date,'role':role,'daily_score_difference':error})
                    if replay_date and role in models:
                        fresh=models[role].predict(features,target)[VALUES].to_numpy(float,copy=True);fresh[:,1]/=link_lane
                        difference=float(np.max(np.abs(fresh-lane_prediction)));max_prediction_error=max(max_prediction_error,difference)
                        if difference>1e-8:raise ValueError('Forward saved-policy replay differs')
                        direct.append({'panel':panel,'date':date,'role':role,'maximum_prediction_difference_lane_units':difference})
            for role,stats in statistics.items():
                if len(stats)==len(run['dates']):
                    score=state_from_sufficient_stats(stats)['S_state']
                    if abs(score-run['metrics'][role]['S_state'])>1e-10:raise ValueError('Forward pooled State differs')
        del models
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result={'registered':registered,'source_snapshot':snapshot,'n_panel_days':len(raw_hashes),'n_saved_role_day_metrics_verified':len(rows),
        'metric_checks':rows,'maximum_daily_score_error':max_score_error,'direct_fixed_date_policy_replays':direct,
        'maximum_prediction_difference_lane_units':max_prediction_error,'raw_teacher_files_sha256':raw_hashes,
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'Complete saved-metric/teacher regeneration and8direct policy/day calls only;not560fresh predictor calls,not independent training or formal accuracy.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('forward_verified','panel_days',len(raw_hashes),'role_metrics',len(rows),'score_max_diff',max_score_error,
        'direct_calls',len(direct),'pred_max_diff',max_prediction_error,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
