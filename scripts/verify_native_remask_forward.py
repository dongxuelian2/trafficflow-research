"""Independent raw-forward metrics,diagnostics and sampled frozen deployment."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,VALUES,eligible
from trafficflow.state import StateFeatures,StateModel
from trafficflow.experiments import partition_date,day_template,target_truth
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from trafficflow.runtime import capture_source_snapshot,sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite forward verification')
    source=json.loads(args.study.read_text());config=source['registered']['protocol'];training_path=Path(config['source_study']);training=json.loads(training_path.read_text())
    if sha256_file(training_path)!=config['source_study_sha256'] or config['dates']!=training['registered']['protocol']['future_dates']:
        raise ValueError('Forward source/calendar differs from pre-fit registration')
    model_source=Path('artifacts/research')/training_path.stem
    if sha256_file(model_source/'selection_sealed.json')!=training['selection_seal_sha256'] or source.get('selection_changed',source.get('choices_changed',True)):
        raise ValueError('Forward choices changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Native_remask_forward_all_metrics_and_sampled_predictor_verification','source_study_sha256':sha256_file(args.study),
        'scope':f"All{len(config['dates'])}raw teacher/target/diagnostic arrays and{len(config['dates'])*len(source['metrics'])}saved role/day metrics;first/last dates x{len(source['exports'])}direct predictors",
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    models={}
    for role,row in source['exports'].items():
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Frozen export changed')
        models[role]=StateModel.load(Path(row['path']))
    release=Release('data/kaggle_public');panel=training['registered']['protocol']['panel'];net,topology=release.network(panel),release.topology(panel)
    anchor=models.get('incumbent',models.get('nn_primary'))
    if anchor is None:raise ValueError('Recognized frozen profile anchor required')
    profile=anchor.profile;version=anchor.feature_version
    files={partition_date(release,p):p for p in release.files(panel,'train')};raw_files={p.name:p for p in release.files(panel,'train',False)};template=release.template(1,panel,'train')
    for path,digest in source['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Public masked/raw source changed')
    rows=[];checks=[];direct=[];maximum_score=maximum_prediction=0.
    for record in source['prediction_files']:
        date=record['date'];frame=release.read(files[date]);raw=release.read(raw_files[files[date].name])
        features=StateFeatures(frame,net,topology,profile,version);target=day_template(release,panel,'train',frame,template)
        teacher=target_truth(release,panel,target,raw)[VALUES].to_numpy(float);lane=target.link_id.map(net.lanes).to_numpy(float);station=release.station_lanes(panel,target)
        teacher[:,1]/=lane;indices=features.lookup(target)
        previous=target[['timestamp','station_id','link_id']].copy();previous['timestamp']=(pd.to_datetime(previous.timestamp,utc=True)-pd.Timedelta(minutes=5)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        raw_good=raw.loc[eligible(raw)].set_index(['timestamp','station_id','link_id'])
        last=raw_good.speed_kmh.reindex(pd.MultiIndex.from_frame(previous)).to_numpy(float);delta=teacher[:,0]-last
        missing=~np.isfinite(features.raw[...,0]);runs=np.zeros(missing.shape,np.int32)
        for link in range(missing.shape[1]):
            edges=np.diff(np.r_[0,missing[:,link].astype(int),0])
            for first,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):runs[first:end,link]=end-first
        gap=runs.ravel()[indices];masks={'drop20':delta<=-20,'recover20':delta>=20,'missing60min':gap>=12,
            'missing180min':gap>=36,'congested50':teacher[:,0]<50}
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Saved forward predictions changed')
        with np.load(record['path']) as a:
            if not np.array_equal(teacher,a['truth_link_lane_units']) or not np.array_equal(station,a['station_lanes']) or not np.array_equal(lane,a['target_link_lanes']) or not np.array_equal(indices,a['target_grid_indices']):
                raise ValueError('Raw teacher/lanes/target order differs')
            if any(not np.array_equal(a['mask_'+key],value) for key,value in masks.items()):raise ValueError('Observed transition/input gap diagnosis differs')
            for role in source['metrics']:
                model=models.get(role)
                prediction=a[role];ev=prediction[:,0]-teacher[:,0];eq=(prediction[:,1]-teacher[:,1])*lane/station
                n=len(teacher);sv=float(ev@ev);sq=float(eq@eq);score=.54*max(0,1-(sv/n)**.5/25)+.46*max(0,1-(sq/n)**.5/600)
                wanted=next(r for r in source['metrics'][role]['days'] if r['date']==date);error=abs(score-wanted['daily_State']);maximum_score=max(maximum_score,error)
                if error>1e-12 or n!=wanted['n'] or not np.isclose(sv,wanted['speed_sse'],rtol=1e-12,atol=1e-6) or not np.isclose(sq,wanted['lane_flow_sse'],rtol=1e-12,atol=1e-6):
                    raise ValueError('Independent forward metric differs')
                rows.append({'date':date,'role':role,'regime':frame.mask_regime.iloc[0],'n':n,'speed_sse':sv,'lane_flow_sse':sq})
                if '_hybrid_' in role:
                    variant,alpha_text=role.split('_hybrid_');alpha=float(alpha_text)
                    if not np.array_equal(prediction,alpha*a[variant+'_seed_mean']+(1-alpha)*a['nn_primary']):raise ValueError('Saved hybrid mixture differs')
                if model is not None and date in [config['dates'][0],config['dates'][-1]]:
                    p=model.predict(features,target)[VALUES].to_numpy(float);p[:,1]/=lane;difference=float(np.abs(p-prediction).max());maximum_prediction=max(maximum_prediction,difference)
                    if difference>1e-8:raise ValueError('Fresh deployed forward prediction differs')
                    direct.append({'date':date,'role':role,'maximum_lane_unit_error':difference})
        checks.append({'date':date,'teacher_indices_lanes_and_all_diagnostic_masks_exact':True})
    for role in source['metrics']:
        if abs(state_from_sufficient_stats([r for r in rows if r['role']==role])['S_state']-source['metrics'][role]['S_state'])>1e-12:
            raise ValueError('Independent regime pooling differs')
    result={'registered':registered,'source_snapshot':snapshot,'n_raw_teacher_days_rebuilt':len(checks),'n_saved_role_day_metrics_recalculated':len(rows),
        'n_direct_predictor_calls':len(direct),'maximum_score_error':maximum_score,'maximum_prediction_difference_lane_units':maximum_prediction,
        'day_checks':checks,'direct_predictor_checks':direct,'seconds':time.perf_counter()-start,
        'direct_neural_device_policy':'cuda_if_available' if torch.cuda.is_available() else 'cpu',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':f"All saved metrics and diagnostic masks verified;{len(direct)}direct policy calls only,not{len(rows)}new predictions. Not independent retraining."}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('native_forward_verified',len(rows),'direct',len(direct),'maxpred',maximum_prediction,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
