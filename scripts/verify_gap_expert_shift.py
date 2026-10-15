"""Independent availability-run gate,teacher and metric verification."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,STATE_KEY,VALUES,eligible
from trafficflow.state import StateModel,StateFeatures
from trafficflow.outage_shift import shifted_outage_masks
from trafficflow.outage_stress import remove_measurements
from trafficflow.experiments import partition_date,day_template,target_truth
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from trafficflow.runtime import sha256_file,capture_source_snapshot


def independent_gate(frame,times,links,target_indices,threshold):
    grid=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id'])
    available=frame[['timestamp','link_id']].copy();available['visible']=eligible(frame)
    seen=available.groupby(['timestamp','link_id']).visible.any().reindex(grid,fill_value=False).to_numpy().reshape(len(times),len(links))
    runs=np.zeros(seen.shape,np.int32)
    for column in range(len(links)):
        missing=~seen[:,column];edges=np.diff(np.r_[0,missing.astype(int),0]);starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        for start,end in zip(starts,ends):runs[start:end,column]=end-start
    return runs.ravel()[target_indices]>=threshold


def score(teacher,prediction,lanes):
    if not np.isfinite(prediction).all() or (prediction<0).any():raise ValueError('Nonfinite/negative predictor')
    n=len(teacher);v=float(np.square(prediction[:,0]-teacher[:,0]).sum());q=float(np.square((prediction[:,1]-teacher[:,1])/lanes).sum())
    value=.54*max(0,1-(v/n)**.5/25)+.46*max(0,1-(q/n)**.5/600) if n else None
    return {'n':n,'speed_sse':v,'lane_flow_sse':q,'synthetic_reconstruction_score':value}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite shift verification')
    study=json.loads(args.study.read_text());config=study['registered']['protocol'];policy_path=Path(config['source_policy']);policy=json.loads(policy_path.read_text())
    if sha256_file(policy_path)!=config['source_policy_sha256'] or policy['registered']['protocol_sha256']!=config['policy_protocol_sha256']:
        raise ValueError('Frozen policy/protocol changed')
    pconfig=policy['registered']['protocol']
    for key in ['ordinary_dates','anchor_dates','cases','mask_seeds']:
        if config[key]!=pconfig[key]:raise ValueError('New evaluation diverged from preregistration')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_shifted_gap_expert_replay',
        'source_study_sha256':sha256_file(args.study),'scope':'All ordinary/segment/case teachers and scores;independent per-column RLE gate;44fixed direct predictor calls',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    models={}
    for role,row in pconfig['inference_exports'].items():
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Reference changed')
        models[role]=StateModel.load(Path(row['path']))
    if sha256_file(policy['export']['path'])!=policy['export']['sha256']:raise ValueError('Gate changed')
    models['gated']=StateModel.load(Path(policy['export']['path']))
    release=Release('data/kaggle_public');panel=pconfig['panel'];net,topology=release.network(panel),release.topology(panel)
    files={partition_date(release,p):p for p in release.files(panel,'train')};raw_files={p.name:p for p in release.files(panel,'train',False)}
    template=release.template(1,panel,'train');profile=models['control'].profile;version=models['control'].feature_version
    frames={d:release.read(files[d]) for d in pconfig['ordinary_dates']};raw={d:release.read(raw_files[files[d].name]) for d in frames}
    for path,digest in study['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Public train source changed')
    ordinary_lookup={(r['date'],r['role']):r for r in study['ordinary_daily_stats']}
    segment_lookup={(r['key'],r['part'],r['role']):r for r in study['segment_rows']}
    case_lookup={(r['key'],r['role']):r for r in study['case_rows']}
    maximum_score=0.;maximum_prediction=0.;direct=[];ordinary_stats=[];segment_count=case_count=gate_count=ordinary_metrics=segment_metrics=0
    def check_score(measured,wanted,ordinary=False):
        nonlocal maximum_score
        key='daily_State' if ordinary else 'synthetic_reconstruction_score'
        error=abs(measured['synthetic_reconstruction_score']-wanted[key]);maximum_score=max(maximum_score,error)
        if measured['n']!=wanted['n'] or error>1e-12 or any(not np.isclose(measured[k],wanted[k],rtol=1e-12,atol=1e-6) for k in ['speed_sse','lane_flow_sse']):
            raise ValueError('Independent sufficient statistics/score differ')
    def check_arrays(path,digest,features,target,teacher,lanes,fresh,population):
        nonlocal maximum_prediction,gate_count
        if sha256_file(path)!=digest:raise ValueError('Saved prediction changed')
        times=features.times if features is not None else sorted(current_frame.timestamp.unique())
        positions=current_frame.groupby('link_id').milepost.median()
        links=features.links if features is not None else sorted(positions.index,key=lambda link:positions[link])
        grid=pd.MultiIndex.from_product([times,links]);indices=grid.get_indexer(pd.MultiIndex.from_frame(target[['timestamp','link_id']]))
        gate=independent_gate(current_frame,times,links,indices,pconfig['threshold_steps'])
        result={}
        with np.load(path) as a:
            if not np.array_equal(a['truth_total_units'],teacher) or not np.array_equal(a['station_lanes'],lanes) or not np.array_equal(a['target_grid_indices'],indices):
                raise ValueError('Teacher/normalization/target indices differ')
            if not np.array_equal(a['gate'],gate):raise ValueError('Independent availability RLE gate differs')
            if not np.array_equal(a['gated'],np.where(gate[:,None],a['expert'],a['control'])):raise ValueError('Derived policy mixture differs')
            gate_count+=1
            for role,model in models.items():
                result[role]=score(teacher,a[role],lanes)
                if fresh and len(target):
                    prediction=model.predict(features,target)[VALUES].to_numpy(float);error=float(np.abs(prediction-a[role]).max())
                    maximum_prediction=max(maximum_prediction,error)
                    if error>1e-8:raise ValueError('Direct serialized predictor differs')
                    direct.append({'population':population,'role':role,'maximum_total_unit_prediction_error':error})
        return result
    for record in study['ordinary_files']:
        date=record['date'];current_frame=frames[date];target=day_template(release,panel,'train',current_frame,template)
        features=StateFeatures(current_frame,net,topology,profile,version);teacher=target_truth(release,panel,target,raw[date])[VALUES].to_numpy(float,copy=True)
        lanes=release.station_lanes(panel,target);fresh=date in [pconfig['ordinary_dates'][0],pconfig['ordinary_dates'][-1]]
        measured=check_arrays(record['path'],record['sha256'],features,target,teacher,lanes,fresh,'ordinary_'+date)
        for role,value in measured.items():
            check_score(value,ordinary_lookup[(date,role)],True);ordinary_metrics+=1
            ordinary_stats.append({**value,'regime':current_frame.mask_regime.iloc[0],'role':role})
    for record in study['case_records']:
        case=next(c for c in pconfig['cases'] if c['name']==record['case']);dates=[r['date'] for r in record['segment_files']]
        masks,geometry=shifted_outage_masks([frames[d] for d in dates],panel,record['anchor_date'],record['mask_seed'],case)
        if geometry!=record['geometry']:raise ValueError('Metadata-selected shift geometry differs')
        accumulated={role:[] for role in models}
        for part,(date,mask,row) in enumerate(zip(dates,masks,record['segment_files'])):
            original=frames[date];teacher_frame=raw[date];cells=pd.MultiIndex.from_frame(original.loc[mask,['timestamp','link_id']].drop_duplicates())
            inside=pd.MultiIndex.from_frame(teacher_frame[['timestamp','link_id']]).isin(cells)
            target=teacher_frame.loc[inside&eligible(teacher_frame),['timestamp','station_id','link_id']].copy();target['panel']=panel
            target['mask_regime']=original.mask_regime.iloc[0];target=target[STATE_KEY]
            teacher=target_truth(release,panel,target,teacher_frame)[VALUES].to_numpy(float,copy=True);lane=release.station_lanes(panel,target)
            current_frame=remove_measurements(original,mask);fresh=record['anchor_date']==pconfig['anchor_dates'][0] and record['mask_seed']==pconfig['mask_seeds'][0]
            features=StateFeatures(current_frame,net,topology,profile,version) if fresh else None
            measured=check_arrays(row['path'],row['sha256'],features,target,teacher,lane,fresh,record['key']+f'_part{part}')
            for role,value in measured.items():check_score(value,segment_lookup[(record['key'],part,role)]);segment_metrics+=1;accumulated[role].append(value)
            segment_count+=1
        for role,values in accumulated.items():
            n=sum(v['n'] for v in values);sv=sum(v['speed_sse'] for v in values);sq=sum(v['lane_flow_sse'] for v in values)
            value=.54*max(0,1-(sv/n)**.5/25)+.46*max(0,1-(sq/n)**.5/600)
            check_score({'n':n,'speed_sse':sv,'lane_flow_sse':sq,'synthetic_reconstruction_score':value},case_lookup[(record['key'],role)])
        case_count+=1
        if case_count%24==0:print('gap_shift_verified',case_count,'case geometries',flush=True)
    for role in models:
        m=state_from_sufficient_stats([r for r in ordinary_stats if r['role']==role])
        if abs(m['S_state']-study['ordinary'][role]['S_state'])>1e-12:raise ValueError('Ordinary pooled State differs')
    result={'registered':registered,'source_snapshot':snapshot,'n_ordinary_saved_metrics_verified':ordinary_metrics,
        'n_case_geometries_regenerated':case_count,'n_day_segments_regenerated':segment_count,'n_segment_scores_verified':segment_metrics,
        'n_pooled_case_scores_verified':case_count*len(models),'n_independent_gate_arrays_exact':gate_count,
        'maximum_score_error':maximum_score,'n_direct_serialized_predictor_calls':len(direct),'direct_predictor_checks':direct,
        'maximum_prediction_error_total_units':maximum_prediction,'seconds':time.perf_counter()-start,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'All saved teacher/metric/gate arrays regenerated;44fixed direct role calls including serialized gate,not every model/case re-inferred. No independent retraining.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('gap_shift_verification_complete',case_count,segment_count,'direct',len(direct),'maxpred',maximum_prediction,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
