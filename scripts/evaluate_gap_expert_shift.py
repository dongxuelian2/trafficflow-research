"""Fixed input-only expert on a new calendar and shifted outage laws."""
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
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats,paired_block_interval
from trafficflow.runtime import capture_source_snapshot,sha256_file,evidence
from public_outage_stress import metric


def pooled_case(rows):
    n=sum(r['n'] for r in rows);sv=sum(r['speed_sse'] for r in rows);sq=sum(r['lane_flow_sse'] for r in rows)
    score=.54*max(0,1-np.sqrt(sv/n)/25)+.46*max(0,1-np.sqrt(sq/n)/600) if n else None
    return {'n':n,'speed_sse':sv,'lane_flow_sse':sq,'synthetic_reconstruction_score':float(score) if score is not None else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite shifted expert evaluation')
    policy=json.loads(args.policy.read_text());config=policy['registered']['protocol']
    if sha256_file(policy['export']['path'])!=policy['export']['sha256']:raise ValueError('Frozen gate changed')
    if sha256_file('src/trafficflow/outage_shift.py')!=config['geometry_sha256'] or sha256_file('src/trafficflow/gap_gated_state.py')!=config['gate_source_sha256']:
        raise ValueError('Registered gate/generator changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Fixed_gap_expert_new_calendar_and_untrained_outage_geometries','source_policy':args.policy.as_posix(),
        'source_policy_sha256':sha256_file(args.policy),'policy_protocol_sha256':policy['registered']['protocol_sha256'],
        'roles':['lgb','control','expert','gated'],'inference':'Three predictor roles;gated derived from control/expert using serialized policy.gate only',
        'ordinary_dates':config['ordinary_dates'],'anchor_dates':config['anchor_dates'],'cases':config['cases'],'mask_seeds':config['mask_seeds'],
        'cross_midnight':'Pool raw station SSE/count across two independently reconstructed day segments before computing a single case score',
        'selection':'No new threshold/weight/date/seed selection from outcomes;all8cases retained',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    folder.mkdir(parents=True);(folder/'ordinary').mkdir();(folder/'cases').mkdir()
    snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    models={}
    for name,row in config['inference_exports'].items():
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Frozen reference/expert changed')
        models[name]=StateModel.load(Path(row['path']))
    gate_model=StateModel.load(Path(policy['export']['path']));roles=['lgb','control','expert','gated']
    profile=gate_model.profile;version=gate_model.feature_version
    if any(m.feature_version!=version or not m.profile.slots.equals(profile.slots) or not m.profile.fallback.equals(profile.fallback) for m in models.values()):
        raise ValueError('Policies have incompatible public input profiles')
    release=Release('data/kaggle_public');panel=config['panel'];net,topology=release.network(panel),release.topology(panel)
    files={partition_date(release,p):p for p in release.files(panel,'train')};raw_files={p.name:p for p in release.files(panel,'train',False)}
    template=release.template(1,panel,'train');frames={};teachers={};source_hashes={}
    for date in config['ordinary_dates']:
        path=files[date];raw_path=raw_files[path.name];frames[date]=release.read(path);teachers[date]=release.read(raw_path)
        source_hashes[path.as_posix()]=sha256_file(path);source_hashes[raw_path.as_posix()]=sha256_file(raw_path)
    def predictions(features,target):
        values={n:m.predict(features,target)[VALUES].to_numpy(float,copy=True) for n,m in models.items()}
        selected=gate_model.gate(features,target)
        values['gated']=np.where(selected[:,None],values['expert'],values['control'])
        return values,selected
    ordinary=[];ordinary_files=[];case_rows=[];segments=[];case_records=[]
    for date in config['ordinary_dates']:
        frame=frames[date];features=StateFeatures(frame,net,topology,profile,version);target=day_template(release,panel,'train',frame,template)
        truth=target_truth(release,panel,target,teachers[date])[VALUES].to_numpy(float,copy=True);lane=release.station_lanes(panel,target)
        values,selected=predictions(features,target);path=folder/'ordinary'/(date+'.npz')
        np.savez_compressed(path,truth_total_units=truth,station_lanes=lane,target_grid_indices=features.lookup(target),gate=selected,**values)
        ordinary_files.append({'date':date,'path':path.as_posix(),'sha256':sha256_file(path),'n_gate_targets':int(selected.sum())})
        for role,p in values.items():
            m=metric(truth,p,lane);ordinary.append({'date':date,'role':role,'regime':frame.mask_regime.iloc[0],
                'n':m['n'],'speed_sse':m['speed_sse'],'lane_flow_sse':m['lane_flow_sse'],'daily_State':m['synthetic_reconstruction_score']})
    print('shift_ordinary_complete',len(ordinary_files),'dates',flush=True)
    for anchor in config['anchor_dates']:
        for seed in config['mask_seeds']:
            for case in config['cases']:
                dates=[anchor]
                if case['placement']=='cross_midnight':dates.append((pd.Timestamp(anchor)+pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
                masks,geometry=shifted_outage_masks([frames[d] for d in dates],panel,anchor,seed,case)
                key=f'{anchor}_{seed}_{case["name"]}';case_stats={r:[] for r in roles};record={'key':key,'anchor_date':anchor,
                    'mask_seed':seed,'case':case['name'],'geometry':geometry,'segment_files':[]}
                gate_count=0;teacher_count=0;congested_count=0
                for index,(date,mask) in enumerate(zip(dates,masks)):
                    frame=frames[date];raw=teachers[date];selected_cells=pd.MultiIndex.from_frame(frame.loc[mask,['timestamp','link_id']].drop_duplicates())
                    inside=pd.MultiIndex.from_frame(raw[['timestamp','link_id']]).isin(selected_cells)
                    target=raw.loc[inside&eligible(raw),['timestamp','station_id','link_id']].copy();target['panel']=panel
                    target['mask_regime']=frame.mask_regime.iloc[0];target=target[STATE_KEY]
                    hidden=StateFeatures(remove_measurements(frame,mask),net,topology,profile,version)
                    truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float,copy=True);lane=release.station_lanes(panel,target)
                    values,chosen=predictions(hidden,target) if len(target) else ({r:np.empty((0,2)) for r in roles},np.zeros(0,bool))
                    path=folder/'cases'/(key+f'_part{index}.npz');np.savez_compressed(path,truth_total_units=truth,station_lanes=lane,
                        target_grid_indices=hidden.lookup(target),gate=chosen,**values)
                    record['segment_files'].append({'date':date,'path':path.as_posix(),'sha256':sha256_file(path),'n':len(target),'gate_targets':int(chosen.sum())})
                    gate_count+=int(chosen.sum());teacher_count+=len(target);congested_count+=int((truth[:,0]<50).sum())
                    for role,p in values.items():
                        m=metric(truth,p,lane);case_stats[role].append(m)
                        segments.append({'key':key,'date':date,'anchor_date':anchor,'mask_seed':seed,'case':case['name'],'part':index,'role':role,**m})
                for role,values in case_stats.items():case_rows.append({'key':key,'anchor_date':anchor,'mask_seed':seed,'case':case['name'],
                    'role':role,**pooled_case(values)})
                record.update(n=teacher_count,gate_targets=gate_count,congested50_teacher_targets=congested_count,
                    start_in_night=geometry['segments'][0]['start_index']<72 or geometry['segments'][0]['start_index']>=240)
                case_records.append(record)
        (folder/'progress.json').write_text(json.dumps({'completed_anchor':anchor,'case_rows':case_rows},indent=2,allow_nan=False),encoding='utf-8')
        print('gap_expert_shift',anchor,'complete',flush=True)
    ordinary_metrics={r:state_from_sufficient_stats([v for v in ordinary if v['role']==r]) for r in roles}
    ordinary_comparisons={r:{'pooled_State_gain':ordinary_metrics[r]['S_state']-ordinary_metrics['control']['S_state'],
        'paired_day_diagnostic':paired_block_interval([v['daily_State'] for v in ordinary if v['role']=='control'],[v['daily_State'] for v in ordinary if v['role']==r])}
        for r in roles if r!='control'}
    summary={}
    for case in config['cases']:
        subset=[r for r in case_rows if r['case']==case['name']];records=[r for r in case_records if r['case']==case['name']]
        if any(r['n']==0 for r in subset):summary[case['name']]={'unknown':'Zero-support case retained,not resampled'};continue
        scores={r:float(np.mean([v['synthetic_reconstruction_score'] for v in subset if v['role']==r])) for r in roles}
        comparisons={r:paired_block_interval([np.mean([v['synthetic_reconstruction_score'] for v in subset if v['anchor_date']==d and v['role']=='control']) for d in config['anchor_dates']],
            [np.mean([v['synthetic_reconstruction_score'] for v in subset if v['anchor_date']==d and v['role']==r]) for d in config['anchor_dates']]) for r in roles if r!='control'}
        summary[case['name']]={'mean_case_scores':scores,'gated_minus_control':scores['gated']-scores['control'],
            'paired_anchor_date_diagnostics':comparisons,'gate_target_fraction':sum(r['gate_targets'] for r in records)/sum(r['n'] for r in records),
            'n_start_in_night':sum(r['start_in_night'] for r in records),'n_case_geometries':len(records)}
    result={'registered':registered,'source_snapshot':snapshot,'ordinary':ordinary_metrics,'ordinary_daily_stats':ordinary,
        'ordinary_comparisons':ordinary_comparisons,'ordinary_files':ordinary_files,'stress_summary':summary,'case_rows':case_rows,
        'segment_rows':segments,'case_records':case_records,'source_files_sha256':source_hashes,
        'n_stress_case_geometries':len(case_records),'n_day_segments':sum(len(r['segment_files']) for r in case_records),
        'seconds':time.perf_counter()-start,'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
        'source':evidence(gpu_used=torch.cuda.is_available()),'threshold_or_models_changed':False,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'One corridor,public research dates with previously viewed history possible. Missingness laws are own generators,not accidents/official targets. '
        'Eight scenarios share14anchor days;cross-midnight cases additionally share adjacent days. Unadjusted moving7day block intervals are conditional. '
        'Gated arrays derived from3predictor roles here;serialized model is separately deployed/replayed. Full comparison wall time is not gated-policy latency.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('gap_expert_shift_COMPLETE',{r:ordinary_metrics[r]['S_state'] for r in roles},
        {k:v.get('gated_minus_control') for k,v in summary.items()},'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
