"""Independent coordinate removal,all scalar-flow training rows,and frozen exports."""
import argparse
import json
from pathlib import Path
import time
import zlib
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,VALUES,eligible
from trafficflow.state import StateModel,StateFeatures
from trafficflow.outage_stress import MEASUREMENTS
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from flow_transfer_common import context,PublicDays,load_exports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--table-only',action='store_true');parser.add_argument('--table-proof',type=Path);args=parser.parse_args()
    if args.table_only and args.table_proof:raise ValueError('Choose early table audit or reuse proof')
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite flow audit')
    source=Path('artifacts/research')/args.study.stem;registration_path=args.study.with_suffix('.registered.json');dataset_path=source/'dataset.json'
    registration=json.loads(registration_path.read_text());dataset=json.loads(dataset_path.read_text());cfg=registration['protocol']
    if args.table_only:
        study={'registered':registration,'dataset':dataset};views=json.loads((Path(cfg['views'])/'cache.json').read_text());original=json.loads(Path(cfg['source_study']).read_text())
    else:
        study,cfg,source,views,original=context(args.study)
        if dataset!=study['dataset'] or registration!=study['registered']:raise ValueError('Fit source registration/table changed')
        if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256'] or json.loads((source/'selection_sealed.json').read_text())!=study['selection']:raise ValueError('Selection changed')
        if study['phase_read_order']!=[{'part':'selection','prior_selection_seal_sha256':None},{'part':'assessment','prior_selection_seal_sha256':study['selection_seal_sha256']}]:raise ValueError('Assessment read order differs')
    old=original['registered']['protocol']
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Independent_flow_table_and_fixed_transfer_verification','source_study_sha256':None if args.table_only else sha256_file(args.study),
        'source_registration_sha256':sha256_file(registration_path),'dataset_manifest_sha256':sha256_file(dataset_path),
        'table_only':args.table_only,'reused_table_proof':None if args.table_proof is None else {'path':args.table_proof.as_posix(),'sha256':sha256_file(args.table_proof)},
        'scope':'All765table groups and8poisons in table phase;full phase all28dates x9exports. Reused table proof is hash checked,not a repeated rebuild.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    release=Release('data/kaggle_public');panel=cfg['panel'];net,topology=release.network(panel),release.topology(panel)
    baseline=StateModel.load(Path(old['baseline_file']));data={}
    for key,row in study['dataset']['files'].items():
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Mapped training array changed')
        data[key]=np.load(row['path'],mmap_mode='r')
    groups=study['dataset']['groups'];counts={r:sum(d['regime']==r for d in views['rows']) for r in ['R1','R2','R3']}
    original_count=sum(d['original_target_grid_cells'] for d in views['rows']);checks=[];poison=0;maximum_teacher=maximum_weight=0.
    if args.table_proof:
        prior=json.loads(args.table_proof.read_text());p=prior['registered']['protocol']
        if not p['table_only'] or p['source_registration_sha256']!=sha256_file(registration_path) or p['dataset_manifest_sha256']!=sha256_file(dataset_path) or prior['n_training_groups_rebuilt']!=len(views['rows'])*5 or prior['n_actual_60feature_poison_checks']!=8:raise ValueError('Early table proof incompatible')
        checks=prior['training_checks'];poison=prior['n_actual_60feature_poison_checks'];maximum_teacher=prior['maximum_raw_flow_teacher_difference_normalized_units'];maximum_weight=prior['maximum_weight_error']
        for path,digest in dataset['source_files_sha256'].items():
            if sha256_file(path)!=digest:raise ValueError('Table raw sources changed since early audit')
    for date_index,row in enumerate([] if args.table_proof else views['rows']):
        raw=release.read(row['raw_file']);masked=release.read(row['masked_file'])
        for key,digest_key in [('path','sha256'),('masked_file','masked_file_sha256'),('raw_file','raw_file_sha256'),('original_cache_path','original_cache_sha256')]:
            if sha256_file(row[key])!=row[digest_key]:raise ValueError('Raw training sources changed')
        if 'mask_regime' not in raw:raw['mask_regime']=row['regime']
        times=sorted(raw.timestamp.unique());links=sorted(raw.link_id.unique());ti=raw.timestamp.map({v:i for i,v in enumerate(times)}).to_numpy(int);li=raw.link_id.map({v:i for i,v in enumerate(links)}).to_numpy(int)
        good=eligible(raw);timestamps=pd.to_datetime(raw.timestamp,utc=True);original_arrays=dict(np.load(row['original_cache_path']));view_arrays=dict(np.load(row['path']))
        original_features=StateFeatures(masked,net,topology,baseline.profile,baseline.feature_version);indices=original_features.lookup(raw.loc[good])
        ngrid=len(original_features.grid);number=np.bincount(indices,minlength=ngrid);flow=raw.loc[good,'flow_vph'].to_numpy(float)/raw.loc[good,'link_id'].map(net.lanes).to_numpy(float)/600
        teacher=np.divide(np.bincount(indices,weights=flow,minlength=ngrid),number,out=np.zeros(ngrid),where=number>0)
        maximum_teacher=max(maximum_teacher,float(np.abs(teacher-view_arrays['y'][...,1].ravel()).max()))
        if maximum_teacher>1e-6:raise ValueError('Independent raw normalized flow teacher differs')
        for group in groups[date_index*5:(date_index+1)*5]:
            view=group['view'];a,b=group['start'],group['end'];mass=.5 if view<0 else .125
            if view<0:features=original_features;mask=original_arrays['mask'];labels=original_arrays['y'][...,1]
            else:
                seed=views['registered']['protocol']['view_seeds'][view]
                rng=np.random.default_rng(np.random.SeedSequence([seed,int(row['date'].replace('-','')),zlib.crc32(panel.encode())]))
                selected=(rng.random((len(times),len(links)))<views['registered']['protocol']['rates'][row['regime']])[ti,li]
                history=np.zeros(len(raw),bool);guard=np.zeros(len(raw),bool)
                for text in row['public_origins']:
                    delta=(timestamps-pd.Timestamp(text)).dt.total_seconds().to_numpy()/60;history|=(delta>=-60)&(delta<=0);guard|=(delta>=5)&(delta<=90)
                hidden=(selected&~history)|guard;targets=selected&~history&~guard&good
                def rebuild(poisoned=False):
                    frame=raw.copy(deep=True)
                    if poisoned:frame.loc[targets,[c for c in MEASUREMENTS if c!='pct_observed']]=1e8
                    for c in MEASUREMENTS:
                        if c in frame:frame.loc[hidden,c]=np.nan
                    for c,value in [('is_observed',0),('is_imputed',0),('is_score_eligible',0),('is_missing',1)]:
                        if c in frame:frame.loc[hidden,c]=value
                    return StateFeatures(frame,net,topology,baseline.profile,baseline.feature_version)
                features=rebuild();mask=np.zeros(original_arrays['mask'].shape,bool);mask.ravel()[features.lookup(raw.loc[targets])]=True;labels=view_arrays['y'][...,1]
                if not np.array_equal(mask,view_arrays['mask_point'][view]) or not np.array_equal(reconstruction_input(features),view_arrays['x_point'][view]):raise ValueError('Independent coordinate/removal differs')
                if date_index in [0,len(views['rows'])-1]:
                    pd.testing.assert_frame_equal(features.features,rebuild(True).features);poison+=1
            index=np.flatnonzero(mask.ravel());wanted_keys=np.c_[np.full(len(index),date_index),np.full(len(index),view),index].astype(np.int32)
            weight=mass/(3*counts[row['regime']]*len(index))*original_count
            error=float(np.abs(data['weight'][a:b]-weight).max());maximum_weight=max(maximum_weight,error)
            if error>1e-12 or not np.array_equal(data['row_keys'][a:b],wanted_keys) or not np.array_equal(data['features'][a:b],features.features.iloc[index].to_numpy(np.float32),equal_nan=True) or not np.array_equal(data['flow_teacher'][a:b],labels.ravel()[index].astype(float)*600) or not np.array_equal(data['flow_base'][a:b],features.base.reshape(-1,2)[index,1]):raise ValueError('Training feature/teacher/base/weight/key differs')
            checks.append({'date':row['date'],'view':view,'rows':len(index),'all60features_and_scalar_training_arrays_exact':True})
        if (date_index+1)%25==0:print('flow_audit_table',date_index+1,'dates',flush=True)
    del data
    if args.table_only:
        result={'registered':registered,'source_snapshot':snapshot,'n_training_groups_rebuilt':len(checks),'n_training_rows_verified':sum(r['rows'] for r in checks),
            'n_actual_60feature_poison_checks':poison,'maximum_raw_flow_teacher_difference_normalized_units':maximum_teacher,'maximum_weight_error':maximum_weight,
            'training_checks':checks,'seconds':time.perf_counter()-start,'formal_State':None,'formal_Physics':None,'formal_Total':None,
            'scope':'Only completed immutable training table. Concurrent model fit not inspected;no assessment or model accuracy claim.'}
        text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
        print('flow_table_audit_COMPLETE',len(checks),'groups','poison',poison,'seconds',result['seconds'],flush=True);return
    models=load_exports(study);evaluator=PublicDays(panel,baseline.profile,baseline.feature_version);calls=[];score_rows={};maximum=maximum_score=0.
    for seed in cfg['seeds']:
        components=models[f'flow_seed{seed}'].model.models
        if components[0].booster_.model_to_string()!=baseline.models[0].booster_.model_to_string() or components[1].booster_.num_trees()!=cfg['flow_tree_parameters']['n_estimators']:raise ValueError('Speed tree reuse or flow tree budget differs')
    for record in study['prediction_files']:
        part,date=record['part'],record['date'];cache=Path(old['cache'])/part/(date+'.npz')
        if sha256_file(cache)!=study['cache_sha256'][cache.as_posix()] or sha256_file(record['path'])!=record['sha256']:raise ValueError('Evaluation source/predictions changed')
        day=evaluator.load(date,cache)
        with np.load(record['path']) as saved:
            for role,model in models.items():
                key=study['selection']['primary'] if part=='selection' and role=='primary_export' else role
                prediction=model.predict(day['features'],day['template'])[VALUES].to_numpy(float);prediction[:,1]/=day['target_link_lanes'];error=float(np.abs(prediction-saved[key]).max());maximum=max(maximum,error)
                if error>1e-8:raise ValueError('Direct deployment differs')
                calls.append({'part':part,'date':date,'role':role,'maximum_lane_unit_error':error})
            metric_source=study['selection']['cal_metrics'] if part=='selection' else study['assessment']
            for role in metric_source:
                p=saved[role];v=p[:,0]-day['station_truth'][:,0];q=(p[:,1]-day['station_truth'][:,1])*day['target_link_lanes']/day['station_lanes']
                row={'date':date,'regime':day['regime'],'n':len(v),'speed_sse':float(v@v),'lane_flow_sse':float(q@q)}
                score=.54*max(0,1-np.sqrt(row['speed_sse']/row['n'])/25)+.46*max(0,1-np.sqrt(row['lane_flow_sse']/row['n'])/600)
                wanted=next(r for r in metric_source[role]['days'] if r['date']==date);error=abs(score-wanted['daily_State']);maximum_score=max(maximum_score,error)
                if row['n']!=wanted['n'] or error>1e-12 or any(not np.isclose(row[k],wanted[k],rtol=1e-12,atol=1e-6) for k in ['speed_sse','lane_flow_sse']):raise ValueError('Independent saved day metric differs')
                score_rows.setdefault((part,role),[]).append(row)
    for (part,role),rows in score_rows.items():
        wanted=(study['selection']['cal_metrics'] if part=='selection' else study['assessment'])[role]['S_state']
        if abs(state_from_sufficient_stats(rows)['S_state']-wanted)>1e-12:raise ValueError('Independent pooled State differs')
    a=state_from_sufficient_stats(score_rows[('selection','nn_primary')]);b=state_from_sufficient_stats(score_rows[('selection','transfer_fixed')])
    gain=b['S_state']-a['S_state'];minimum=min(v['S_state']-u['S_state'] for u,v in zip(a['details'],b['details']))
    primary='transfer_fixed' if gain>=.0005 and minimum>=-.001 else 'nn_primary'
    if primary!=study['selection']['primary'] or abs(gain-study['selection']['fixed_transfer_cal_gain'])>1e-12 or abs(minimum-study['selection']['minimum_regime_cal_gain'])>1e-12:raise ValueError('Fixed transfer cal gate differs')
    result={'registered':registered,'source_snapshot':snapshot,'n_training_groups_rebuilt':0 if args.table_proof else len(checks),
        'n_training_groups_previously_verified':len(checks) if args.table_proof else 0,'n_training_rows_verified':sum(r['rows'] for r in checks),
        'n_actual_60feature_poison_checks':poison,'maximum_raw_flow_teacher_difference_normalized_units':maximum_teacher,'maximum_weight_error':maximum_weight,
        'n_direct_predictor_calls':len(calls),'maximum_prediction_error_lane_units':maximum,'n_saved_role_day_metrics_recalculated':sum(map(len,score_rows.values())),
        'maximum_score_error':maximum_score,'cal_gate_and_reused_speed_trees_verified':True,
        'training_checks':checks,'prediction_checks':calls,'seconds':time.perf_counter()-start,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'scope':'Independent masks/removal,full tables/raw teachers,and saved deployment;not independent model retraining.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('flow_audit_COMPLETE',len(checks),'groups','poison',poison,'direct',len(calls),'maxpred',maximum,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
