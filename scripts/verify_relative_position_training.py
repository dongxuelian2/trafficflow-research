"""Rebuild static distances,shared initialization,cell exposure,and saved policies."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import VALUES
from trafficflow.recurrent_state import make_network
from trafficflow.relative_position_state import make_relative_network,common_initial_sha256
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from flow_transfer_common import load_exports,PublicDays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite relative verification')
    study=json.loads(args.study.read_text());cfg=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem
    control=json.loads(Path(cfg['control_study']).read_text());control_folder=Path('artifacts/research')/Path(cfg['control_study']).stem
    views=json.loads((Path(cfg['views'])/'cache.json').read_text());old=json.loads(Path(cfg['source_study']).read_text())['registered']['protocol'];cache=Path(old['cache'])
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256'] or json.loads((source/'selection_sealed.json').read_text())!=study['selection']:raise ValueError('Seal changed')
    if study['phase_read_order']!=[{'part':'selection','prior_selection_seal_sha256':None},{'part':'assessment','prior_selection_seal_sha256':study['selection_seal_sha256']}]:raise ValueError('Assessment order changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_relative_geometry_and_matched_training_verification',
        'source_study_sha256':sha256_file(args.study),'scope':'Static metadata/distance matrix,both common initializations and full actual cell exposures,selected checkpoints/alpha,all28dates x9export calls and saved scores. No refit controls/unselected grid replay.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if sha256_file(cfg['geometry_source'])!=cfg['geometry_source_sha256']:raise ValueError('Geometry source changed')
    meta=pd.read_parquet(cfg['geometry_source'],columns=cfg['geometry_columns']);meta.link_id=meta.link_id.astype(str)
    position=meta.assign(milepost=pd.to_numeric(meta.milepost,errors='coerce')).groupby('link_id').milepost.mean().reindex(cfg['link_order']).to_numpy(float)
    positive=np.diff(np.unique(position));scale=float(np.median(positive)) if len(positive) else 1.
    distance=np.clip(np.abs(np.subtract.outer(position,position))/scale,0,20).astype(np.float32)
    if not np.array_equal(position,cfg['positions']):raise ValueError('Registered public positions differ')
    with np.load(source/'geometry.npz') as a:
        if a['link_order'].tolist()!=cfg['link_order'] or not np.array_equal(a['positions'],position) or not np.array_equal(a['distance_matrix'],distance):raise ValueError('Saved static geometry differs')
    fits=[];view_rows={r['date']:r for r in views['rows']}
    for path in sorted((cache/'fit').glob('*.npz')):
        if sha256_file(path)!=study['cache_sha256'][path.as_posix()]:raise ValueError('Fit changed')
        day=dict(np.load(path));day['date']=path.stem;row=view_rows[path.stem]
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Remask inputs changed')
        with np.load(row['path']) as a:day['mask_queue']=a['mask_queue']
        fits.append(day)
    groups=[[d for d in fits if str(d['regime'])==r] for r in ['R1','R2','R3']];checks=[]
    for seed in cfg['seeds']:
        role=f'geo_seed{seed}';paired=f'seed{seed}_queue';record=study['training'][role]
        torch.manual_seed(seed);base=make_network('tcn_gru',48);torch.manual_seed(seed);candidate=make_relative_network(position,48)
        if any(not torch.equal(v,candidate.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('Shared state values differ')
        initial=hashlib.sha256(b''.join(k.encode()+v.numpy().tobytes() for k,v in base.state_dict().items())).hexdigest()
        if initial!=record['shared_initial_sha256'] or common_initial_sha256(candidate)!=initial or initial!=control['training'][paired]['initial_sha256']:raise ValueError('Shared initialization hash differs')
        if sum(p.numel() for p in candidate.parameters())!=97309 or any(float(m.distance_strength.detach())!=0 for m in candidate.spatial):raise ValueError('New parameter budget/initial value differs')
        plan_record=study['plans'][str(seed)]
        if plan_record!=control['plans'][str(seed)] or sha256_file(plan_record['path'])!=plan_record['sha256']:raise ValueError('Plan changed')
        with np.load(plan_record['path']) as a:plan=a['plan']
        exposure={d['date']:np.zeros(d['mask'].shape,np.int32) for d in fits};patch=hashlib.sha256()
        for entries in plan:
            for group,index,center,view in entries:
                d=groups[group][index];first=min(max(int(center)-32,0),len(d['mask'])-64);last=first+64;left=min(max(first-32,0),len(d['mask'])-128)
                mask=d['mask_queue'][view] if view>=0 else d['mask'];exposure[d['date']][first:last]+=mask[first:last]
                patch.update((d['date']+':'+str(left)+':'+str(first)+':'+str(last)+';').encode())
        if sha256_file(source/(role+'_exposure.npz'))!=record['exposure_sha256']:raise ValueError('Actual counter changed')
        with np.load(source/(role+'_exposure.npz')) as a,np.load(control_folder/(paired+'_exposure.npz')) as b:
            if any(not np.array_equal(v,a[k]) or not np.array_equal(v,b[k]) for k,v in exposure.items()):raise ValueError('Actual/control supervision mismatch')
        if patch.hexdigest()!=record['patch_sequence_sha256'] or patch.hexdigest()!=control['training'][paired]['patch_sequence_sha256'] or sum(int(v.sum()) for v in exposure.values())!=record['total_label_exposures']:raise ValueError('Patch sequence or totals differ')
        checks.append({'role':role,'shared_initialization_and_full_cell_exposure_exact':True,'parameters':97309,'total_label_exposures':record['total_label_exposures']})
    models=load_exports(study);evaluator=PublicDays(cfg['panel'],models['nn_primary'].profile,models['nn_primary'].feature_version);maximum=maximum_score=0.;calls=[];rows={}
    for seed in cfg['seeds']:
        role=f'geo_seed{seed}';record=study['training'][role]
        if sha256_file(source/('State_'+role+'.pkl'))!=record['model_sha256']:raise ValueError('Per-fit export changed')
        pt=torch.load(source/(role+'.pt'),map_location='cpu',weights_only=True)
        if set(pt)!=set(models[role].state) or any(not torch.equal(v,models[role].state[k]) for k,v in pt.items()) or models[role].alpha!=study['selection']['choices'][role]['alpha']:raise ValueError('Selected state/alpha differs')
        for i in range(2):
            if not np.array_equal(pt[f'spatial.{i}.distance_matrix'].numpy(),distance) or float(pt[f'spatial.{i}.distance_strength'])!=record['selected_distance_strength'][i]:raise ValueError('Checkpoint geometry/strength differs')
    for record in study['prediction_files']:
        part,date=record['part'],record['date'];path=cache/part/(date+'.npz')
        if sha256_file(path)!=study['cache_sha256'][path.as_posix()] or sha256_file(record['path'])!=record['sha256']:raise ValueError('Evaluator bytes changed')
        day=evaluator.load(date,path);metrics=study['selection']['cal_metrics'] if part=='selection' else study['assessment']
        with np.load(record['path']) as a:
            if not np.array_equal(a['geo_speed_tree_flow'][:,0],a['geo_seed_mean'][:,0]) or not np.array_equal(a['geo_speed_tree_flow'][:,1],a['strong_channel'][:,1]):raise ValueError('Hybrid changed fixed channel weights')
            for role,model in models.items():
                key=study['selection']['primary'] if role=='primary_export' else role
                p=model.predict(day['features'],day['template'])[VALUES].to_numpy(float);p[:,1]/=day['target_link_lanes'];error=float(np.abs(p-a[key]).max());maximum=max(maximum,error)
                if error>1e-8:raise ValueError('Frozen predictor differs')
                calls.append({'part':part,'date':date,'role':role,'maximum_lane_unit_error':error})
            for role in metrics:
                p=a[role];v=p[:,0]-day['station_truth'][:,0];q=(p[:,1]-day['station_truth'][:,1])*day['target_link_lanes']/day['station_lanes'];n=len(v)
                row={'date':date,'regime':day['regime'],'n':n,'speed_sse':float(v@v),'lane_flow_sse':float(q@q)}
                score=.54*max(0,1-np.sqrt(row['speed_sse']/n)/25)+.46*max(0,1-np.sqrt(row['lane_flow_sse']/n)/600);wanted=next(r for r in metrics[role]['days'] if r['date']==date);error=abs(score-wanted['daily_State']);maximum_score=max(maximum_score,error)
                if n!=wanted['n'] or error>1e-12 or any(not np.isclose(row[k],wanted[k],rtol=1e-12,atol=1e-6) for k in ['speed_sse','lane_flow_sse']):raise ValueError('Independent score differs')
                rows.setdefault((part,role),[]).append(row)
    for (part,role),values in rows.items():
        wanted=(study['selection']['cal_metrics'] if part=='selection' else study['assessment'])[role]['S_state']
        if abs(state_from_sufficient_stats(values)['S_state']-wanted)>1e-12:raise ValueError('Pooled metrics differ')
    reference=state_from_sufficient_stats(rows[('selection','strong_channel')]);eligible=[]
    for role in ['geo_seed_mean','geo_speed_tree_flow']:
        metrics=state_from_sufficient_stats(rows[('selection',role)]);minimum=min(a['S_state']-b['S_state'] for a,b in zip(metrics['details'],reference['details']))
        if metrics['S_state']-reference['S_state']>=.0005 and minimum>=-.001:eligible.append(role)
    primary=max(eligible,key=lambda r:state_from_sufficient_stats(rows[('selection',r)])['S_state']) if eligible else 'strong_channel'
    if eligible!=study['selection']['eligible_candidates'] or primary!=study['selection']['primary']:raise ValueError('Strong-reference cal gate differs')
    result={'registered':registered,'source_snapshot':snapshot,'exposure_checks':checks,'public_geometry_exact':True,'n_direct_predictor_calls':len(calls),
        'maximum_prediction_error_lane_units':maximum,'maximum_score_error':maximum_score,'n_saved_role_day_metrics_recalculated':sum(map(len,rows.values())),
        'prediction_checks':calls,'seconds':time.perf_counter()-start,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'Static geometry,common state/counters and selected deployment/scoring verified;not physical connectivity,control retraining or all unselected cal checkpoints/alpha.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8');print('relative_verified','calls',len(calls),'maxpred',maximum,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
