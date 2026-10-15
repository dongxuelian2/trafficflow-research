"""Independently rebuild fit inputs/teachers, paired exposure and deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import VALUES
from trafficflow.state import neighbors
from trafficflow.recurrent_state import make_network
from trafficflow.relative_position_state import make_relative_network,common_initial_sha256
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from trafficflow.runtime import sha256_file,capture_source_snapshot
from flow_transfer_common import PublicDays,load_exports


def sufficient(day,p):
    v=p[:,0]-day['station_truth'][:,0];q=(p[:,1]-day['station_truth'][:,1])*day['target_link_lanes']/day['station_lanes']
    return {'date':day['date'],'regime':day['regime'],'n':len(v),'speed_sse':float(v@v),'lane_flow_sse':float(q@q)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite third-family verification')
    study=json.loads(args.study.read_text());cfg=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem;cache=Path(cfg['cache'])
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256'] or json.loads((source/'selection_sealed.json').read_text())!=study['selection']:raise ValueError('Selection seal changed')
    if study['phase_read_order']!=[{'part':'selection','prior_selection_seal_sha256':None},{'part':'assessment','prior_selection_seal_sha256':study['selection_seal_sha256']}]:raise ValueError('Assessment order differs')
    for key in ['baseline','incumbent']:
        if sha256_file(cfg[key+'_file'])!=cfg[key+'_sha256']:raise ValueError('Existing reference changed')
    if sha256_file(cfg['source_study'])!=cfg['source_study_sha256'] or sha256_file(cache/'prepared.json')!=cfg['prepared_sha256']:raise ValueError('Source changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Third_family_raw_fit_paired_exposure_and_selected_deployment_verification',
        'source_study_sha256':sha256_file(args.study),'scope':'All92fit raw inputs/teachers,independent metadata/graph/plans/four actual exposure archives,common initialization/selected pt,28days all exports and full selected-checkpoint cal alpha grid. Does not rerun training or unselected epochs.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if sha256_file(cfg['geometry_source'])!=cfg['geometry_source_sha256']:raise ValueError('Geometry source changed')
    meta=pd.read_parquet(cfg['geometry_source'],columns=['link_id','milepost']);meta.link_id=meta.link_id.astype(str)
    position=meta.assign(milepost=pd.to_numeric(meta.milepost,errors='coerce')).groupby('link_id').milepost.mean().reindex(cfg['link_order']).to_numpy(float)
    gaps=np.diff(np.unique(position));scale=np.median(gaps) if len(gaps) else 1.;distance=np.minimum(np.abs(position[:,None]-position[None,:])/scale,20).astype(np.float32)
    if not np.array_equal(position,cfg['positions']):raise ValueError('Registered positions differ')
    with np.load(source/'geometry.npz') as a:
        if a['link_order'].tolist()!=cfg['link_order'] or not np.array_equal(a['positions'],position) or not np.array_equal(a['distance_matrix'],distance):raise ValueError('Saved geometry differs')
    models=load_exports(study);evaluator=PublicDays(cfg['panel'],models['nn_primary'].profile,models['nn_primary'].feature_version);fits=[];fit_checks=[]
    for path in sorted((cache/'fit').glob('*.npz')):
        if sha256_file(path)!=study['cache_sha256'][path.as_posix()]:raise ValueError('Fit cache changed')
        day=evaluator.load(path.stem,path);shape=day['mask'].shape;teacher=np.zeros((*shape,2),np.float32);mask=np.zeros(shape,bool)
        aggregate=pd.DataFrame({'i':day['target_indices'],'v':day['station_truth'][:,0],'q':day['station_truth'][:,1]}).groupby('i')[['v','q']].mean()
        teacher.reshape(-1,2)[aggregate.index]=aggregate.to_numpy()/np.array([25.,600.]);mask.ravel()[day['target_indices']]=True
        if not np.array_equal(teacher,day['y']) or not np.array_equal(mask,day['mask']):raise ValueError('Independent full teacher/mask differs')
        if day['features'].links!=cfg['link_order']:raise ValueError('Fit link order differs')
        if not fits:
            n=len(position);rebuilt={}
            for direction in ['incoming','outgoing']:
                matrix=np.zeros((n,n),np.float32)
                for j,connected in enumerate(neighbors(evaluator.topology,cfg['link_order'],direction)):
                    if connected:matrix[j,connected]=1/len(connected)
                rebuilt[direction]=matrix
            lanes=evaluator.net.lanes.reindex(cfg['link_order']).to_numpy(float)
            rebuilt['fd']=np.stack([evaluator.net.free_speed_kmh.reindex(cfg['link_order']),evaluator.net.capacity_vph.reindex(cfg['link_order'])/lanes,evaluator.net.k_jam.reindex(cfg['link_order'])/lanes]).astype(np.float32)
            with np.load(cache/'graph.npz') as graph:
                if sha256_file(cache/'graph.npz')!=cfg['graph_sha256'] or any(not np.array_equal(graph[k],v) for k,v in rebuilt.items()):raise ValueError('Published graph/FD differs')
        fits.append({'date':day['date'],'mask':mask,'regime':str(day['regime'])});fit_checks.append({'date':day['date'],'raw_inputs_teacher_mask_exact':True})
        if len(fits)%20==0:print('third_family_verify_fit',len(fits),flush=True)
    groups=[[d for d in fits if d['regime']==r] for r in ['R1','R2','R3']];exposure_checks=[]
    for seed in cfg['seeds']:
        record=study['plans'][str(seed)]
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Plan changed')
        with np.load(record['path']) as a:plan=a['plan']
        rng=np.random.default_rng(seed);exposure={d['date']:np.zeros(d['mask'].shape,np.int32) for d in fits};patch=hashlib.sha256()
        if plan.shape!=(2880,4,4):raise ValueError('Plan budget differs')
        for entries in plan:
            for saved in entries:
                group=int(rng.integers(3));index=int(rng.integers(len(groups[group])));day=groups[group][index];center=int(rng.integers(len(day['mask'])))
                if not np.array_equal(saved,[group,index,center,-1]):raise ValueError('Independently sampled plan differs')
                first=min(max(center-32,0),len(day['mask'])-64);last=first+64;left=min(max(first-32,0),len(day['mask'])-128)
                exposure[day['date']][first:last]+=day['mask'][first:last];patch.update((day['date']+':'+str(left)+':'+str(first)+':'+str(last)+';').encode())
        torch.manual_seed(seed);base=make_network('tcn_gru',48);torch.manual_seed(seed);geo=make_relative_network(position,48)
        if any(not torch.equal(v,geo.state_dict()[k]) for k,v in base.state_dict().items()) or any(float(block.distance_strength.detach())!=0 for block in geo.spatial):raise ValueError('Initial state differs')
        initial=common_initial_sha256(base)
        if common_initial_sha256(geo)!=initial:raise ValueError('Common tensor order/hash differs')
        for variant in ['control','geo']:
            role=f'{variant}_seed{seed}';trained=study['training'][role];pt=torch.load(source/(role+'.pt'),map_location='cpu',weights_only=True);model=models[role]
            if trained['shared_initial_sha256']!=initial or trained['parameters']!=(97307 if variant=='control' else 97309) or sum(p.numel() for p in (base if variant=='control' else geo).parameters())!=trained['parameters']:raise ValueError('Initial budget differs')
            if sha256_file(source/(role+'_exposure.npz'))!=trained['exposure_sha256']:raise ValueError('Counter changed')
            with np.load(source/(role+'_exposure.npz')) as a:
                if set(a.files)!=set(exposure) or any(not np.array_equal(a[k],v) for k,v in exposure.items()):raise ValueError('Actual per-cell exposure differs')
            if trained['patch_sequence_sha256']!=patch.hexdigest() or trained['total_label_exposures']!=sum(int(v.sum()) for v in exposure.values()):raise ValueError('Patch hash/total differs')
            if set(pt)!=set(model.state) or any(not torch.equal(v,model.state[k]) for k,v in pt.items()) or model.alpha!=study['selection']['choices'][role]['alpha']:raise ValueError('Selected checkpoint/alpha differs')
            if variant=='geo':
                for i in range(2):
                    if not np.array_equal(pt[f'spatial.{i}.distance_matrix'].numpy(),distance) or float(pt[f'spatial.{i}.distance_strength'])!=trained['selected_distance_strength'][i]:raise ValueError('Selected geometry differs')
            exposure_checks.append({'role':role,'shared_initialization_and_full_cell_exposure_exact':True,'parameters':trained['parameters'],'total_label_exposures':trained['total_label_exposures']})
    rows={};grid_rows={};calls=[];pure_calls=0;maximum=maximum_score=0.
    for record in study['prediction_files']:
        part,date=record['part'],record['date'];path=cache/part/(date+'.npz')
        if sha256_file(path)!=study['cache_sha256'][path.as_posix()] or sha256_file(record['path'])!=record['sha256']:raise ValueError('Prediction inputs changed')
        day=evaluator.load(date,path);metrics=study['selection']['cal_metrics'] if part=='selection' else study['assessment']
        with np.load(record['path']) as a:
            reference=study['selection']['reference_choice']
            if not np.array_equal(a['strong_reference'],a[reference]) or not np.array_equal(a['geo_speed_reference_flow'][:,0],a['geo_seed_mean'][:,0]) or not np.array_equal(a['geo_speed_reference_flow'][:,1],a['strong_reference'][:,1]):raise ValueError('Fixed alias/channels differ')
            for name,variant in [('nn_primary','control'),('geo_seed_mean','geo')]:
                if not np.array_equal(a[name],(a[f'{variant}_seed{cfg["seeds"][0]}']+a[f'{variant}_seed{cfg["seeds"][1]}'])/2):raise ValueError('Seed weights differ')
            for role,model in models.items():
                key=study['selection']['primary'] if role=='primary_export' else role;p=model.predict(day['features'],day['template'])[VALUES].to_numpy(float);p[:,1]/=day['target_link_lanes']
                error=float(np.abs(p-a[key]).max());maximum=max(maximum,error)
                if error>1e-8:raise ValueError('Actual deployment differs: '+role)
                calls.append({'part':part,'date':date,'role':role,'maximum_lane_unit_error':error})
                if part=='selection' and role in study['training']:
                    alpha=model.alpha
                    try:
                        model.alpha=1.;neural=model.predict(day['features'],day['template'])[VALUES].to_numpy(float);neural[:,1]/=day['target_link_lanes']
                        # Training multiplies alpha into float32 before adding the float64 baseline.
                        neural=neural.astype(np.float32);pure_calls+=1
                    finally:model.alpha=alpha
                    for blend in [0.,.25,.5,.75,1.]:grid_rows.setdefault((role,blend),[]).append(sufficient(day,(1-blend)*a['lgb']+blend*neural))
            for role in metrics:
                row=sufficient(day,a[role]);wanted=next(d for d in metrics[role]['days'] if d['date']==date);score=.54*max(0,1-np.sqrt(row['speed_sse']/row['n'])/25)+.46*max(0,1-np.sqrt(row['lane_flow_sse']/row['n'])/600)
                error=abs(score-wanted['daily_State']);maximum_score=max(maximum_score,error)
                if row['n']!=wanted['n'] or error>1e-12 or any(not np.isclose(row[k],wanted[k],rtol=1e-12,atol=1e-6) for k in ['speed_sse','lane_flow_sse']):raise ValueError('Saved daily score differs')
                rows.setdefault((part,role),[]).append(row)
    for (part,role),values in rows.items():
        expected=(study['selection']['cal_metrics'] if part=='selection' else study['assessment'])[role]['S_state']
        if abs(state_from_sufficient_stats(values)['S_state']-expected)>1e-12:raise ValueError('Pooled score differs')
    for role in study['training']:
        scores={}
        for entry in study['selection']['choices'][role]['blend_grid']:
            alpha=entry['alpha'];actual=state_from_sufficient_stats(grid_rows[(role,alpha)]);scores[alpha]=actual['S_state']
            if abs(actual['S_state']-entry['metrics']['S_state'])>1e-12:raise ValueError('Selected-checkpoint alpha grid differs')
        if max(scores,key=lambda a:(scores[a],-a))!=study['selection']['choices'][role]['alpha']:raise ValueError('Alpha selection differs')
    cal={r:state_from_sufficient_stats(v) for (part,r),v in rows.items() if part=='selection'}
    if max(cfg['reference_candidates'],key=lambda r:cal[r]['S_state'])!=study['selection']['reference_choice']:raise ValueError('Strong existing reference choice differs')
    eligible=[r for r in cfg['candidates'] if cal[r]['S_state']-cal['strong_reference']['S_state']>=.0005 and min(a['S_state']-b['S_state'] for a,b in zip(cal[r]['details'],cal['strong_reference']['details']))>=-.001]
    primary=max(eligible,key=lambda r:cal[r]['S_state']) if eligible else 'strong_reference'
    if eligible!=study['selection']['eligible_candidates'] or primary!=study['selection']['primary']:raise ValueError('Primary cal gate differs')
    result={'registered':registered,'source_snapshot':snapshot,'public_geometry_and_graph_exact':True,'n_raw_fit_days_rebuilt':len(fit_checks),'fit_checks':fit_checks,
        'exposure_checks':exposure_checks,'n_direct_predictor_calls':len(calls),'n_additional_pure_cal_predictor_calls':pure_calls,'n_saved_role_day_metrics_recalculated':sum(map(len,rows.values())),
        'n_selected_checkpoint_alpha_metrics_recalculated':len(grid_rows),'maximum_prediction_error_lane_units':maximum,'maximum_score_error':maximum_score,
        'prediction_checks':calls,'source_files_sha256':evaluator.hashes,'seconds':time.perf_counter()-start,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'Four fits with matched two-seed comparisons. Rebuilt all fit inputs/teachers/plans/cell exposure and selected deployment/grid;does not replay optimizer updates or unselected checkpoints.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8')
    print('third_family_verified',len(calls),'calls','pure_cal',pure_calls,'maxpred',maximum,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
