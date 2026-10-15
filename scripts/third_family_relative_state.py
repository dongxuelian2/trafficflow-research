"""Matched ordinary GRU/relative GRU on a third corridor family.

Uses the original released fit targets, not the native-remask label pool.
All model choices seal before the existing public assessment is read.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import VALUES
from trafficflow.state import StateModel
from trafficflow.bounded_state import BoundedState
from trafficflow.state_ensemble import StateEnsemble
from trafficflow.channel_state import ChannelStateEnsemble
from trafficflow.adaptive_graph_state import patch_score_loss
from trafficflow.recurrent_state import make_network,training_window,WindowedStateModel
from trafficflow.relative_position_state import make_relative_network,relative_distance,common_initial_sha256,RelativePositionStateModel
from trafficflow.state_augmentation import draw_training_plan
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol,paired_block_interval
from recurrent_state_experiment import neural_predictions,mix,summarize
from flow_transfer_common import PublicDays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=Path('reports/research')/(args.output.name+'.json')
    if report.exists() or args.output.exists():raise FileExistsError('Never overwrite third-family study')
    source_path=Path('reports/research/A47_adaptive_attention_D12_I405_S.json');source=json.loads(source_path.read_text());old=source['registered']['protocol']
    cache=Path(old['cache']);info=json.loads((cache/'prepared.json').read_text());baseline=StateModel.load(Path(old['baseline_file']))
    incumbent_path=Path('artifacts/research')/source_path.stem/'State_selected.pkl'
    if sha256_file(old['baseline_file'])!=old['baseline_sha256'] or sha256_file(incumbent_path)!=source['deployable_model_sha256']:raise ValueError('Existing model changed')
    incumbent=StateModel.load(incumbent_path);cached=StateModel.load(cache/'lightgbm.pkl')
    for model in [incumbent,cached]:
        if not baseline.profile.slots.equals(model.profile.slots) or not baseline.profile.fallback.equals(model.profile.fallback):raise ValueError('Profile differs')
    first_path=cache/'fit'/(info['dates']['fit'][0]+'.npz');links=np.load(first_path)['links'].tolist()
    metadata_path=Path(next(row['path'] for row in info['sources'] if row['part']=='fit' and row['cache']==first_path.as_posix()))
    meta=pd.read_parquet(metadata_path,columns=['link_id','milepost']);meta.link_id=meta.link_id.astype(str)
    position=meta.assign(milepost=pd.to_numeric(meta.milepost,errors='coerce')).groupby('link_id').milepost.mean().reindex(links).to_numpy(float)
    distance=relative_distance(position);seeds=[20261013,20261029];candidates=['nn_primary','geo_seed_mean','geo_speed_reference_flow']
    future=[f'2031-02-{day:02d}' for day in range(15,29)]
    if set(future)&{d for dates in info['dates'].values() for d in dates}:raise ValueError('Forward overlaps fitted partitions')
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'Third_family_original_targets_matched_GRU_relative_GRU','panel':old['panel'],'cache':cache.as_posix(),
        'source_study':source_path.as_posix(),'source_study_sha256':sha256_file(source_path),'prepared_sha256':sha256_file(cache/'prepared.json'),
        'baseline_file':old['baseline_file'],'baseline_sha256':old['baseline_sha256'],'incumbent_file':incumbent_path.as_posix(),'incumbent_sha256':sha256_file(incumbent_path),
        'geometry_source':metadata_path.as_posix(),'geometry_source_sha256':sha256_file(metadata_path),'geometry_columns':['link_id','milepost'],
        'link_order':links,'positions':position.tolist(),'graph_sha256':sha256_file(cache/'graph.npz'),
        'geometry_rule':'Same abs distance/median positive gap,cap20,two signed scalars initialized0;finite global attention',
        'seeds':seeds,'epochs':24,'steps_per_epoch':120,'batch_size':4,'hidden':48,'context':128,'core':64,
        'optimizer':'AdamW lr.001 wd1e-4 clip1','loss':'Unmodified ordinary patch State RMSE',
        'labels':'Original public released fit target masks only,no new remask views or extra teacher positions. Same exact patches/labels/cell exposures for each seed pair.',
        'baseline_scope':'Full-label existing LGB and frozen A47 selected TCN with density auxiliary loss;available dates match,actual repeated patch exposure differs from trees.',
        'initialization':'Common tensor bytes identical per seed;relative coefficients0. No dropout;only stored patch plans sample training.',
        'checkpoint':'Pure neural ordinary cal State every4epochs,first max','alpha':'0/.25/.5/.75/1 cal grid,lower-alpha tie',
        'reference_role':'strong_reference','reference_candidates':['incumbent','lgb'],'reference':'Max cal of existing frozen models;incumbent wins exact tie',
        'candidates':candidates,'hybrid_role':'geo_speed_reference_flow','hybrid':'New equal-seed speed plus strongest existing reference flow,no weight search',
        'primary':'Keep existing reference unless candidate cal gain>=.0005 and min regime gain>=-.001;max eligible cal. New plain GRU also eligible.',
        'assessment':'Old public development dates,read after four fits/checkpoints/alphas/reference/primary seal;not final/private',
        'future_dates':future,'future_scope':'Pre-fit fixed public research calendar,previously viewed for other studies,not final/private',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    np.savez_compressed(args.output/'geometry.npz',link_order=np.array(links),positions=position,distance_matrix=distance)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    graph_np=dict(np.load(cache/'graph.npz'));graph=[torch.tensor(graph_np[k],device=device) for k in ['incoming','outgoing','fd']]
    evaluator=PublicDays(old['panel'],baseline.profile,baseline.feature_version);hashes={};read_order=[];fits=[]
    for path in sorted((cache/'fit').glob('*.npz')):
        if sha256_file(path)!=source['cache_integrity'][path.as_posix()]:raise ValueError('Original fit cache changed')
        day=dict(np.load(path));day['date']=path.stem
        if day['links'].tolist()!=links or day['x'].shape[-1]!=12:raise ValueError('Grid/channels changed')
        fits.append(day);hashes[path.as_posix()]=sha256_file(path)
    groups=[[d for d in fits if str(d['regime'])==r] for r in ['R1','R2','R3']]
    if any(not rows for rows in groups):raise ValueError('All fit regimes required')
    models={'lgb':BoundedState(baseline),'incumbent':incumbent};training={};choices={};plans={}
    def load(part,seal=None):
        if part=='assessment' and (seal is None or sha256_file(args.output/'selection_sealed.json')!=seal):raise ValueError('Assessment before seal')
        read_order.append({'part':part,'prior_selection_seal_sha256':seal});days=[];values={'lgb':[],'incumbent':[]}
        for path in sorted((cache/part).glob('*.npz')):
            if part=='selection' and sha256_file(path)!=source['cache_integrity'][path.as_posix()]:raise ValueError('Cal cache changed')
            day=evaluator.load(path.stem,path);days.append(day);hashes[path.as_posix()]=sha256_file(path)
            if day['features'].links!=links:raise ValueError('Evaluation link order differs')
            for role in values:
                p=models[role].predict(day['features'],day['template'])[VALUES].to_numpy(float);p[:,1]/=day['target_link_lanes'];values[role].append(p)
            day['comparator']=values['lgb'][-1]
        return days,values
    cal,cal_predictions=load('selection')
    for seed in seeds:
        plan=draw_training_plan(seed,[[(d['date'],len(d['x'])) for d in rows] for rows in groups],24*120,probability=0.)
        plan_path=args.output/(f'plan_{seed}.npz');np.savez_compressed(plan_path,plan=plan);plans[str(seed)]={'path':plan_path.as_posix(),'sha256':sha256_file(plan_path)}
        prior_initial=None;prior_exposure=None;prior_patch=None
        for variant in ['control','geo']:
            role=f'{variant}_seed{seed}';torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            model=(make_network('tcn_gru',48) if variant=='control' else make_relative_network(position,48)).to(device)
            initial=common_initial_sha256(model);parameters=sum(p.numel() for p in model.parameters())
            if parameters!=(97307 if variant=='control' else 97309) or (prior_initial is not None and prior_initial!=initial):raise ValueError('Matched parameter initialization differs')
            prior_initial=initial;optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);history=[];best=-np.inf;tic=time.perf_counter()
            patch=hashlib.sha256();exposure={d['date']:np.zeros(d['mask'].shape,np.int32) for d in fits}
            for epoch in range(24):
                model.train();losses=[]
                for entries in plan[epoch*120:(epoch+1)*120]:
                    batch=[]
                    for group,index,center,view in entries:
                        if view!=-1:raise ValueError('Unexpected augmentation')
                        day=groups[group][index];left,first,last=training_window(len(day['x']),int(center),128,64);batch.append((day,left,first-left))
                        exposure[day['date']][first:last]+=day['mask'][first:last];patch.update((day['date']+':'+str(left)+':'+str(first)+':'+str(last)+';').encode())
                    x=torch.from_numpy(np.stack([d['x'][left:left+128].transpose(2,0,1) for d,left,first in batch])).to(device)
                    y=torch.from_numpy(np.stack([d['y'][left:left+128].transpose(2,0,1) for d,left,first in batch])).to(device)
                    mask=torch.from_numpy(np.stack([d['mask'][left:left+128] for d,left,first in batch])).to(device)
                    for j,(d,left,first) in enumerate(batch):mask[j,:first]=False;mask[j,first+64:]=False
                    optimizer.zero_grad(set_to_none=True);loss=patch_score_loss(model(x,*graph),y,mask)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
                if (epoch+1)%4==0:
                    model.eval();score=summarize(cal,neural_predictions(model,cal,graph,device,128,64),evaluator.factor)['S_state']
                    history.append({'epoch':epoch+1,'ordinary_cal_State':score,'mean_training_loss':float(np.mean(losses))})
                    if score>best:best=score;torch.save(model.state_dict(),args.output/(role+'.pt'))
                    print('third_family',role,'epoch',epoch+1,'cal',score,flush=True)
            if prior_exposure is not None and (prior_patch!=patch.hexdigest() or any(not np.array_equal(v,prior_exposure[k]) for k,v in exposure.items())):raise ValueError('Paired actual supervision differs')
            prior_exposure=exposure;prior_patch=patch.hexdigest();model.load_state_dict(torch.load(args.output/(role+'.pt'),map_location=device,weights_only=True));model.eval()
            neural=neural_predictions(model,cal,graph,device,128,64);grid=[{'alpha':a,'metrics':summarize(cal,mix(cal,neural,a),evaluator.factor)} for a in [0.,.25,.5,.75,1.]]
            choice=max(grid,key=lambda r:(r['metrics']['S_state'],-r['alpha']));choices[role]={'alpha':choice['alpha'],'blend_grid':grid};cal_predictions[role]=mix(cal,neural,choice['alpha'])
            deployed=(WindowedStateModel(baseline,model.state_dict(),evaluator.topology,choice['alpha'],'tcn_gru') if variant=='control' else RelativePositionStateModel(baseline,model.state_dict(),evaluator.topology,choice['alpha'],links,position))
            model_path=args.output/('State_'+role+'.pkl');deployed.save(model_path);models[role]=deployed;exposure_path=args.output/(role+'_exposure.npz');np.savez_compressed(exposure_path,**exposure)
            training[role]={'seconds':time.perf_counter()-tic,'history':history,'shared_initial_sha256':initial,'parameters':parameters,
                'patch_sequence_sha256':patch.hexdigest(),'total_label_exposures':sum(int(v.sum()) for v in exposure.values()),'exposure_sha256':sha256_file(exposure_path),
                'model_sha256':sha256_file(model_path),'selected_distance_strength':[float(block.distance_strength.detach().cpu()) for block in model.spatial] if variant=='geo' else None}
            del model;(args.output/'progress.json').write_text(json.dumps({'training':training,'choices':choices},indent=2,allow_nan=False),encoding='utf-8')
    for role,variant in [('nn_primary','control'),('geo_seed_mean','geo')]:
        members=[f'{variant}_seed{s}' for s in seeds];models[role]=StateEnsemble([models[m] for m in members],[.5,.5]);cal_predictions[role]=[(a+b)/2 for a,b in zip(cal_predictions[members[0]],cal_predictions[members[1]])]
    preliminary={r:summarize(cal,p,evaluator.factor) for r,p in cal_predictions.items()};reference_choice=max(['incumbent','lgb'],key=lambda r:preliminary[r]['S_state'])
    models['strong_reference']=models[reference_choice];cal_predictions['strong_reference']=cal_predictions[reference_choice]
    models['geo_speed_reference_flow']=ChannelStateEnsemble([models['geo_seed_mean'],models['strong_reference']],[[1,0],[0,1]])
    def hybrid(values):
        values['geo_speed_reference_flow']=[]
        for a,b in zip(values['geo_seed_mean'],values['strong_reference']):p=a.copy();p[:,1]=b[:,1];values['geo_speed_reference_flow'].append(p)
    hybrid(cal_predictions);cal_metrics={r:summarize(cal,p,evaluator.factor) for r,p in cal_predictions.items()};eligible=[]
    for role in candidates:
        a,b=cal_metrics[role],cal_metrics['strong_reference'];minimum=min(x['S_state']-y['S_state'] for x,y in zip(a['details'],b['details']))
        if a['S_state']-b['S_state']>=.0005 and minimum>=-.001:eligible.append(role)
    primary=max(eligible,key=lambda r:cal_metrics[r]['S_state']) if eligible else 'strong_reference'
    selection={'primary':primary,'choices':choices,'cal_metrics':cal_metrics,'reference_choice':reference_choice,'reference_role':'strong_reference','eligible_candidates':eligible}
    seal=args.output/'selection_sealed.json';seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8');seal_sha=sha256_file(seal);print('third_family_PRIMARY_SEALED',primary,'reference',reference_choice,flush=True)
    models['primary_export']=models[primary];exports={}
    for role,model in models.items():
        path=args.output/('State_'+role+'.pkl')
        if not path.exists():model.save(path)
        exports[role]={'path':path.as_posix(),'sha256':sha256_file(path)}
    assessment,values=load('assessment',seal_sha)
    for role in training:values[role]=[models[role].predict(d['features'],d['template'])[VALUES].to_numpy(float)/np.column_stack([np.ones(len(d['template'])),d['target_link_lanes']]) for d in assessment]
    for role,variant in [('nn_primary','control'),('geo_seed_mean','geo')]:values[role]=[(a+b)/2 for a,b in zip(values[f'{variant}_seed{seeds[0]}'],values[f'{variant}_seed{seeds[1]}'])]
    values['strong_reference']=values[reference_choice];hybrid(values);metrics={r:summarize(assessment,p,evaluator.factor) for r,p in values.items()};prediction_files=[]
    for part,days,outputs in [('selection',cal,cal_predictions),('assessment',assessment,values)]:
        folder=args.output/'predictions'/part;folder.mkdir(parents=True)
        for j,day in enumerate(days):
            path=folder/(day['date']+'.npz');np.savez_compressed(path,**{r:p[j] for r,p in outputs.items()});prediction_files.append({'part':part,'date':day['date'],'path':path.as_posix(),'sha256':sha256_file(path)})
    comparisons={r:{'pooled_State_gain':m['S_state']-metrics['strong_reference']['S_state'],'paired_day_diagnostic':paired_block_interval([d['daily_State'] for d in metrics['strong_reference']['days']],[d['daily_State'] for d in m['days']])} for r,m in metrics.items()}
    result={'registered':registered,'source_snapshot':snapshot,'prepared':info,'training':training,'plans':plans,'selection':selection,'selection_seal_sha256':seal_sha,
        'assessment':metrics,'comparisons':comparisons,'exports':exports,'prediction_files':prediction_files,'cache_sha256':hashes,'source_files_sha256':{metadata_path.as_posix():sha256_file(metadata_path),**evaluator.hashes},
        'phase_read_order':read_order,'seconds':time.perf_counter()-start,'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
        'source':evidence(gpu_used=device.type=='cuda'),'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Four new fits,two paired seeds,one additional direction/fold. Original masked fit target pool,not previous native-remask augmentation. Statistical geometry,not physical edges. Research public dates,not final/private.'}
    raw=json.dumps(result,indent=2,allow_nan=False);report.write_text(raw,encoding='utf-8');(args.output/'experiment.json').write_text(raw,encoding='utf-8')
    print('third_family_COMPLETE',primary,'reference',metrics['strong_reference']['S_state'],'plain',metrics['nn_primary']['S_state'],'geometry',metrics['geo_seed_mean']['S_state'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
