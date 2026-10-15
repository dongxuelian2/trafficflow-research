"""Matched TCN vs TCN+bidirectional GRU,with fixed-window State inference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,VALUES,eligible
from trafficflow.state import StateModel,StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input,patch_score_loss
from trafficflow.recurrent_state import make_network,windowed_prediction,WindowedStateModel,training_window
from trafficflow.experiments import partition_date,day_template,raw_for,target_truth
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats,paired_block_interval
from trafficflow.runtime import capture_source_snapshot,sha256_file,evidence
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats


def transition_masks(day,template,raw):
    observed=raw.loc[eligible(raw),['timestamp','station_id','link_id','speed_kmh']].set_index(['timestamp','station_id','link_id'])
    previous=template[['timestamp','station_id','link_id']].copy()
    previous['timestamp']=(pd.to_datetime(previous.timestamp,utc=True)-pd.Timedelta(minutes=5)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    last=observed.speed_kmh.reindex(pd.MultiIndex.from_frame(previous)).to_numpy(float)
    delta=day['station_truth'][:,0]-last
    missing=day['x'][...,2]<.5;forward=np.zeros_like(missing,int);backward=np.zeros_like(missing,int)
    for i in range(len(missing)):forward[i]=missing[i]*(1+(forward[i-1] if i else 0))
    for i in range(len(missing)-1,-1,-1):backward[i]=missing[i]*(1+(backward[i+1] if i+1<len(missing) else 0))
    gap=np.where(missing,forward+backward-1,0).ravel()[day['target_indices']]
    return {'drop20':delta<=-20.,'recover20':delta>=20.,'missing60min':gap>=12,
        'missing180min':gap>=36,'congested50':day['station_truth'][:,0]<50.}


def summarize(days,predictions,factor):
    rows=[];increments=[];diagnostics={k:[] for k in days[0]['diagnostic_masks']}
    for day,prediction in zip(days,predictions):
        row=stats(day,prediction);row['date']=day['date'];rows.append(row)
        increments.append(increment_stats(day,prediction,factor))
        speed=prediction[:,0]-day['station_truth'][:,0]
        flow=(prediction[:,1]-day['station_truth'][:,1])*day['target_link_lanes']/day['station_lanes']
        for key,mask in day['diagnostic_masks'].items():
            diagnostics[key].append({'date':day['date'],'n':int(mask.sum()),'speed_sse':float(np.square(speed[mask]).sum()),
                'lane_flow_sse':float(np.square(flow[mask]).sum())})
    strata={}
    for key,values in diagnostics.items():
        n=sum(r['n'] for r in values)
        strata[key]={'n':n,'rmse_speed':float(np.sqrt(sum(r['speed_sse'] for r in values)/n)) if n else None,
            'rmse_lane_flow':float(np.sqrt(sum(r['lane_flow_sse'] for r in values)/n)) if n else None,'days':values}
    return {**state_from_sufficient_stats(rows),'days':rows,'diagnostic_strata':strata,
        'legacy_target_pair_increment_L1':sum(r['error_L1'] for r in increments)/max(sum(r['truth_increment_L1'] for r in increments),1e-9)}


def neural_predictions(model,days,graph,device,context,core):
    result=[];fd=graph[2].detach().cpu().numpy()
    for day in days:
        grid=windowed_prediction(model,day['x'],graph,device,context,core)
        grid[...,0]=np.clip(grid[...,0]*25,0,fd[0][None]*1.05)
        grid[...,1]=np.clip(grid[...,1]*600,0,fd[1][None]*1.05)
        result.append(grid.reshape(-1,2)[day['target_indices']])
    return result


def mix(days,neural,alpha):return [(1-alpha)*d['comparator']+alpha*p for d,p in zip(days,neural)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--baseline-file',type=Path,required=True);parser.add_argument('--incumbent-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seeds',type=int,nargs='+',default=[20261013,20261029])
    parser.add_argument('--epochs',type=int,default=24);parser.add_argument('--steps',type=int,default=120);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite State research')
    if len(args.seeds)<2 or len(set(args.seeds))!=len(args.seeds):raise ValueError('Distinct multiple optimizer seeds required')
    context,core,hidden=128,64,48;margin=(context-core)//2
    info=json.loads((args.cache/'prepared.json').read_text());report=Path('reports/research')/(args.output.name+'.json')
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'Fixed_context_graph_State_with_latent_bidirectional_GRU_branch','panel':info['panel'],'fold':info['fold'],
        'cache':args.cache.as_posix(),'baseline_file':args.baseline_file.as_posix(),'baseline_sha256':sha256_file(args.baseline_file),
        'incumbent_file':args.incumbent_file.as_posix(),'incumbent_sha256':sha256_file(args.incumbent_file),
        'architectures':['tcn','tcn_gru'],'seeds':args.seeds,'hidden':hidden,'context':context,'core':core,
        'epochs':args.epochs,'steps_per_epoch':args.steps,'batch_size':4,
        'optimizer':'AdamW lr=.001 weight_decay=.0001 clip_norm=1;uniform regime/fit day/context start',
        'matching':'All original TCN/attention parameters identical before training per seed;identical patch sequence and center supervision',
        'GRU':'Two layers,bidirectional hidden24x2,residual gain init.1;additional parameters,not a capacity-matched causal proof',
        'sampling_policy':'Uniform supervised center time;clip core then context at day boundaries;all day cells have positive sampling probability',
        'cache_aux_sha256':{(args.cache/n).as_posix():sha256_file(args.cache/n) for n in ['prepared.json','graph.npz','lightgbm.pkl']},
        'checkpoint':'Max pure neural cal State every4epochs,first wins ties','blend_grid':[0,.25,.5,.75,1.],
        'selection':'Per-seed architecture blend cal only;fixed equal-seed means;primary max cal over all plus incumbent/full-LGB',
        'assessment_read':'Only after all seeds/checkpoints/blends/primary sealed',
        'diagnostics':'Observed5min speed drop/recover>=20kmh;missing-speed input run>=12/36steps;observed speed<50',
        'truth_scope':'Public train raw labels evaluator/loss only;fixed-context predictor accepts no labels/masks',
        'assessment_status':'Previously viewed public-train development calendars;not new final/private evidence',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    release=Release('data/kaggle_public');panel=info['panel'];net,topology=release.network(panel),release.topology(panel)
    baseline=StateModel.load(args.baseline_file);incumbent=StateModel.load(args.incumbent_file)
    cached=StateModel.load(args.cache/'lightgbm.pkl')
    if not cached.profile.slots.equals(baseline.profile.slots) or not cached.profile.fallback.equals(baseline.profile.fallback):
        raise ValueError('Strong model and cached inputs use different profiles')
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train')
    graph_np=dict(np.load(args.cache/'graph.npz'));graph=[torch.tensor(graph_np[k],device=device) for k in ['incoming','outgoing','fd']]
    factor=None;hashes={};read_order=[]
    def load(part,seal=None):
        nonlocal factor
        if part=='assessment' and (seal is None or sha256_file(args.output/'selection_sealed.json')!=seal):raise ValueError('Assessment before seal')
        days=[];read_order.append({'part':part,'prior_selection_seal_sha256':seal})
        for path in sorted((args.cache/part).glob('*.npz')):
            day=dict(np.load(path,allow_pickle=False));day['date']=path.stem;hashes[path.as_posix()]=sha256_file(path)
            if day['x'].shape[-1]!=12:raise ValueError('Audited strict12 cache required')
            if factor is None:factor=(net.lanes*net.length_km).reindex(day['links']).to_numpy(np.float32)
            if part!='fit':
                frame=release.read(files[path.stem]);features=StateFeatures(frame,net,topology,baseline.profile,version=baseline.feature_version)
                if not np.array_equal(reconstruction_input(features),day['x']):raise ValueError('Fresh legal input differs')
                target=day_template(release,panel,'train',frame,template)
                if not np.array_equal(features.lookup(target),day['target_indices']):raise ValueError('Target ordering differs')
                raw=next(raw_for(release,panel,[files[path.stem]]));truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float)
                truth[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
                if not np.array_equal(truth,day['station_truth']):raise ValueError('Raw public train evaluator truth differs')
                p=baseline.predict(features,target)[VALUES].to_numpy(float)
                p[:,0]=np.minimum(p[:,0],target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                p[:,1]=np.minimum(p[:,1],target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/target.link_id.map(net.lanes).to_numpy(float)
                day['comparator']=p
                old=incumbent.predict(features,target)[VALUES].to_numpy(float);old[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
                day['incumbent']=old;day['diagnostic_masks']=transition_masks(day,target,raw)
            days.append(day)
        if not days:raise ValueError('Missing phase')
        return days
    fit,cal=load('fit'),load('selection');regimes={r:[d for d in fit if str(d['regime'])==r] for r in ['R1','R2','R3']}
    if any(not d for d in regimes.values()):raise ValueError('All fit regimes required')
    incumbent._network=None;torch.cuda.empty_cache() if device.type=='cuda' else None
    result={'registered':registered,'source_snapshot':snapshot,'prepared':info,'training':{},'selection':{},
        'device':str(device),'source_at_start':evidence(gpu_used=device.type=='cuda')}
    cal_predictions={'lgb':[d['comparator'] for d in cal],'incumbent':[d['incumbent'] for d in cal]}
    frozen_window=make_network('tcn',hidden).to(device);frozen_window.load_state_dict(incumbent.state);frozen_window.eval()
    cal_predictions['incumbent_windowed']=mix(cal,neural_predictions(frozen_window,cal,graph,device,context,core),incumbent.alpha)
    del frozen_window
    common_initial={};patch_hashes={};choices={}
    for seed in args.seeds:
        for kind in ['tcn','tcn_gru']:
            name=f'seed{seed}_{kind}';torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            model=make_network(kind,hidden).to(device);shared={k:v for k,v in model.state_dict().items() if not k.startswith('temporal')}
            initial=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in shared.items())).hexdigest()
            if seed in common_initial and common_initial[seed]!=initial:raise ValueError('Shared model initialization differs')
            common_initial[seed]=initial;rng=np.random.default_rng(seed);patch_hash=hashlib.sha256()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);history=[];best=-np.inf;tic=time.perf_counter()
            exposure={d['date']:np.zeros(d['mask'].shape,np.int32) for d in fit}
            for epoch in range(args.epochs):
                model.train();losses=[]
                for step in range(args.steps):
                    batch=[]
                    for _ in range(4):
                        group=regimes[['R1','R2','R3'][rng.integers(3)]];day=group[rng.integers(len(group))]
                        at,first,last=training_window(len(day['x']),int(rng.integers(len(day['x']))),context,core)
                        batch.append((day,at,first-at));exposure[day['date']][first:last]+=day['mask'][first:last]
                        patch_hash.update((day['date']+':'+str(at)+':'+str(first)+':'+str(last)+';').encode())
                    x=torch.from_numpy(np.stack([d['x'][at:at+context].transpose(2,0,1) for d,at,first in batch])).to(device)
                    y=torch.from_numpy(np.stack([d['y'][at:at+context].transpose(2,0,1) for d,at,first in batch])).to(device)
                    mask=torch.from_numpy(np.stack([d['mask'][at:at+context] for d,at,first in batch])).to(device)
                    for index,(day,at,first) in enumerate(batch):mask[index,:first]=False;mask[index,first+core:]=False
                    optimizer.zero_grad(set_to_none=True);prediction=model(x,*graph);loss=patch_score_loss(prediction,y,mask)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
                if (epoch+1)%4==0 or epoch+1==args.epochs:
                    model.eval();neural=neural_predictions(model,cal,graph,device,context,core);score=summarize(cal,neural,factor)
                    history.append({'epoch':epoch+1,'mean_training_loss':float(np.mean(losses)),'cal_State':score['S_state']})
                    if score['S_state']>best:best=score['S_state'];torch.save(model.state_dict(),args.output/(name+'.pt'))
                    print(panel,name,'epoch',epoch+1,'cal_State',score['S_state'],flush=True)
            patch_hashes[name]=patch_hash.hexdigest()
            if kind=='tcn_gru' and patch_hashes[name]!=patch_hashes[f'seed{seed}_tcn']:raise ValueError('Matched patches differ')
            model.load_state_dict(torch.load(args.output/(name+'.pt'),map_location=device,weights_only=True));model.eval()
            neural=neural_predictions(model,cal,graph,device,context,core)
            grid=[{'alpha':a,'metrics':summarize(cal,mix(cal,neural,a),factor)} for a in [0.,.25,.5,.75,1.]]
            chosen=max(grid,key=lambda r:(r['metrics']['S_state'],-r['alpha']));choices[name]={'kind':kind,'seed':seed,'alpha':chosen['alpha']}
            cal_predictions[name]=mix(cal,neural,chosen['alpha'])
            result['training'][name]={'history':history,'seconds':time.perf_counter()-tic,'parameters':sum(p.numel() for p in model.parameters()),
                'shared_initial_sha256':initial,'patch_sequence_sha256':patch_hashes[name],
                'total_available_target_grid_cells':sum(int(d['mask'].sum()) for d in fit),
                'unique_target_grid_cells_supervised':sum(int((v>0).sum()) for v in exposure.values()),
                'selected_temporal_gain':float(model.temporal_gain.detach().cpu()) if kind=='tcn_gru' else None}
            exposure_path=args.output/(name+'_exposure.npz');np.savez_compressed(exposure_path,**exposure)
            result['training'][name]['exposure_sha256']=sha256_file(exposure_path)
            result['selection'][name]={'blend_grid':grid,'choice':choices[name]}
            deployed=WindowedStateModel(baseline,model.state_dict(),topology,chosen['alpha'],kind,hidden,context,core)
            deployed.save(args.output/('State_'+name+'.pkl'))
            del model,deployed
            (args.output/'progress.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for kind in ['tcn','tcn_gru']:
        name=kind+'_seed_mean';members=[f'seed{s}_{kind}' for s in args.seeds]
        cal_predictions[name]=[np.mean([cal_predictions[m][i] for m in members],axis=0) for i in range(len(cal))]
    cal_metrics={name:summarize(cal,p,factor) for name,p in cal_predictions.items()}
    eligible_names=['incumbent','lgb']+list(choices)+['tcn_seed_mean','tcn_gru_seed_mean']
    primary=max(eligible_names,key=lambda name:cal_metrics[name]['S_state'])
    selection={'choices':choices,'primary':primary,'cal_metrics':cal_metrics,'seed_means':'Equal means of per-seed cal-selected mixtures'}
    seal=args.output/'selection_sealed.json';seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8')
    seal_sha=sha256_file(seal);print(panel,'PRIMARY SEALED',primary,'cal',cal_metrics[primary]['S_state'],flush=True)
    assessment=load('assessment',seal_sha);incumbent._network=None
    predictions={'lgb':[d['comparator'] for d in assessment],'incumbent':[d['incumbent'] for d in assessment]}
    frozen_window=make_network('tcn',hidden).to(device);frozen_window.load_state_dict(incumbent.state);frozen_window.eval()
    predictions['incumbent_windowed']=mix(assessment,neural_predictions(frozen_window,assessment,graph,device,context,core),incumbent.alpha);del frozen_window
    for name,choice in choices.items():
        model=make_network(choice['kind'],hidden).to(device)
        model.load_state_dict(torch.load(args.output/(name+'.pt'),map_location=device,weights_only=True));model.eval()
        predictions[name]=mix(assessment,neural_predictions(model,assessment,graph,device,context,core),choice['alpha']);del model
    for kind in ['tcn','tcn_gru']:
        members=[f'seed{s}_{kind}' for s in args.seeds]
        predictions[kind+'_seed_mean']=[np.mean([predictions[m][i] for m in members],axis=0) for i in range(len(assessment))]
    assessment_metrics={name:summarize(assessment,p,factor) for name,p in predictions.items()}
    for phase,days,values in [('selection',cal,cal_predictions),('assessment',assessment,predictions)]:
        folder=args.output/'predictions'/phase;folder.mkdir(parents=True)
        for i,day in enumerate(days):np.savez_compressed(folder/(day['date']+'.npz'),**{k:v[i] for k,v in values.items()},
            **{'mask_'+k:v for k,v in day['diagnostic_masks'].items()})
    def paired(left,right):return paired_block_interval([r['daily_State'] for r in assessment_metrics[left]['days']],
        [r['daily_State'] for r in assessment_metrics[right]['days']])
    comparisons={name:{'gain_vs_incumbent':assessment_metrics[name]['S_state']-assessment_metrics['incumbent']['S_state'],
        'paired_days_vs_incumbent':paired('incumbent',name)} for name in predictions if name not in ['incumbent','incumbent_windowed']}
    for seed in args.seeds:
        a,b=f'seed{seed}_tcn',f'seed{seed}_tcn_gru'
        comparisons[b]['matched_GRU_minus_TCN_gain']=assessment_metrics[b]['S_state']-assessment_metrics[a]['S_state']
        comparisons[b]['paired_days_vs_matched_TCN']=paired(a,b)
    result.update(selection=selection,selection_seal_sha256=seal_sha,assessment=assessment_metrics,comparisons=comparisons,
        cache_sha256=hashes,phase_read_order=read_order,seconds=time.perf_counter()-start,
        gpu_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
        model_sha256={p.name:sha256_file(p) for p in args.output.glob('*.pt')},
        exported_model_sha256={p.name:sha256_file(p) for p in args.output.glob('State_*.pkl')},
        source_at_end=evidence(gpu_used=device.type=='cuda'),formal_State=None,formal_Physics=None,formal_Total=None,
        limitations='Two optimizer seeds share existing fit/cal/assessment dates and fixed full-LGB. All are previously viewed public-train development. '
        'GRU adds capacity;not exact parameter-budget isolation. Fixed finite context changes inference versus incumbent full-day. '
        'Observed transitions are noisy measurement diagnostics,not latent event truth. Increment score is incomplete physical proxy only.')
    text=json.dumps(result,indent=2,allow_nan=False);(args.output/'experiment.json').write_text(text,encoding='utf-8');report.write_text(text,encoding='utf-8')
    print('COMPLETE',panel,'primary',primary,'incumbent',assessment_metrics['incumbent']['S_state'],
        'primary_State',assessment_metrics[primary]['S_state'],'seconds',result['seconds'],'GPU_MiB',result['gpu_peak_allocated_MiB'],flush=True)


if __name__=='__main__':main()
