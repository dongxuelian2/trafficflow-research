"""Matched input-only augmentation ablation,with frozen future stress calendar."""
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
from trafficflow.data import Release,VALUES
from trafficflow.state import StateModel,StateFeatures
from trafficflow.state_ensemble import StateEnsemble
from trafficflow.adaptive_graph_state import reconstruction_input,patch_score_loss
from trafficflow.recurrent_state import make_network,WindowedStateModel,training_window
from trafficflow.state_augmentation import draw_training_plan
from trafficflow.experiments import partition_date,day_template,raw_for,target_truth
from trafficflow.research_eval import register_protocol,paired_block_interval
from trafficflow.runtime import capture_source_snapshot,sha256_file,evidence
from recurrent_state_experiment import neural_predictions,mix,summarize,transition_masks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-study',type=Path,required=True)
    parser.add_argument('--augmentation-cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Use new State experiment outputs')
    source=json.loads(args.source_study.read_text());config=source['registered']['protocol'];cache=Path(config['cache'])
    views_report=args.augmentation_cache/'cache.json';views=json.loads(views_report.read_text())
    if views['registered']['protocol']['prepared_sha256']!=sha256_file(cache/'prepared.json'):raise ValueError('Wrong fit augmentation calendar')
    for key in ['baseline','incumbent']:
        if sha256_file(config[key+'_file'])!=config[key+'_sha256']:raise ValueError('Frozen reference changed')
    future_dates=pd.date_range('2030-11-27','2030-12-10').strftime('%Y-%m-%d').tolist()
    if set(future_dates)&{d for v in source['prepared']['dates'].values() for d in v}:raise ValueError('Future calendar overlaps fit/cal')
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'Matched_GRU_input_only_long_outage_augmentation','panel':config['panel'],'source_study':args.source_study.as_posix(),
        'source_study_sha256':sha256_file(args.source_study),'augmentation_cache':args.augmentation_cache.as_posix(),
        'augmentation_cache_report_sha256':sha256_file(views_report),'seeds':config['seeds'],
        'architecture':'tcn_gru hidden48,context128/core64;both arms have97307parameters',
        'arms':['control','augmented'],'epochs':24,'steps_per_epoch':120,'batch_size':4,
        'optimizer':'AdamW lr.001 wd1e-4 clip1;identical initialized weights and original uniform regime/day/core plan',
        'augmentation':'Independent RNG,probability.5;uniform among4fit-only rebuilt3/6h30/70pct input views',
        'labels':'Only original fit y/mask;no new teacher targets;same original target exposure in both arms',
        'checkpoint':'Max ordinary pure-neural cal State every4epochs,first tie;no synthetic outcome selects checkpoint',
        'alpha_grid':[0.,.25,.5,.75,1.],'alpha_policy':'Perseed/arm ordinary cal only;then equal seed means',
        'primary':'Max ordinary cal over2arm means,frozen prior GRU,incumbent andLGB;never use assessment',
        'original_assessment':'Previously viewed public dates used only as regression after selection seal',
        'future_evaluation_dates':future_dates,'future_mask_seeds':[20261111,20261113,20261117],
        'future_cases':[[36,.3],[36,.7],[72,.3],[72,.7]],
        'future_policy':'All weight/alpha/primary choices sealed before these dates are opened;no outcome-based tuning',
        'scope':'One direction,public-train research only;not final/private;formal State/Physics/Total unknown',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    release=Release('data/kaggle_public');panel=config['panel'];net,topology=release.network(panel),release.topology(panel)
    baseline=StateModel.load(Path(config['baseline_file']));incumbent=StateModel.load(Path(config['incumbent_file']))
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train')
    graph_np=dict(np.load(cache/'graph.npz'));graph=[torch.tensor(graph_np[k],device=device) for k in ['incoming','outgoing','fd']]
    cache_hashes={};read_order=[];factor=None
    def load(part,seal=None):
        nonlocal factor
        if part=='assessment' and (seal is None or sha256_file(args.output/'selection_sealed.json')!=seal):raise ValueError('Assessment before seal')
        read_order.append({'part':part,'prior_selection_seal_sha256':seal});days=[]
        for path in sorted((cache/part).glob('*.npz')):
            if sha256_file(path)!=source['cache_sha256'][path.as_posix()]:raise ValueError('Original cache changed')
            day=dict(np.load(path,allow_pickle=False));day['date']=path.stem;cache_hashes[path.as_posix()]=sha256_file(path)
            if factor is None:factor=(net.lanes*net.length_km).reindex(day['links']).to_numpy(np.float32)
            if part!='fit':
                frame=release.read(files[path.stem]);features=StateFeatures(frame,net,topology,baseline.profile,version=baseline.feature_version)
                if not np.array_equal(reconstruction_input(features),day['x']):raise ValueError('Fresh inputs differ')
                target=day_template(release,panel,'train',frame,template)
                if not np.array_equal(features.lookup(target),day['target_indices']):raise ValueError('Target order differs')
                truth=target_truth(release,panel,target,next(raw_for(release,panel,[files[path.stem]])))[VALUES].to_numpy(float)
                truth[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
                if not np.array_equal(truth,day['station_truth']):raise ValueError('Raw teacher differs')
                p=baseline.predict(features,target)[VALUES].to_numpy(float)
                p[:,0]=np.minimum(p[:,0],target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                p[:,1]=np.minimum(p[:,1],target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/target.link_id.map(net.lanes).to_numpy(float)
                day['comparator']=p
                old=incumbent.predict(features,target)[VALUES].to_numpy(float);old[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
                day['incumbent']=old;day['diagnostic_masks']=transition_masks(day,target,next(raw_for(release,panel,[files[path.stem]])))
            days.append(day)
        return days
    fit,cal=load('fit'),load('selection');regimes=[[d for d in fit if str(d['regime'])==r] for r in ['R1','R2','R3']]
    view_rows={r['date']:r for r in views['rows']}
    for day in fit:
        row=view_rows[day['date']];path=Path(row['path'])
        if sha256_file(path)!=row['sha256'] or row['source_cache_sha256']!=cache_hashes[(cache/'fit'/(day['date']+'.npz')).as_posix()]:
            raise ValueError('Augmentation view/input provenance differs')
        with np.load(path) as a:day['augmented_x']=a['x'];day['outage_regions']=a['regions']
    incumbent._network=None
    source_folder=Path('artifacts/research')/args.source_study.stem
    def reference_predictions(phase,days):
        prior=[]
        for d in days:
            with np.load(source_folder/'predictions'/phase/(d['date']+'.npz')) as a:prior.append(a['tcn_gru_seed_mean'])
        return {'lgb':[d['comparator'] for d in days],'incumbent':[d['incumbent'] for d in days],'prior_GRU':prior}
    cal_predictions=reference_predictions('selection',cal);choices={};training={};initials={};plans={}
    for seed in config['seeds']:
        plan=draw_training_plan(seed,[[(d['date'],len(d['x'])) for d in group] for group in regimes],24*120)
        plan_path=args.output/(f'seed{seed}_plan.npz');np.savez_compressed(plan_path,plan=plan)
        plans[str(seed)]={'path':plan_path.as_posix(),'sha256':sha256_file(plan_path)}
        for arm in ['control','augmented']:
            name=f'seed{seed}_{arm}';torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            model=make_network('tcn_gru',48).to(device)
            initial=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items())).hexdigest()
            if seed in initials and initials[seed]!=initial:raise ValueError('Initial weights differ')
            initials[seed]=initial;optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
            exposure={d['date']:np.zeros(d['mask'].shape,np.int32) for d in fit};history=[];best=-np.inf;tic=time.perf_counter()
            patch_hash=hashlib.sha256();affected_supervision=0;all_supervision=0;augmented_samples=0
            for epoch in range(24):
                model.train();losses=[]
                for entries in plan[epoch*120:(epoch+1)*120]:
                    batch=[]
                    for group,index,center,view in entries:
                        d=regimes[group][index];at,first,last=training_window(len(d['x']),int(center),128,64)
                        use=arm=='augmented' and view>=0
                        x=d['augmented_x'][view] if use else d['x'];batch.append((d,at,first-at,x))
                        exposure[d['date']][first:last]+=d['mask'][first:last]
                        all_supervision+=int(d['mask'][first:last].sum())
                        if use:
                            augmented_samples+=1;affected_supervision+=int((d['mask'][first:last]&d['outage_regions'][view,first:last]).sum())
                        patch_hash.update((d['date']+':'+str(at)+':'+str(first)+':'+str(last)+';').encode())
                    x=torch.from_numpy(np.stack([a[at:at+128].transpose(2,0,1) for d,at,first,a in batch])).to(device)
                    y=torch.from_numpy(np.stack([d['y'][at:at+128].transpose(2,0,1) for d,at,first,a in batch])).to(device)
                    mask=torch.from_numpy(np.stack([d['mask'][at:at+128] for d,at,first,a in batch])).to(device)
                    for i,(d,at,first,a) in enumerate(batch):mask[i,:first]=False;mask[i,first+64:]=False
                    optimizer.zero_grad(set_to_none=True);loss=patch_score_loss(model(x,*graph),y,mask)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
                if (epoch+1)%4==0:
                    model.eval();score=summarize(cal,neural_predictions(model,cal,graph,device,128,64),factor)['S_state']
                    history.append({'epoch':epoch+1,'mean_training_loss':float(np.mean(losses)),'ordinary_cal_State':score})
                    if score>best:best=score;torch.save(model.state_dict(),args.output/(name+'.pt'))
                    print(panel,name,'epoch',epoch+1,'cal',score,flush=True)
            if patch_hash.hexdigest()!=source['training'][f'seed{seed}_tcn_gru']['patch_sequence_sha256']:raise ValueError('Original patch sampler not reproduced')
            model.load_state_dict(torch.load(args.output/(name+'.pt'),map_location=device,weights_only=True));model.eval()
            neural=neural_predictions(model,cal,graph,device,128,64)
            grid=[{'alpha':a,'metrics':summarize(cal,mix(cal,neural,a),factor)} for a in [0.,.25,.5,.75,1.]]
            chosen=max(grid,key=lambda row:(row['metrics']['S_state'],-row['alpha']))
            choices[name]={'seed':seed,'arm':arm,'alpha':chosen['alpha'],'blend_grid':grid}
            cal_predictions[name]=mix(cal,neural,chosen['alpha'])
            model_path=args.output/('State_'+name+'.pkl');WindowedStateModel(baseline,model.state_dict(),topology,chosen['alpha'],'tcn_gru').save(model_path)
            exposure_path=args.output/(name+'_exposure.npz');np.savez_compressed(exposure_path,**exposure)
            original_state=torch.load(source_folder/(f'seed{seed}_tcn_gru.pt'),map_location='cpu',weights_only=True)
            weight_error=max(float((v.detach().cpu()-original_state[k]).abs().max()) for k,v in model.state_dict().items()) if arm=='control' else None
            training[name]={'history':history,'seconds':time.perf_counter()-tic,'parameters':sum(p.numel() for p in model.parameters()),
                'initial_sha256':initial,'patch_sequence_sha256':patch_hash.hexdigest(),'model_sha256':sha256_file(model_path),
                'exposure_sha256':sha256_file(exposure_path),'total_target_grid_cells':sum(int(d['mask'].sum()) for d in fit),
                'unique_target_grid_cells_supervised':sum(int((v>0).sum()) for v in exposure.values()),
                'total_supervision_exposures':all_supervision,'augmented_samples':augmented_samples,
                'supervised_exposures_inside_extra_outage':affected_supervision,'control_max_checkpoint_weight_error_vs_original':weight_error}
            del model
            (args.output/'progress.json').write_text(json.dumps({'training':training,'choices':choices},indent=2,allow_nan=False),encoding='utf-8')
    for arm in ['control','augmented']:
        names=[f'seed{s}_{arm}' for s in config['seeds']]
        cal_predictions[arm+'_seed_mean']=[np.mean([cal_predictions[n][i] for n in names],axis=0) for i in range(len(cal))]
        ensemble=StateEnsemble([StateModel.load(args.output/('State_'+n+'.pkl')) for n in names],np.ones(len(names)))
        ensemble.save(args.output/('State_'+arm+'_seed_mean.pkl'))
    cal_metrics={n:summarize(cal,p,factor) for n,p in cal_predictions.items()}
    primary=max(['incumbent','lgb','prior_GRU','control_seed_mean','augmented_seed_mean'],key=lambda n:cal_metrics[n]['S_state'])
    selection={'choices':choices,'primary':primary,'cal_metrics':cal_metrics}
    seal=args.output/'selection_sealed.json';seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8');seal_sha=sha256_file(seal)
    print('augmentation_PRIMARY_SEALED',primary,flush=True)
    assessment=load('assessment',seal_sha);predictions=reference_predictions('assessment',assessment);incumbent._network=None
    for name,choice in choices.items():
        model=make_network('tcn_gru',48).to(device);model.load_state_dict(torch.load(args.output/(name+'.pt'),map_location=device,weights_only=True));model.eval()
        predictions[name]=mix(assessment,neural_predictions(model,assessment,graph,device,128,64),choice['alpha']);del model
    for arm in ['control','augmented']:
        names=[f'seed{s}_{arm}' for s in config['seeds']]
        predictions[arm+'_seed_mean']=[np.mean([predictions[n][i] for n in names],axis=0) for i in range(len(assessment))]
    metrics={n:summarize(assessment,p,factor) for n,p in predictions.items()}
    for phase,days,values in [('selection',cal,cal_predictions),('assessment',assessment,predictions)]:
        folder=args.output/'predictions'/phase;folder.mkdir(parents=True)
        for i,d in enumerate(days):np.savez_compressed(folder/(d['date']+'.npz'),**{n:p[i] for n,p in values.items()})
    comparisons={}
    for left,right in [(f'seed{s}_control',f'seed{s}_augmented') for s in config['seeds']]+[('control_seed_mean','augmented_seed_mean'),('prior_GRU','augmented_seed_mean')]:
        comparisons[right+'_vs_'+left]={'pooled_State_gain':metrics[right]['S_state']-metrics[left]['S_state'],
            'paired_day_diagnostic':paired_block_interval([r['daily_State'] for r in metrics[left]['days']],[r['daily_State'] for r in metrics[right]['days']])}
    result={'registered':registered,'source_snapshot':snapshot,'training':training,'plans':plans,'selection':selection,
        'selection_seal_sha256':seal_sha,'assessment':metrics,'comparisons':comparisons,'cache_sha256':cache_hashes,
        'phase_read_order':read_order,'seconds':time.perf_counter()-start,'source':evidence(gpu_used=device.type=='cuda'),
        'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Same architecture/targets/patches but input ablation changes available evidence. Reuses viewed original public development dates. '
        'Future calendar is fixed research evidence,not final/private. No synthetic test result selects models or alpha.'}
    text=json.dumps(result,indent=2,allow_nan=False);report.write_text(text,encoding='utf-8');(args.output/'experiment.json').write_text(text,encoding='utf-8')
    print('augmentation_COMPLETE',primary,{n:metrics[n]['S_state'] for n in ['incumbent','prior_GRU','control_seed_mean','augmented_seed_mean']},'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
