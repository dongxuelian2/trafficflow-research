"""Native-rate data augmentation ablation with frozen matched old controls."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--views',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Never overwrite remask training')
    view_report=args.views/'cache.json';views=json.loads(view_report.read_text());source_path=Path(views['registered']['protocol']['source_study'])
    original=json.loads(source_path.read_text());config=original['registered']['protocol'];cache=Path(config['cache'])
    control_path=Path('reports/research/A87_I210_W_outage_augmentation.json');controls=json.loads(control_path.read_text())
    verified_path=Path('reports/research/A90_I210_W_augmentation_training_independent_replay.json');verified=json.loads(verified_path.read_text())
    if verified['registered']['protocol']['source_study_sha256']!=sha256_file(control_path):raise ValueError('Wrong frozen controls')
    future_dates=pd.date_range('2031-01-09','2031-01-22').strftime('%Y-%m-%d').tolist()
    if set(future_dates)&{d for group in original['prepared']['dates'].values() for d in group}:raise ValueError('Forward calendar overlaps fit/cal')
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'Native_rate_point_remask_and_mock_Queue_buffer_GRU_ablation','panel':config['panel'],
        'source_study':source_path.as_posix(),'source_study_sha256':sha256_file(source_path),'views_report_sha256':sha256_file(view_report),
        'views':args.views.as_posix(),'frozen_control_study_sha256':sha256_file(control_path),'control_verification_sha256':sha256_file(verified_path),
        'variants':['point','queue'],'seeds':config['seeds'],'epochs':24,'steps_per_epoch':120,'batch_size':4,'parameters':97307,
        'context':128,'core':64,'optimizer':'AdamW lr.001 wd1e-4 clip1',
        'sampling':'Same original uniform regime/day/core centers and shared augmentation plan perseed;probability.5 selects one of4views',
        'control_reuse':'A87control checkpoints/cal choices/exposure already tensor-exact with A72b and independently deployed;reuse explicitly,no new control fitting',
        'labels':'Original target labels on nonaug samples;new public raw-fit target positions on augmented samples. Same update budget,NOT same label population/count',
        'checkpoint':'Pure ordinary cal State every4epochs,first max;perseed blend0/.25/.5/.75/1 ordinary cal only',
        'primary':'Max cal of incumbent,LGB,prior_GRU,point_seed_mean,queue_seed_mean;equal seed weights fixed',
        'assessment':'Old public assessment is regression,read only after all choices sealed;no threshold tuning',
        'future_dates':future_dates,'future_scope':'New calendar for these trained policies,other public historical studies possible;not final/private',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    release=Release('data/kaggle_public');panel=config['panel'];net,topology=release.network(panel),release.topology(panel)
    for key in ['baseline','incumbent']:
        if sha256_file(config[key+'_file'])!=config[key+'_sha256']:raise ValueError('Reference changed')
    baseline=StateModel.load(Path(config['baseline_file']));incumbent=StateModel.load(Path(config['incumbent_file']))
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train')
    graph_np=dict(np.load(cache/'graph.npz'));graph=[torch.tensor(graph_np[k],device=device) for k in ['incoming','outgoing','fd']]
    factor=None;cache_hashes={};read_order=[]
    def load(part,seal=None):
        nonlocal factor
        if part=='assessment' and (seal is None or sha256_file(args.output/'selection_sealed.json')!=seal):raise ValueError('Assessment before seal')
        read_order.append({'part':part,'prior_selection_seal_sha256':seal});days=[]
        for path in sorted((cache/part).glob('*.npz')):
            if sha256_file(path)!=original['cache_sha256'][path.as_posix()]:raise ValueError('Original cache changed')
            day=dict(np.load(path));day['date']=path.stem;cache_hashes[path.as_posix()]=sha256_file(path)
            if factor is None:factor=(net.lanes*net.length_km).reindex(day['links']).to_numpy(np.float32)
            if part!='fit':
                frame=release.read(files[path.stem]);features=StateFeatures(frame,net,topology,baseline.profile,baseline.feature_version)
                if not np.array_equal(reconstruction_input(features),day['x']):raise ValueError('Fresh legal input differs')
                target=day_template(release,panel,'train',frame,template);raw=next(raw_for(release,panel,[files[path.stem]]))
                truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float);truth[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
                if not np.array_equal(truth,day['station_truth']) or not np.array_equal(features.lookup(target),day['target_indices']):raise ValueError('Raw evaluator differs')
                p=baseline.predict(features,target)[VALUES].to_numpy(float)
                p[:,0]=np.minimum(p[:,0],target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                p[:,1]=np.minimum(p[:,1],target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/target.link_id.map(net.lanes).to_numpy(float)
                day['comparator']=p;old=incumbent.predict(features,target)[VALUES].to_numpy(float);old[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
                day['incumbent']=old;day['diagnostic_masks']=transition_masks(day,target,raw)
            days.append(day)
        return days
    fit,cal=load('fit'),load('selection');regimes=[[d for d in fit if str(d['regime'])==r] for r in ['R1','R2','R3']]
    rows={r['date']:r for r in views['rows']}
    for day in fit:
        row=rows[day['date']]
        if sha256_file(row['path'])!=row['sha256'] or row['original_cache_sha256']!=cache_hashes[row['original_cache_path']]:raise ValueError('Remask views changed')
        with np.load(row['path']) as a:
            day['raw_y']=a['y']
            for variant in ['point','queue']:
                day['x_'+variant]=a['x_'+variant];day['mask_'+variant]=a['mask_'+variant]
    source_folder=Path('artifacts/research')/source_path.stem;incumbent._network=None
    reused={}
    for seed in config['seeds']:
        role=f'seed{seed}_control';row=verified['exports'][role]
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Frozen perseed control changed')
        reused[role]=row
    def references(part,days):
        result={'lgb':[d['comparator'] for d in days],'incumbent':[d['incumbent'] for d in days],'prior_GRU':[]}
        for seed in config['seeds']:result[f'seed{seed}_control']=[]
        for d in days:
            with np.load(source_folder/'predictions'/part/(d['date']+'.npz')) as a:
                result['prior_GRU'].append(a['tcn_gru_seed_mean'])
                for seed in config['seeds']:result[f'seed{seed}_control'].append(a[f'seed{seed}_tcn_gru'])
        return result
    cal_predictions=references('selection',cal);training={};choices={};plans={}
    for seed in config['seeds']:
        plan=draw_training_plan(seed,[[(d['date'],len(d['x'])) for d in group] for group in regimes],24*120)
        plan_path=args.output/(f'seed{seed}_plan.npz');np.savez_compressed(plan_path,plan=plan)
        plans[str(seed)]={'path':plan_path.as_posix(),'sha256':sha256_file(plan_path)}
        for variant in ['point','queue']:
            name=f'seed{seed}_{variant}';torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            model=make_network('tcn_gru',48).to(device)
            initial=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items())).hexdigest()
            if initial!=controls['training'][f'seed{seed}_control']['initial_sha256']:raise ValueError('Initial weights differ from control')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);history=[];best=-np.inf;tic=time.perf_counter()
            exposure={d['date']:np.zeros(d['mask'].shape,np.int32) for d in fit};patch_hash=hashlib.sha256();augmented_samples=0;augmented_label_exposures=0
            for epoch in range(24):
                model.train();losses=[]
                for entries in plan[epoch*120:(epoch+1)*120]:
                    batch=[]
                    for group,index,center,view in entries:
                        d=regimes[group][index];at,first,last=training_window(len(d['x']),int(center),128,64);use=view>=0
                        x=d['x_'+variant][view] if use else d['x'];y=d['raw_y'] if use else d['y'];mask=d['mask_'+variant][view] if use else d['mask']
                        batch.append((x,y,mask,at,first-at));exposure[d['date']][first:last]+=mask[first:last]
                        if use:augmented_samples+=1;augmented_label_exposures+=int(mask[first:last].sum())
                        patch_hash.update((d['date']+':'+str(at)+':'+str(first)+':'+str(last)+';').encode())
                    x=torch.from_numpy(np.stack([v[at:at+128].transpose(2,0,1) for v,y,m,at,first in batch])).to(device)
                    y=torch.from_numpy(np.stack([v[at:at+128].transpose(2,0,1) for x,v,m,at,first in batch])).to(device)
                    mask=torch.from_numpy(np.stack([v[at:at+128] for x,y,v,at,first in batch])).to(device)
                    for i,(x0,y0,m0,at,first) in enumerate(batch):mask[i,:first]=False;mask[i,first+64:]=False
                    optimizer.zero_grad(set_to_none=True);loss=patch_score_loss(model(x,*graph),y,mask)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
                if (epoch+1)%4==0:
                    model.eval();score=summarize(cal,neural_predictions(model,cal,graph,device,128,64),factor)['S_state']
                    history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),'ordinary_cal_State':score})
                    if score>best:best=score;torch.save(model.state_dict(),args.output/(name+'.pt'))
                    print(panel,name,'epoch',epoch+1,'cal',score,flush=True)
            if patch_hash.hexdigest()!=original['training'][f'seed{seed}_tcn_gru']['patch_sequence_sha256']:raise ValueError('Original patch centers differ')
            model.load_state_dict(torch.load(args.output/(name+'.pt'),map_location=device,weights_only=True));model.eval()
            neural=neural_predictions(model,cal,graph,device,128,64);grid=[{'alpha':a,'metrics':summarize(cal,mix(cal,neural,a),factor)} for a in [0.,.25,.5,.75,1.]]
            chosen=max(grid,key=lambda row:(row['metrics']['S_state'],-row['alpha']));choices[name]={'seed':seed,'variant':variant,'alpha':chosen['alpha'],'blend_grid':grid}
            cal_predictions[name]=mix(cal,neural,chosen['alpha']);model_path=args.output/('State_'+name+'.pkl')
            WindowedStateModel(baseline,model.state_dict(),topology,chosen['alpha'],'tcn_gru').save(model_path)
            exposure_path=args.output/(name+'_exposure.npz');np.savez_compressed(exposure_path,**exposure)
            potential={d['date']:d['mask']|d['mask_'+variant].any(axis=0) for d in fit}
            training[name]={'history':history,'seconds':time.perf_counter()-tic,'parameters':sum(p.numel() for p in model.parameters()),
                'initial_sha256':initial,'patch_sequence_sha256':patch_hash.hexdigest(),'model_sha256':sha256_file(model_path),
                'exposure_sha256':sha256_file(exposure_path),'unique_target_cells_supervised':sum(int((v>0).sum()) for v in exposure.values()),
                'potential_target_cells_in_all_views':sum(int(v.sum()) for v in potential.values()),
                'original_target_cells_supervised':sum(int(((exposure[d['date']]>0)&d['mask']).sum()) for d in fit),
                'new_distinct_target_cells_outside_original_mask':sum(int(((exposure[d['date']]>0)&~d['mask']).sum()) for d in fit),
                'total_label_exposures':sum(int(v.sum()) for v in exposure.values()),'augmented_samples':augmented_samples,
                'augmented_label_exposures':augmented_label_exposures}
            del model
            (args.output/'progress.json').write_text(json.dumps({'training':training,'choices':choices},indent=2,allow_nan=False),encoding='utf-8')
    for variant in ['point','queue']:
        names=[f'seed{s}_{variant}' for s in config['seeds']]
        cal_predictions[variant+'_seed_mean']=[np.mean([cal_predictions[n][i] for n in names],axis=0) for i in range(len(cal))]
        StateEnsemble([StateModel.load(args.output/('State_'+n+'.pkl')) for n in names],np.ones(len(names))).save(args.output/('State_'+variant+'_seed_mean.pkl'))
    cal_metrics={n:summarize(cal,p,factor) for n,p in cal_predictions.items()}
    primary=max(['incumbent','lgb','prior_GRU','point_seed_mean','queue_seed_mean'],key=lambda n:cal_metrics[n]['S_state'])
    selection={'choices':choices,'primary':primary,'cal_metrics':cal_metrics};seal=args.output/'selection_sealed.json'
    seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8');seal_sha=sha256_file(seal)
    print('native_remask_PRIMARY_SEALED',primary,flush=True)
    assessment=load('assessment',seal_sha);predictions=references('assessment',assessment);incumbent._network=None
    for name,choice in choices.items():
        model=make_network('tcn_gru',48).to(device);model.load_state_dict(torch.load(args.output/(name+'.pt'),map_location=device,weights_only=True));model.eval()
        predictions[name]=mix(assessment,neural_predictions(model,assessment,graph,device,128,64),choice['alpha']);del model
    for variant in ['point','queue']:
        names=[f'seed{s}_{variant}' for s in config['seeds']];predictions[variant+'_seed_mean']=[np.mean([predictions[n][i] for n in names],axis=0) for i in range(len(assessment))]
    metrics={n:summarize(assessment,p,factor) for n,p in predictions.items()}
    for phase,days,values in [('selection',cal,cal_predictions),('assessment',assessment,predictions)]:
        folder=args.output/'predictions'/phase;folder.mkdir(parents=True)
        for i,d in enumerate(days):np.savez_compressed(folder/(d['date']+'.npz'),**{n:p[i] for n,p in values.items()})
    pairs=[(f'seed{s}_control',f'seed{s}_{v}') for s in config['seeds'] for v in ['point','queue']]+[
        ('prior_GRU','point_seed_mean'),('prior_GRU','queue_seed_mean'),('point_seed_mean','queue_seed_mean')]
    comparisons={right+'_vs_'+left:{'pooled_State_gain':metrics[right]['S_state']-metrics[left]['S_state'],
        'paired_day_diagnostic':paired_block_interval([r['daily_State'] for r in metrics[left]['days']],[r['daily_State'] for r in metrics[right]['days']])} for left,right in pairs}
    result={'registered':registered,'source_snapshot':snapshot,'training':training,'plans':plans,'selection':selection,
        'selection_seal_sha256':seal_sha,'assessment':metrics,'comparisons':comparisons,'reused_controls':reused,
        'cache_sha256':cache_hashes,'phase_read_order':read_order,'seconds':time.perf_counter()-start,
        'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,'source':evidence(gpu_used=device.type=='cuda'),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'New label population and mask distribution with same initialization/patch/update budget;not architecture or equal-label-budget isolation. '
        'Frozen control model reuse is explicit. Old assessment viewed;new forward calendar fixed research evidence,not final/private.'}
    text=json.dumps(result,indent=2,allow_nan=False);report.write_text(text,encoding='utf-8');(args.output/'experiment.json').write_text(text,encoding='utf-8')
    print('native_remask_COMPLETE',primary,{n:metrics[n]['S_state'] for n in ['prior_GRU','point_seed_mean','queue_seed_mean']},'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
