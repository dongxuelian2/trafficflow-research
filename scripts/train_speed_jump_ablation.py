"""Matched two-seed GRU with one fixed speed-jump auxiliary objective."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.state import StateModel
from trafficflow.state_ensemble import StateEnsemble
from trafficflow.data import VALUES
from trafficflow.adaptive_graph_state import patch_score_loss
from trafficflow.recurrent_state import make_network,WindowedStateModel,training_window
from trafficflow.speed_jump_loss import speed_jump_pair_loss
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol,paired_block_interval
from recurrent_state_experiment import neural_predictions,mix,summarize
from flow_transfer_common import PublicDays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Never overwrite jump study')
    control_path=Path('reports/research/A104_I5_N_native_remask_ablation.json');control=json.loads(control_path.read_text())
    replay_path=Path('reports/research/A107_I5_N_native_remask_training_independent_replay.json');replay=json.loads(replay_path.read_text())
    audit_path=Path('reports/research/A122_fit_transition_population_audit.json');proof_path=Path('reports/research/A123_fit_transition_independent_verification.json');proof=json.loads(proof_path.read_text())
    if proof['registered']['protocol']['source_study_sha256']!=sha256_file(audit_path) or proof['n_raw_fit_days']!=245:raise ValueError('Fit diagnostic not verified')
    if replay['registered']['protocol']['source_study_sha256']!=sha256_file(control_path):raise ValueError('Matched controls not verified')
    cfg=control['registered']['protocol'];original_path=Path(cfg['source_study']);original=json.loads(original_path.read_text());old=original['registered']['protocol'];cache=Path(old['cache'])
    views_path=Path(cfg['views'])/'cache.json';views=json.loads(views_path.read_text());baseline=StateModel.load(Path(old['baseline_file']))
    if sha256_file(views_path)!=cfg['views_report_sha256'] or sha256_file(old['baseline_file'])!=old['baseline_sha256'] or control['selection']['primary']!='queue_seed_mean':raise ValueError('Matched source changed')
    future=[f'2031-02-{day:02d}' for day in range(15,29)]
    if set(future)&{d for dates in original['prepared']['dates'].values() for d in dates}:raise ValueError('Future overlaps model calendar')
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Matched_supervised_speed_jump_pair_GRU_loss','panel':cfg['panel'],
        'source_study':original_path.as_posix(),'source_study_sha256':sha256_file(original_path),'views':cfg['views'],'view_report_sha256':sha256_file(views_path),
        'control_study':control_path.as_posix(),'control_study_sha256':sha256_file(control_path),'control_replay_sha256':sha256_file(replay_path),
        'fit_audit_sha256':sha256_file(audit_path),'fit_verification_sha256':sha256_file(proof_path),'graph_sha256':sha256_file(cache/'graph.npz'),
        'variant':'queue,as frozen control','parameters':97307,'seeds':cfg['seeds'],'epochs':24,'steps_per_epoch':120,'batch_size':4,'context':128,'core':64,
        'optimizer':'Identical AdamW lr.001 wd1e-4 clip1','sampling':'Exact stored control plans/inputs/labels/core masks;no added supervised cells',
        'loss':'Original patch State RMSE + .05*speed-difference RMSE on consecutive core cells both already supervised and float32 abs(delta truth)>=.8 normalized',
        'loss_budget':'Extra pair objective,not extra labels. Pair counts stored at endpoint;no flow pair term or other coefficient trial',
        'controls':'A104 queue models reused from verified exports,not refitted','checkpoint':'Pure ordinary cal State every4epochs,first maximum',
        'alpha':'Original cal grid0/.25/.5/.75/1,first lower-alpha maximum','primary':'New2seed mean only if cal gain>=.0005 and no regime loss>.001,else keep original NN',
        'assessment':'Viewed old public assessment,read after both new fits and selection seal','future_dates':future,
        'future_scope':'Registered before fitting. Public research dates,other historical studies possible,not final/private',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    graph_np=dict(np.load(cache/'graph.npz'));graph=[torch.tensor(graph_np[k],device=device) for k in ['incoming','outgoing','fd']]
    fits=[];view_rows={r['date']:r for r in views['rows']};hashes={};read_order=[]
    for path in sorted((cache/'fit').glob('*.npz')):
        if sha256_file(path)!=control['cache_sha256'][path.as_posix()]:raise ValueError('Original fit changed')
        day=dict(np.load(path));day['date']=path.stem;row=view_rows[path.stem]
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Augmented fit changed')
        with np.load(row['path']) as a:day.update(raw_y=a['y'],x_queue=a['x_queue'],mask_queue=a['mask_queue'])
        fits.append(day);hashes[path.as_posix()]=sha256_file(path)
    groups=[[d for d in fits if str(d['regime'])==r] for r in ['R1','R2','R3']]
    evaluator=PublicDays(cfg['panel'],baseline.profile,baseline.feature_version);control_folder=Path('artifacts/research')/control_path.stem
    def load(part,seal=None):
        if part=='assessment' and (seal is None or sha256_file(args.output/'selection_sealed.json')!=seal):raise ValueError('Assessment before both fits and seal')
        read_order.append({'part':part,'prior_selection_seal_sha256':seal});days=[];predictions={r:[] for r in ['nn_primary']+[f'control_seed{s}' for s in cfg['seeds']]}
        for path in sorted((cache/part).glob('*.npz')):
            if sha256_file(path)!=control['cache_sha256'][path.as_posix()]:raise ValueError('Evaluation cache changed')
            d=evaluator.load(path.stem,path);p=baseline.predict(d['features'],d['template'])[VALUES].to_numpy(float)
            p[:,0]=np.minimum(p[:,0],d['template'].link_id.map(evaluator.net.free_speed_kmh).to_numpy(float)*1.05)
            p[:,1]=np.minimum(p[:,1],d['template'].link_id.map(evaluator.net.capacity_vph).to_numpy(float)*1.05)/d['target_link_lanes'];d['comparator']=p;days.append(d)
            hashes[path.as_posix()]=sha256_file(path);source=control_folder/'predictions'/part/(path.stem+'.npz');hashes[source.as_posix()]=sha256_file(source)
            with np.load(source) as a:
                predictions['nn_primary'].append(a['queue_seed_mean'])
                for seed in cfg['seeds']:predictions[f'control_seed{seed}'].append(a[f'seed{seed}_queue'])
        return days,predictions
    cal,cal_predictions=load('selection');training={};choices={};plans={};models={}
    for seed in cfg['seeds']:
        plan_record=control['plans'][str(seed)];plan_path=Path(plan_record['path'])
        if sha256_file(plan_path)!=plan_record['sha256']:raise ValueError('Stored plan changed')
        with np.load(plan_path) as a:plan=a['plan']
        plans[str(seed)]=plan_record;torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        model=make_network('tcn_gru',48).to(device);old_role=f'seed{seed}_queue';role=f'jump_seed{seed}'
        initial=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items())).hexdigest()
        if initial!=control['training'][old_role]['initial_sha256']:raise ValueError('Initial model differs from paired control')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);history=[];best=-np.inf;tic=time.perf_counter();patch=hashlib.sha256()
        exposure={d['date']:np.zeros(d['mask'].shape,np.int32) for d in fits};pairs={k:np.zeros_like(v) for k,v in exposure.items()}
        for epoch in range(24):
            model.train();base_losses=[];pair_losses=[]
            for entries in plan[epoch*120:(epoch+1)*120]:
                batch=[]
                for group,index,center,view in entries:
                    d=groups[group][index];at,first,last=training_window(len(d['x']),int(center),128,64);use=view>=0
                    x=d['x_queue'][view] if use else d['x'];y=d['raw_y'] if use else d['y'];mask=d['mask_queue'][view] if use else d['mask']
                    batch.append((x,y,mask,at,first-at));exposure[d['date']][first:last]+=mask[first:last]
                    jump=mask[first+1:last]&mask[first:last-1]&(np.abs(np.diff(y[first:last,:,0],axis=0))>=np.float32(.8));pairs[d['date']][first+1:last]+=jump
                    patch.update((d['date']+':'+str(at)+':'+str(first)+':'+str(last)+';').encode())
                x=torch.from_numpy(np.stack([v[at:at+128].transpose(2,0,1) for v,y,m,at,first in batch])).to(device)
                y=torch.from_numpy(np.stack([v[at:at+128].transpose(2,0,1) for x,v,m,at,first in batch])).to(device)
                mask=torch.from_numpy(np.stack([v[at:at+128] for x,y,v,at,first in batch])).to(device)
                for i,(x0,y0,m0,at,first) in enumerate(batch):mask[i,:first]=False;mask[i,first+64:]=False
                optimizer.zero_grad(set_to_none=True);prediction=model(x,*graph);base=patch_score_loss(prediction,y,mask);auxiliary=speed_jump_pair_loss(prediction,y,mask);loss=base+.05*auxiliary
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();base_losses.append(float(base.detach().cpu()));pair_losses.append(float(auxiliary.detach().cpu()))
            if (epoch+1)%4==0:
                model.eval();score=summarize(cal,neural_predictions(model,cal,graph,device,128,64),evaluator.factor)['S_state'];history.append({'epoch':epoch+1,'base_loss':float(np.mean(base_losses)),'pair_loss':float(np.mean(pair_losses)),'ordinary_cal_State':score})
                if score>best:best=score;torch.save(model.state_dict(),args.output/(role+'.pt'))
                print('speed_jump',seed,'epoch',epoch+1,'cal',score,flush=True)
        if patch.hexdigest()!=control['training'][old_role]['patch_sequence_sha256']:raise ValueError('Patch sequence differs')
        source_exposure=control_folder/(old_role+'_exposure.npz')
        with np.load(source_exposure) as a:
            if any(not np.array_equal(v,a[k]) for k,v in exposure.items()):raise ValueError('Cell supervision differs from control')
        exposure_path=args.output/(role+'_exposure.npz');pair_path=args.output/(role+'_pair_exposure.npz');np.savez_compressed(exposure_path,**exposure);np.savez_compressed(pair_path,**pairs)
        model.load_state_dict(torch.load(args.output/(role+'.pt'),map_location=device,weights_only=True));model.eval();neural=neural_predictions(model,cal,graph,device,128,64)
        grid=[{'alpha':a,'metrics':summarize(cal,mix(cal,neural,a),evaluator.factor)} for a in [0.,.25,.5,.75,1.]];chosen=max(grid,key=lambda r:(r['metrics']['S_state'],-r['alpha']))
        choices[role]={'alpha':chosen['alpha'],'blend_grid':grid};cal_predictions[role]=mix(cal,neural,chosen['alpha'])
        exported=WindowedStateModel(baseline,model.state_dict(),evaluator.topology,chosen['alpha'],'tcn_gru');path=args.output/('State_'+role+'.pkl');exported.save(path);models[role]=exported
        training[role]={'seconds':time.perf_counter()-tic,'initial_sha256':initial,'patch_sequence_sha256':patch.hexdigest(),'history':history,'paired_control_role':old_role,
            'actual_cell_exposure_exactly_matches_control':True,'exposure_sha256':sha256_file(exposure_path),'pair_exposure_sha256':sha256_file(pair_path),
            'total_label_exposures':sum(int(v.sum()) for v in exposure.values()),'total_jump_pair_exposures':sum(int(v.sum()) for v in pairs.values()),'model_sha256':sha256_file(path)}
        del model; (args.output/'progress.json').write_text(json.dumps({'training':training,'choices':choices},indent=2,allow_nan=False),encoding='utf-8')
    roles=[f'jump_seed{s}' for s in cfg['seeds']];models['jump_seed_mean']=StateEnsemble([models[r] for r in roles],[.5,.5]);cal_predictions['jump_seed_mean']=[(a+b)/2 for a,b in zip(cal_predictions[roles[0]],cal_predictions[roles[1]])]
    cal_metrics={r:summarize(cal,p,evaluator.factor) for r,p in cal_predictions.items()};gain=cal_metrics['jump_seed_mean']['S_state']-cal_metrics['nn_primary']['S_state'];minimum=min(a['S_state']-b['S_state'] for a,b in zip(cal_metrics['jump_seed_mean']['details'],cal_metrics['nn_primary']['details']))
    primary='jump_seed_mean' if gain>=.0005 and minimum>=-.001 else 'nn_primary';selection={'primary':primary,'choices':choices,'cal_metrics':cal_metrics,'cal_gain':gain,'minimum_regime_cal_gain':minimum}
    seal=args.output/'selection_sealed.json';seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8');seal_sha=sha256_file(seal);print('speed_jump_PRIMARY_SEALED',primary,flush=True)
    for role,source_role in [('nn_primary','primary_export')]+[(f'control_seed{s}',f'seed{s}_queue') for s in cfg['seeds']]:
        record=replay['exports'][source_role]
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Frozen control export changed')
        models[role]=StateModel.load(Path(record['path']))
    models['primary_export']=models[primary];exports={}
    for role,model in models.items():
        path=args.output/('State_'+role+'.pkl');model.save(path);exports[role]={'path':path.as_posix(),'sha256':sha256_file(path)}
    ass,ass_predictions=load('assessment',seal_sha)
    for role in roles:
        network=make_network('tcn_gru',48).to(device);network.load_state_dict(torch.load(args.output/(role+'.pt'),map_location=device,weights_only=True));network.eval()
        ass_predictions[role]=mix(ass,neural_predictions(network,ass,graph,device,128,64),choices[role]['alpha']);del network
    ass_predictions['jump_seed_mean']=[(a+b)/2 for a,b in zip(ass_predictions[roles[0]],ass_predictions[roles[1]])];metrics={r:summarize(ass,p,evaluator.factor) for r,p in ass_predictions.items()};prediction_files=[]
    for part,days,values in [('selection',cal,cal_predictions),('assessment',ass,ass_predictions)]:
        target=args.output/'predictions'/part;target.mkdir(parents=True)
        for i,day in enumerate(days):
            path=target/(day['date']+'.npz');np.savez_compressed(path,**{r:p[i] for r,p in values.items()});prediction_files.append({'part':part,'date':day['date'],'path':path.as_posix(),'sha256':sha256_file(path)})
    comparisons={r:{'pooled_State_gain':m['S_state']-metrics['nn_primary']['S_state'],'paired_day_diagnostic':paired_block_interval([d['daily_State'] for d in metrics['nn_primary']['days']],[d['daily_State'] for d in m['days']])} for r,m in metrics.items()}
    result={'registered':registered,'source_snapshot':snapshot,'training':training,'plans':plans,'selection':selection,'selection_seal_sha256':seal_sha,'assessment':metrics,'comparisons':comparisons,
        'exports':exports,'prediction_files':prediction_files,'cache_sha256':hashes,'source_files_sha256':evaluator.hashes,'phase_read_order':read_order,'seconds':time.perf_counter()-start,
        'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,'source':evidence(gpu_used=device.type=='cuda'),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'limits':'Reused controls explicit,same cell exposure but extra gradient objective. Cal checkpoint/alpha choices can differ. Public research,not final/private. Formal Physics/Total unknown.'}
    text=json.dumps(result,indent=2,allow_nan=False);report.write_text(text,encoding='utf-8');(args.output/'experiment.json').write_text(text,encoding='utf-8');print('speed_jump_COMPLETE',primary,'NN',metrics['nn_primary']['S_state'],'jump',metrics['jump_seed_mean']['S_state'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
