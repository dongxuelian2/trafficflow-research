"""Matched relative-milepost attention with frozen NN and strong channel controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.state import StateModel
from trafficflow.state_ensemble import StateEnsemble
from trafficflow.channel_state import ChannelStateEnsemble
from trafficflow.data import VALUES
from trafficflow.adaptive_graph_state import patch_score_loss
from trafficflow.recurrent_state import training_window
from trafficflow.relative_position_state import make_relative_network,common_initial_sha256,relative_distance,RelativePositionStateModel
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol,paired_block_interval
from recurrent_state_experiment import neural_predictions,mix,summarize
from flow_transfer_common import PublicDays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--winter',action='store_true');args=parser.parse_args();report=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Never overwrite relative study')
    control_path=Path('reports/research')/('A104_I5_N_native_remask_ablation.json' if args.winter else 'A97_I210_W_native_remask_ablation.json');control=json.loads(control_path.read_text())
    replay_path=Path('reports/research')/('A107_I5_N_native_remask_training_independent_replay.json' if args.winter else 'A100_I210_W_native_remask_training_independent_replay.json');replay=json.loads(replay_path.read_text())
    strong_path=control_path if args.winter else Path('reports/research/A112_I210_W_channel_state_ensemble.json');strong=json.loads(strong_path.read_text())
    strong_proof_path=replay_path if args.winter else Path('reports/research/A115b_I210_W_channel_selection_independent_verification.json');strong_proof=json.loads(strong_proof_path.read_text())
    reference='nn_primary' if args.winter else 'strong_channel';hybrid_role='geo_speed_reference_flow' if args.winter else 'geo_speed_tree_flow';candidates=['geo_seed_mean',hybrid_role]
    if replay['registered']['protocol']['source_study_sha256']!=sha256_file(control_path) or strong_proof['registered']['protocol']['source_study_sha256']!=sha256_file(strong_path):raise ValueError('Frozen references not verified')
    cfg=control['registered']['protocol'];original_path=Path(cfg['source_study']);original=json.loads(original_path.read_text());old=original['registered']['protocol'];cache=Path(old['cache'])
    views_path=Path(cfg['views'])/'cache.json';views=json.loads(views_path.read_text());baseline=StateModel.load(Path(old['baseline_file']))
    if sha256_file(views_path)!=cfg['views_report_sha256'] or sha256_file(old['baseline_file'])!=old['baseline_sha256'] or control['selection']['primary']!='queue_seed_mean':raise ValueError('Matched source changed')
    first=views['rows'][0];link_order=np.load(first['original_cache_path'])['links'].tolist();metadata=pd.read_parquet(first['masked_file'],columns=['link_id','milepost']);metadata.link_id=metadata.link_id.astype(str)
    position=metadata.assign(milepost=pd.to_numeric(metadata.milepost,errors='coerce')).groupby('link_id').milepost.mean().reindex(link_order).to_numpy(float);distance=relative_distance(position)
    future=[f'2031-02-{day:02d}' for day in range(15,29)]
    if set(future)&{d for dates in original['prepared']['dates'].values() for d in dates}:raise ValueError('Future calendar overlaps fit/cal')
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Matched_static_relative_position_attention_GRU','panel':cfg['panel'],
        'source_study':original_path.as_posix(),'source_study_sha256':sha256_file(original_path),'views':cfg['views'],'view_report_sha256':sha256_file(views_path),
        'control_study':control_path.as_posix(),'control_study_sha256':sha256_file(control_path),'control_replay':replay_path.as_posix(),'control_replay_sha256':sha256_file(replay_path),
        'strong_reference_study':strong_path.as_posix(),'strong_reference_sha256':sha256_file(strong_path),'strong_verification_sha256':sha256_file(strong_proof_path),
        'reference_role':reference,'hybrid_role':hybrid_role,'geometry_source':first['masked_file'],'geometry_source_sha256':sha256_file(first['masked_file']),
        'geometry_columns':['link_id','milepost'],'link_order':link_order,'positions':position.tolist(),'graph_sha256':sha256_file(cache/'graph.npz'),
        'geometry_rule':'Absolute pair distance/median positive gap,cap20;two signed learned scalar biases initialized0. Finite global attention,not physical edges.',
        'shared_parameters':97307,'new_parameters':2,'seeds':cfg['seeds'],'epochs':24,'steps_per_epoch':120,'batch_size':4,'context':128,'core':64,
        'optimizer':'Same AdamW lr.001 wd1e-4 clip1','loss':'Unmodified ordinary patch State RMSE;no temporal auxiliary objective',
        'data':'Exact stored Queue-shaped control plans/inputs/teachers/core masks/cell exposure;no new labels',
        'initialization':'All shared state tensor bytes equal original control initialization;new coefficients zero. Additional constructor RNG consumed after shared initialization,training has no dropout/random sampling beyond frozen plan.',
        'controls':f'{control_path.stem} NN and perseed queue models; strongest reference {strong_path.stem},all reused,not refit',
        'checkpoint':'Ordinary pure cal State every4epochs,first maximum','alpha':'Original0/.25/.5/.75/1 ordinarycal grid,lower-alpha tie',
        'candidates':candidates,'hybrid':f'Fixed new mean speed + {reference} flow;no additional weight search',
        'primary':f'Keep {reference} unless candidate cal gain>=.0005 and minimum regime gain>=-.001;best eligible cal score,pure mean tie preference',
        'assessment':'Viewed old public assessment,read after both fits and all choices seal','future_dates':future,
        'future_scope':'Registered before fits;public research,prior studies may have viewed dates,not final/private','formal_State':None,'formal_Physics':None,'formal_Total':None})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    np.savez_compressed(args.output/'geometry.npz',link_order=np.array(link_order),positions=position,distance_matrix=distance)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    graph_np=dict(np.load(cache/'graph.npz'));graph=[torch.tensor(graph_np[k],device=device) for k in ['incoming','outgoing','fd']]
    fits=[];view_rows={r['date']:r for r in views['rows']};hashes={};read_order=[];source_hashes={first['masked_file']:sha256_file(first['masked_file'])}
    for path in sorted((cache/'fit').glob('*.npz')):
        if sha256_file(path)!=control['cache_sha256'][path.as_posix()]:raise ValueError('Original fit changed')
        day=dict(np.load(path));day['date']=path.stem;row=view_rows[path.stem]
        if day['links'].tolist()!=link_order or sha256_file(row['path'])!=row['sha256']:raise ValueError('View or node order changed')
        with np.load(row['path']) as a:day.update(raw_y=a['y'],x_queue=a['x_queue'],mask_queue=a['mask_queue'])
        fits.append(day);hashes[path.as_posix()]=sha256_file(path)
    groups=[[d for d in fits if str(d['regime'])==r] for r in ['R1','R2','R3']];evaluator=PublicDays(cfg['panel'],baseline.profile,baseline.feature_version)
    control_folder=Path('artifacts/research')/control_path.stem;strong_folder=Path('artifacts/research')/strong_path.stem
    def load(part,seal=None):
        if part=='assessment' and (seal is None or sha256_file(args.output/'selection_sealed.json')!=seal):raise ValueError('Assessment before both fits/seal')
        read_order.append({'part':part,'prior_selection_seal_sha256':seal});days=[];predictions={r:[] for r in ['nn_primary']+([] if args.winter else ['strong_channel'])+[f'control_seed{s}' for s in cfg['seeds']]}
        for path in sorted((cache/part).glob('*.npz')):
            if sha256_file(path)!=control['cache_sha256'][path.as_posix()]:raise ValueError('Evaluation cache changed')
            day=evaluator.load(path.stem,path)
            if day['features'].links!=link_order:raise ValueError('Evaluation node order differs')
            p=baseline.predict(day['features'],day['template'])[VALUES].to_numpy(float);p[:,0]=np.minimum(p[:,0],day['template'].link_id.map(evaluator.net.free_speed_kmh).to_numpy(float)*1.05)
            p[:,1]=np.minimum(p[:,1],day['template'].link_id.map(evaluator.net.capacity_vph).to_numpy(float)*1.05)/day['target_link_lanes'];day['comparator']=p;days.append(day);hashes[path.as_posix()]=sha256_file(path)
            old_file=control_folder/'predictions'/part/(path.stem+'.npz');strong_file=strong_folder/'predictions'/part/(path.stem+'.npz')
            hashes[old_file.as_posix()]=sha256_file(old_file)
            with np.load(old_file) as a:
                predictions['nn_primary'].append(a['queue_seed_mean'])
                for seed in cfg['seeds']:predictions[f'control_seed{seed}'].append(a[f'seed{seed}_queue'])
                if not args.winter:
                    hashes[strong_file.as_posix()]=sha256_file(strong_file)
                    with np.load(strong_file) as b:
                        if not np.array_equal(a['queue_seed_mean'],b['nn_primary']):raise ValueError('Reference prediction domain differs')
                        predictions['strong_channel'].append(b['primary_export'])
        return days,predictions
    cal,cal_predictions=load('selection');training={};choices={};plans={};models={}
    for seed in cfg['seeds']:
        plan_record=control['plans'][str(seed)];plan_path=Path(plan_record['path'])
        if sha256_file(plan_path)!=plan_record['sha256']:raise ValueError('Frozen plan changed')
        with np.load(plan_path) as a:plan=a['plan']
        plans[str(seed)]=plan_record;torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        model=make_relative_network(position,48).to(device);paired=f'seed{seed}_queue';role=f'geo_seed{seed}';initial=common_initial_sha256(model)
        if initial!=control['training'][paired]['initial_sha256'] or sum(p.numel() for p in model.parameters())!=97309:raise ValueError('Shared initialization/parameter budget differs')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);history=[];best=-np.inf;tic=time.perf_counter();patch=__import__('hashlib').sha256();exposure={d['date']:np.zeros(d['mask'].shape,np.int32) for d in fits}
        for epoch in range(24):
            model.train();losses=[]
            for entries in plan[epoch*120:(epoch+1)*120]:
                batch=[]
                for group,index,center,view in entries:
                    d=groups[group][index];at,first,last=training_window(len(d['x']),int(center),128,64);use=view>=0
                    x=d['x_queue'][view] if use else d['x'];y=d['raw_y'] if use else d['y'];mask=d['mask_queue'][view] if use else d['mask']
                    batch.append((x,y,mask,at,first-at));exposure[d['date']][first:last]+=mask[first:last];patch.update((d['date']+':'+str(at)+':'+str(first)+':'+str(last)+';').encode())
                x=torch.from_numpy(np.stack([v[at:at+128].transpose(2,0,1) for v,y,m,at,first in batch])).to(device)
                y=torch.from_numpy(np.stack([v[at:at+128].transpose(2,0,1) for x,v,m,at,first in batch])).to(device)
                mask=torch.from_numpy(np.stack([v[at:at+128] for x,y,v,at,first in batch])).to(device)
                for j,(xx,yy,mm,at,first) in enumerate(batch):mask[j,:first]=False;mask[j,first+64:]=False
                optimizer.zero_grad(set_to_none=True);loss=patch_score_loss(model(x,*graph),y,mask)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
            if (epoch+1)%4==0:
                model.eval();score=summarize(cal,neural_predictions(model,cal,graph,device,128,64),evaluator.factor)['S_state'];strength=[float(m.distance_strength.detach().cpu()) for m in model.spatial]
                history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'ordinary_cal_State':score,'distance_strength':strength})
                if score>best:best=score;torch.save(model.state_dict(),args.output/(role+'.pt'))
                print('relative_position',seed,'epoch',epoch+1,'cal',score,'strength',strength,flush=True)
        if patch.hexdigest()!=control['training'][paired]['patch_sequence_sha256']:raise ValueError('Patch centers changed')
        with np.load(control_folder/(paired+'_exposure.npz')) as a:
            if any(not np.array_equal(v,a[k]) for k,v in exposure.items()):raise ValueError('Actual cell exposure differs')
        exposure_path=args.output/(role+'_exposure.npz');np.savez_compressed(exposure_path,**exposure)
        model.load_state_dict(torch.load(args.output/(role+'.pt'),map_location=device,weights_only=True));model.eval();neural=neural_predictions(model,cal,graph,device,128,64)
        grid=[{'alpha':a,'metrics':summarize(cal,mix(cal,neural,a),evaluator.factor)} for a in [0.,.25,.5,.75,1.]];chosen=max(grid,key=lambda r:(r['metrics']['S_state'],-r['alpha']))
        choices[role]={'alpha':chosen['alpha'],'blend_grid':grid};cal_predictions[role]=mix(cal,neural,chosen['alpha']);exported=RelativePositionStateModel(baseline,model.state_dict(),evaluator.topology,chosen['alpha'],link_order,position)
        path=args.output/('State_'+role+'.pkl');exported.save(path);models[role]=exported
        training[role]={'seconds':time.perf_counter()-tic,'shared_initial_sha256':initial,'parameters':97309,'patch_sequence_sha256':patch.hexdigest(),'history':history,'paired_control_role':paired,
            'actual_cell_exposure_exactly_matches_control':True,'exposure_sha256':sha256_file(exposure_path),'total_label_exposures':sum(int(v.sum()) for v in exposure.values()),'model_sha256':sha256_file(path),
            'selected_distance_strength':[float(m.distance_strength.detach().cpu()) for m in model.spatial]}
        del model;(args.output/'progress.json').write_text(json.dumps({'training':training,'choices':choices},indent=2,allow_nan=False),encoding='utf-8')
    roles=[f'geo_seed{s}' for s in cfg['seeds']];models['geo_seed_mean']=StateEnsemble([models[r] for r in roles],[.5,.5]);cal_predictions['geo_seed_mean']=[(a+b)/2 for a,b in zip(cal_predictions[roles[0]],cal_predictions[roles[1]])]
    def hybrid(values):
        values[hybrid_role]=[]
        for a,b in zip(values['geo_seed_mean'],values[reference]):p=a.copy();p[:,1]=b[:,1];values[hybrid_role].append(p)
    hybrid(cal_predictions);cal_metrics={r:summarize(cal,p,evaluator.factor) for r,p in cal_predictions.items()};eligible=[]
    for role in candidates:
        gain=cal_metrics[role]['S_state']-cal_metrics[reference]['S_state'];minimum=min(a['S_state']-b['S_state'] for a,b in zip(cal_metrics[role]['details'],cal_metrics[reference]['details']))
        if gain>=.0005 and minimum>=-.001:eligible.append(role)
    primary=max(eligible,key=lambda r:cal_metrics[r]['S_state']) if eligible else reference;selection={'primary':primary,'choices':choices,'cal_metrics':cal_metrics,'eligible_candidates':eligible,'reference_role':reference}
    seal=args.output/'selection_sealed.json';seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8');seal_sha=sha256_file(seal);print('relative_PRIMARY_SEALED',primary,flush=True)
    for role,source_role in [('nn_primary','primary_export')]+[(f'control_seed{s}',f'seed{s}_queue') for s in cfg['seeds']]:
        row=replay['exports'][source_role]
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Frozen reference changed')
        models[role]=StateModel.load(Path(row['path']))
    if not args.winter:
        if sha256_file(strong['primary_export']['path'])!=strong['primary_export']['sha256']:raise ValueError('Strong model changed')
        models['strong_channel']=StateModel.load(Path(strong['primary_export']['path']))
    models[hybrid_role]=ChannelStateEnsemble([models['geo_seed_mean'],models[reference]],[[1,0],[0,1]]);models['primary_export']=models[primary];exports={}
    for role,model in models.items():
        path=args.output/('State_'+role+'.pkl')
        if not path.exists():model.save(path)
        exports[role]={'path':path.as_posix(),'sha256':sha256_file(path)}
    ass,ass_predictions=load('assessment',seal_sha)
    for role in roles:
        network=make_relative_network(position,48).to(device);network.load_state_dict(torch.load(args.output/(role+'.pt'),map_location=device,weights_only=True));network.eval()
        ass_predictions[role]=mix(ass,neural_predictions(network,ass,graph,device,128,64),choices[role]['alpha']);del network
    ass_predictions['geo_seed_mean']=[(a+b)/2 for a,b in zip(ass_predictions[roles[0]],ass_predictions[roles[1]])];hybrid(ass_predictions);metrics={r:summarize(ass,p,evaluator.factor) for r,p in ass_predictions.items()};prediction_files=[]
    for part,days,values in [('selection',cal,cal_predictions),('assessment',ass,ass_predictions)]:
        target=args.output/'predictions'/part;target.mkdir(parents=True)
        for j,day in enumerate(days):
            path=target/(day['date']+'.npz');np.savez_compressed(path,**{r:p[j] for r,p in values.items()});prediction_files.append({'part':part,'date':day['date'],'path':path.as_posix(),'sha256':sha256_file(path)})
    comparisons={r:{'pooled_State_gain':m['S_state']-metrics[reference]['S_state'],'paired_day_diagnostic':paired_block_interval([d['daily_State'] for d in metrics[reference]['days']],[d['daily_State'] for d in m['days']])} for r,m in metrics.items()}
    result={'registered':registered,'source_snapshot':snapshot,'training':training,'plans':plans,'selection':selection,'selection_seal_sha256':seal_sha,'assessment':metrics,'comparisons':comparisons,
        'exports':exports,'prediction_files':prediction_files,'cache_sha256':hashes,'source_files_sha256':{**source_hashes,**evaluator.hashes},'phase_read_order':read_order,
        'seconds':time.perf_counter()-start,'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,'source':evidence(gpu_used=device.type=='cuda'),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'limits':'Two learned statistical relation parameters,not road edges. Controls reused,shared weights and cell exposure matched. Cal checkpoint/alpha may differ. Strong-channel baseline used for promotion. No formal Physics/Total.'}
    text=json.dumps(result,indent=2,allow_nan=False);report.write_text(text,encoding='utf-8');(args.output/'experiment.json').write_text(text,encoding='utf-8');print('relative_COMPLETE',primary,'reference',metrics[reference]['S_state'],'geo',metrics['geo_seed_mean']['S_state'],'hybrid',metrics[hybrid_role]['S_state'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
