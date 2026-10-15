"""Replay joint models, legal masked-State contexts and selection evidence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
from lightgbm import Booster
from scipy.special import expit
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.causal_queue import released_origin_rows
from trafficflow.joint_graph_queue import JointGraphQueue
from trafficflow.joint_queue_data import reconstruction_features
from trafficflow.queue import QueueModel
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol
from verify_joint_dynamics_data import reference_day,reference_state,reference_joint
from verify_dense_pretrain_models import digest,checkpoint_score,calibration,event_sequence
from verify_onset_rank_queue import manual_baseline,manual_metric,same_metric,paired,same_paired


ARMS=['direct_no_flow','direct_flow','latent_flow'];SEEDS=[20261013,20261029]


def reference_features(x,geometry):
    """Independent double-precision spatial means; canonical hashes separately."""
    n=x.shape[2];latest=[];ages=[]
    for field,mask,default in [(0,2,geometry['geometry'][0]/25.),(1,3,geometry['geometry'][1]/1200.)]:
        valid=x[:,:,:,mask]>.5;at=12-np.argmax(valid[:,::-1],axis=1);at=np.where(valid.any(axis=1),at,-1);v=np.take_along_axis(x[:,:,:,field],np.maximum(at,0)[:,None],axis=1)[:,0];latest.append(np.where(at>=0,v,default));ages.append(((12-at)/13.).astype(np.float32))
    state=np.stack(latest,axis=1).astype(np.float32);age=np.stack(ages,axis=1);rows=[x.transpose(0,2,1,3).reshape(-1,117),state.transpose(0,2,1).reshape(-1,2),age.transpose(0,2,1).reshape(-1,2)];static=geometry['geometry'].T/np.array([25.,600.,25.,25.],np.float32);rows.append(np.tile(static,(len(x),1)))
    for kernel in [geometry['incoming'],geometry['outgoing']]:rows.append((state.astype(float)@kernel.astype(float).T).transpose(0,2,1).reshape(-1,2))
    return np.concatenate(rows,axis=1),state


def ref_sufficient(a,p):
    result={}
    for regime in ['R1','R2','R3']:
        use=(a['current_regime']==regime)&a['current_domain'][:,0]&a['current_domain'][:,1];v=(p[:,0].astype(float)-a['truth_current'][:,0].astype(float))[use];q=(p[:,1].astype(float)-a['truth_current'][:,1].astype(float))[use]
        result[regime]={'n':int(np.count_nonzero(use)),'speed_normalized_SSE':float(np.dot(v,v)),'flow_normalized_SSE':float(np.dot(q,q))}
    return result


def same_stats(actual,wanted):
    for regime in ['R1','R2','R3']:
        if actual[regime]['n']!=wanted[regime]['n'] or any(not np.isclose(actual[regime][k],wanted[regime][k],atol=1e-9,rtol=1e-12) for k in ['speed_normalized_SSE','flow_normalized_SSE']):raise ValueError('Independent State sufficient statistics differ')


def ref_combined(rows):
    result=[]
    for regime in ['R1','R2','R3']:
        n=sum(r[regime]['n'] for r in rows);v=sum(r[regime]['speed_normalized_SSE'] for r in rows);q=sum(r[regime]['flow_normalized_SSE'] for r in rows);rv=float(np.sqrt(v/n)) if n else None;rq=float(np.sqrt(q/n)) if n else None
        result.append({'regime':regime,'n':n,'rmse_speed_kmh':rv*25. if n else None,'rmse_flow_per_lane':rq*600. if n else None,'public_subset_state_formula':.54*max(0.,1-rv)+.46*max(0.,1-rq) if n else None})
    return result,float(np.mean([r['public_subset_state_formula'] for r in result])) if all(r['n'] for r in result) else None


def forward(state,rollout,a,graph,device):
    model=JointGraphQueue(rollout=rollout).to(device);model.load_state_dict(state);model.eval();values={k:[] for k in ['probability','speed','flow','analysis']}
    with torch.no_grad():
        for x,sx in zip(a['x'],a['state_x']):
            output=list(model(torch.tensor(x.transpose(2,0,1)[None].copy(),device=device),*graph));output[3]=model.analyze(torch.tensor(sx.transpose(2,0,1)[None].copy(),device=device),*graph);out=[v.cpu().numpy()[0] for v in output];out[0]=expit(out[0].astype(float)-np.log(4.))
            for k,v in zip(values,out):values[k].append(v)
    return {k:np.array(v) for k,v in values.items()}


def independent_exposure(packs,origins,event,batches,dense):
    counts={'windows_drawn':int(batches.size),'unique_contexts':len(np.unique(batches)),'classification_cells':0,'positive_class_cells':0,'forecast_speed_flow_cells':0,'current_masked_speed_cells':0,'current_masked_flow_cells':0}
    for i,multiplicity in zip(*np.unique(batches,return_counts=True)):
        if dense:
            di,at=map(int,origins[i]);p=packs[di];a=reference_joint(p,at,bool((p['truth_queue'][at-12:at].sum(axis=0)>=2).any()))
        else:a={k:event[k][i] for k in ['available','labels','ongoing','current_domain']}
        domain=a['available'].copy()
        if not dense and not bool(a['ongoing']):domain[:5]=False
        counts['classification_cells']+=int(domain.sum())*int(multiplicity);counts['positive_class_cells']+=int((domain*a['labels']).sum())*int(multiplicity);counts['forecast_speed_flow_cells']+=int(a['available'].sum())*int(multiplicity)
        for field,name in enumerate(['speed','flow']):counts['current_masked_'+name+'_cells']+=int(a['current_domain'][field].sum())*int(multiplicity)
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite joint model verification')
    root=Path('reports/research');study_path=root/'C91_joint_flow_latent_dynamics.json';study=json.loads(study_path.read_text());proof_path=root/'C92_joint_dynamics_data_verification.json';proof=json.loads(proof_path.read_text())
    study_folder=Path('artifacts/research')/study_path.stem
    if proof['registered']['protocol']['dataset_seal_sha256']!=sha256_file(study_folder/'all_datasets_sealed.json'):raise ValueError('Input proof binding differs')
    hashes=dict(proof['source_files_sha256'])
    for case in study['cases']:hashes.update(case['source_files_sha256'])
    for path,d in hashes.items():
        if sha256_file(path)!=d:raise ValueError('Audited source changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Joint_dynamics_checkpoint_tree_and_fullState_replay','source_study_sha256':sha256_file(study_path),'source_data_proof_sha256':sha256_file(proof_path),
        'scope':'Reuse C92 fullfit/CP data proof;all new evaluation contexts and all full30dayState data independently reconstructed. Init/batches/exposure/48NNsnapshots/4treefitdatahashes/inference/gates and portable interfaces replayed. Tree feature semantics independently computed with double spatial accumulation,canonical hashes reuse feature function explicitly. No independent re-optimization of12NN/4trees or formal score.'})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');tic=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');release=Release('data/kaggle_public');cases=[]
    totals={k:0 for k in ['network_fits','checkpoint_window_forwards','evaluation_event_windows','event_member_forwards','tree_fit_state_contexts','tree_fit_rows','full_state_contexts','full_state_member_forwards','full_state_days','decision_grid_scores','portable_history_windows','portable_Queue_calls','portable_State_calls','target_shuffle_calls','future_rejections']};maximum={k:0. for k in ['probability','speed','flow','analysis','independent_tree_features']}
    for case in study['cases']:
        base=study_folder/case['name'];geometry=dict(np.load(case['geometry']['path']));graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']];fit=dict(np.load(next(r for r in case['event_records'] if r['phase']=='fit')['path']));cp=dict(np.load(next(r for r in case['event_records'] if r['phase']=='checkpoint')['path']));packs=[dict(np.load(r['path'])) for r in case['dataset_description']['day_records']];origins=np.load(case['dataset_description']['origin_record']['path'])['indices'];initials={};sequences={}
        for r in case['initial_records']:
            torch.manual_seed(r['seed']+101);model=JointGraphQueue();model.load_state_dict(torch.load(r['source_old_initial']['path'],map_location='cpu',weights_only=True),strict=False);expected=model.state_dict();saved=torch.load(r['path'],map_location='cpu',weights_only=True)
            if sha256_file(r['path'])!=r['sha256'] or digest(saved)!=r['state_digest'] or any(not torch.equal(expected[k],saved[k]) for k in saved):raise ValueError('Joint initialization differs')
            initials[r['seed']]=saved;del model
        for r in case['batch_records']:
            if sha256_file(r['path'])!=r['sha256'] or sha256_file(r['source_old_batches']['path'])!=r['source_old_batches']['sha256']:raise ValueError('Common batch archive changed')
            a=dict(np.load(r['path']));wanted={'warm':np.random.default_rng(r['seed']+500003).integers(0,len(origins),(400,8),dtype=np.int64),'event':event_sequence(fit,r['seed']+100003,800,4)}
            if any(not np.array_equal(a[k],v) for k,v in wanted.items()):raise ValueError('Common RNG batches differ')
            er=next(v for v in case['actual_exposures'] if v['seed']==r['seed'])
            if er['warm_available']!=independent_exposure(packs,origins,fit,a['warm'],True) or er['event_available']!=independent_exposure(packs,origins,fit,a['event'],False):raise ValueError('Actual supervision exposure differs')
            sequences[r['seed']]=a
        selected={}
        for t in case['training']:
            if t['initial_state_digest']!=digest(initials[t['seed']]) or t['flow_loss_weight']!=(0. if t['arm']=='direct_no_flow' else .1) or t['rollout']!=(t['arm']=='latent_flow'):raise ValueError('Matched arm differs')
            if sha256_file(t['before_event']['path'])!=t['before_event']['sha256']:raise ValueError('Warm-up weights changed')
            for r in t['snapshots']:
                for value in [r,r['optimizer'],r['checkpoint_predictions']]:
                    if sha256_file(value['path'])!=value['sha256']:raise ValueError('Checkpoint file changed')
                state=torch.load(r['path'],map_location='cpu',weights_only=True);opt=torch.load(r['optimizer']['path'],map_location='cpu',weights_only=True)
                if digest(state)!=r['state_digest'] or opt['step']!=r['step'] or any(int(v['step'])!=r['step'] for v in opt['optimizer']['state'].values()):raise ValueError('Optimizer reset/snapshot step differs')
                p=forward(state,t['rollout'],cp,graph,device);totals['checkpoint_window_forwards']+=len(cp['x'])
                with np.load(r['checkpoint_predictions']['path']) as saved:
                    for k,v in p.items():
                        error=float(np.abs(v-saved[k]).max());maximum[k]=max(maximum[k],error)
                        if error>(1e-6 if k=='probability' else 1e-5):raise ValueError('Checkpoint replay differs: '+k)
                if abs(checkpoint_score(p['probability'],cp)-r['checkpoint_Brier'])>1e-10:raise ValueError('Independent checkpoint Brier differs')
            if min(t['snapshots'],key=lambda r:r['checkpoint_Brier'])!=t['selected']:raise ValueError('Checkpoint selection differs')
            selected[(t['arm'],t['seed'])]=torch.load(t['selected']['path'],map_location='cpu',weights_only=True);totals['network_fits']+=1
        # Full strong-tree fitting population: hash canonical arrays, separately
        # check their feature semantics against an independent implementation.
        accum={k:hashlib.sha256() for k in ['features','labels','domain','coordinates']};fit_rows=0;fit_contexts=0
        for di,p in enumerate(packs):
            for start in range(12,288,48):
                ats=list(range(start,min(288,start+48)));rows=[reference_state(p,at) for at in ats];a={k:np.stack([w[k] for w in rows]) for k in ['state_x','truth_current','current_domain']};canonical=reconstruction_features(a['state_x'],geometry);ref,_=reference_features(a['state_x'],geometry);error=float(np.abs(canonical-ref).max());maximum['independent_tree_features']=max(maximum['independent_tree_features'],error)
                if error>2e-6:raise ValueError('Independent tree feature semantics differ')
                use=a['current_domain'].any(axis=1).ravel();values={'features':canonical[use],'labels':a['truth_current'].transpose(0,2,1).reshape(-1,2)[use],'domain':a['current_domain'].transpose(0,2,1).reshape(-1,2)[use],
                    'coordinates':np.array([(di,at,j) for at in ats for j in range(len(fit['links']))],np.int32)[use]}
                for k,v in values.items():accum[k].update(memoryview(np.ascontiguousarray(v)).cast('B'))
                fit_rows+=int(use.sum());fit_contexts+=len(ats)
        if {k:v.hexdigest() for k,v in accum.items()}!=case['state_trees']['training_arrays_sha256'] or fit_rows!=case['state_trees']['n_feature_rows'] or fit_contexts!=case['state_trees']['fit_state_origins']:raise ValueError('Full State-tree fitting population/hash differs')
        totals['tree_fit_rows']+=fit_rows;totals['tree_fit_state_contexts']+=fit_contexts;trees=[]
        for r in case['state_trees']['records']:
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('State tree changed')
            tree=Booster(model_file=r['path']);trees.append(tree)
            if tree.current_iteration()!=400:raise ValueError('State baseline iterations differ')
        portable={}
        for role,r in case['exports'].items():
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Portable joint model changed')
            model=QueueModel.load(Path(r['path']));portable[role]=model;candidate=r['candidate']
            if candidate!='incumbent':
                if model.alpha!=.25 or model.rollout!=(candidate=='latent_flow') or model.setting!=case['choices'][candidate]['setting']:raise ValueError('Portable joint policy differs')
                for state,seed in zip(model.states,SEEDS):
                    if any(not torch.equal(state[k],selected[(candidate,seed)][k]) for k in state):raise ValueError('Portable states differ')
        panel=case['panel'];net,topology=release.network(panel),release.topology(panel);links=fit['links'].tolist();raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};masked_paths={partition_date(release,p):p for p in release.files(panel,'train')}
        @lru_cache(maxsize=8)
        def day(date):return reference_day(raw_paths[date],masked_paths[date],links,net)
        @lru_cache(maxsize=4)
        def frames(date):return release.read(raw_paths[date]),release.read(masked_paths[date])
        runs={'selection':[],'assessment':[]};poisoned=False
        for r in case['event_records']:
            phase=r['phase']
            if phase in ['fit','checkpoint']:continue
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('New event cache changed')
            a=dict(np.load(r['path']));pr=next(v for v in case['prediction_records'] if v['phase']==phase and v['seed']==r['seed'])
            if sha256_file(pr['path'])!=pr['sha256']:raise ValueError('Event predictions changed')
            saved=dict(np.load(pr['path']));means={}
            for i,text in enumerate(a['origins']):
                origin=pd.Timestamp(str(text));at=int((origin-pd.Timestamp(origin.date().isoformat(),tz='UTC'))/pd.Timedelta(minutes=5));expected=reference_joint(day(origin.date().isoformat()),at,bool(a['ongoing'][i]))
                if any(not np.array_equal(v,a[k][i]) for k,v in expected.items()):raise ValueError('Independent eval teacher/context differs')
                totals['evaluation_event_windows']+=1
            for arm in ARMS:
                members=[]
                for seed in SEEDS:
                    p=forward(selected[(arm,seed)],arm=='latent_flow',a,graph,device);members.append(p);totals['event_member_forwards']+=len(a['x'])
                    for k,v in p.items():
                        error=float(np.abs(v-saved[arm+'_'+str(seed)+'_'+k]).max());maximum[k]=max(maximum[k],error)
                        if error>(1e-6 if k=='probability' else 1e-5):raise ValueError('Independent event forward differs')
                means[arm]={k:np.mean([m[k] for m in members],axis=0) for k in members[0]}
                for k,v in means[arm].items():
                    if not np.allclose(v,saved[arm+'_'+k],atol=1e-6,rtol=0):raise ValueError('Joint ensemble mean differs')
            # Recompute reference probabilities from the previously verified
            # C71 feature cache; its history proof is reused, not overstated.
            original=json.loads((root/'C71_rolling_onset_threshold_refits.json').read_text());old=next(c for c in original['cases'] if c['name']==case['name']);rec=next(v for v in old['cache_records'] if v['phase']==phase and v['seed']==r['seed']);feat=next(f for f in rec['feature_files'] if Path(f['path']).stem.endswith('_'+old['feature_arm']))
            if sha256_file(feat['path'])!=feat['sha256']:raise ValueError('Frozen Queue tree features changed')
            tp=portable['incumbent'].baseline.probability(pd.read_parquet(feat['path']))
            if not np.array_equal(tp,saved['tree_probability']):raise ValueError('Strong reference probabilities differ')
            if phase=='decision':
                for arm in ARMS:
                    setting,n=calibration(a,.75*tp+.25*means[arm]['probability'].ravel());totals['decision_grid_scores']+=n
                    wanted=case['choices'][arm]['setting']
                    if any(setting[k]!=wanted[k] for k in ['mode','thresholds','require_nonempty','timing']) or abs(setting['calibration_proxy_IoU']-wanted['calibration_proxy_IoU'])>1e-12:raise ValueError('Full joint decision grid differs')
            preds={role:manual_baseline(a,tp if role=='incumbent' else .75*tp+.25*means[role]['probability'].ravel(),case['choices'][role]['setting']) for role in ['incumbent']+ARMS}
            if phase=='assessment':preds['selected']=preds[case['selected']]
            metrics={k:manual_metric(a,v) for k,v in preds.items()}
            if any(not np.array_equal(saved[k],v) for k,v in preds.items()):raise ValueError('Full-domain joint predictions differ')
            wanted=case['choices'] if phase=='decision' else next(v for v in case[phase] if v['seed']==r['seed'])['models']
            for k,v in metrics.items():same_metric(v,wanted[k]['metrics'] if phase=='decision' else wanted[k])
            if phase=='decision':
                if max(['incumbent']+ARMS,key=lambda k:metrics[k]['proxy_IoU'])!=case['decision_selected']:raise ValueError('Joint challenger differs')
            else:runs[phase].append({'seed':r['seed'],'models':metrics})
            width=6*len(links);sample=set();first=set()
            for ongoing in [False,True]:
                ids=np.flatnonzero(a['ongoing']==ongoing);sample.update([int(ids[0]),int(ids[-1])]);first.add(int(ids[0]))
            for i in sorted(sample):
                origin=pd.Timestamp(str(a['origins'][i]));raw,masked=frames(origin.date().isoformat());ts=pd.to_datetime(raw.timestamp,utc=True);past=raw.loc[(ts>=origin-pd.Timedelta(minutes=60))&(ts<origin)];history=pd.concat([past,released_origin_rows(release,panel,'train',origin)],ignore_index=True);mt=pd.to_datetime(masked.timestamp,utc=True);state_history=masked.loc[(mt>=origin-pd.Timedelta(minutes=60))&(mt<=origin)].copy();future=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]];target=pd.MultiIndex.from_product([future,links],names=['timestamp','link_id']).to_frame(index=False);condition='queue_ongoing' if a['ongoing'][i] else 'queue_onset';offset=slice(i*width,(i+1)*width);totals['portable_history_windows']+=1
                for role,model in portable.items():
                    out=model.predict_window(history,target,net,topology,origin,condition);totals['portable_Queue_calls']+=1;candidate=case['exports'][role]['candidate']
                    if not np.array_equal(out.queue_pred,preds[candidate][offset]):raise ValueError('Portable Queue differs')
                    if i in first:
                        order=np.random.default_rng(107).permutation(width);out=model.predict_window(history,target.iloc[order],net,topology,origin,condition);totals['target_shuffle_calls']+=1
                        if not np.array_equal(out.queue_pred,preds[candidate][offset][order]):raise ValueError('Portable Queue target permutation differs')
                    if role in ARMS:
                        analysis=model.reconstruct_current(state_history,net,origin);totals['portable_State_calls']+=1
                        if not np.allclose(analysis,means[role]['analysis'][i],atol=1e-5,rtol=0):raise ValueError('Portable State masked-only contract differs')
                if not poisoned and condition=='queue_onset':
                    for role,model in portable.items():
                        bad=pd.concat([history,history.iloc[[0]].assign(timestamp=future[0])],ignore_index=True)
                        try:model.predict_window(bad,target,net,topology,origin,condition)
                        except ValueError:totals['future_rejections']+=1
                        else:raise ValueError('Joint model accepted future Queue history')
                        if role in ARMS:
                            bad=pd.concat([state_history,state_history.iloc[[0]].assign(timestamp=future[0])],ignore_index=True)
                            try:model.reconstruct_current(bad,net,origin)
                            except ValueError:totals['future_rejections']+=1
                            else:raise ValueError('Joint model accepted future State history')
                    poisoned=True
        gate=paired(runs['selection'],case['dates']['selection'],'incumbent',case['decision_selected']);same_paired(gate,case['selection_gate']);winner=case['decision_selected'] if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent'
        if winner!=case['selected']:raise ValueError('Joint selection gate differs')
        for role,value in case['comparisons'].items():same_paired(paired(runs['assessment'],case['dates']['assessment'],'incumbent',role),value)
        models={}
        for (arm,seed),state in selected.items():
            model=JointGraphQueue(rollout=arm=='latent_flow').to(device);model.load_state_dict(state);model.eval();models[arm+'_'+str(seed)]=model
        daily=[]
        if [r['date'] for r in case['state_assessment_records']]!=case['dates']['assessment'] or any(r['n_contexts']!=276 for r in case['state_assessment_records']):raise ValueError('Incomplete daily State evaluation calendar')
        for r in case['state_assessment_records']:
            if sha256_file(r['path'])!=r['sha256'] or r['prior_selection_seal_sha256']!=case['seals']['selection']:raise ValueError('Full State predictions/seal changed')
            saved=dict(np.load(r['path']));p=day(r['date']);predictions={};fields={}
            for start in range(12,288,48):
                ats=list(range(start,min(288,start+48)));rows=[reference_state(p,at) for at in ats];a={k:np.stack([w[k] for w in rows]) for k in rows[0]};features=reconstruction_features(a['state_x'],geometry);independent,base_state=reference_features(a['state_x'],geometry)
                if np.abs(features-independent).max()>2e-6:raise ValueError('Independent assessment feature semantics differ')
                values=np.stack([tree.predict(features,num_threads=4) for tree in trees],axis=-1).reshape(len(rows),len(links),2).transpose(0,2,1).astype(np.float32).clip(min=0);visible=a['state_x'][:,-1,:,2:4].transpose(0,2,1)>.5;out={'causal_carry':base_state,'causal_tree':np.where(visible,a['state_x'][:,-1,:,:2].transpose(0,2,1),values)}
                with torch.no_grad():
                    sx=torch.tensor(a['state_x'].transpose(0,3,1,2).copy(),device=device)
                    for key,model in models.items():out[key]=model.analyze(sx,*graph).cpu().numpy();totals['full_state_member_forwards']+=len(rows)
                for arm in ARMS:out[arm]=np.mean([out[arm+'_'+str(seed)] for seed in SEEDS],axis=0)
                for k,v in out.items():predictions.setdefault(k,[]).append(v)
                for k in ['truth_current','current_domain','current_regime','current_available']:fields.setdefault(k,[]).append(a[k])
                totals['full_state_contexts']+=len(rows)
            a={k:np.concatenate(v) for k,v in fields.items()};out={k:np.concatenate(v) for k,v in predictions.items()}
            if not np.array_equal(saved['origins'],p['timestamps'][12:288]) or any(not np.array_equal(saved[k],v) for k,v in a.items()):raise ValueError('Full State complete-day contexts or teacher differ')
            stats={k:ref_sufficient(a,v) for k,v in out.items()}
            for k,v in out.items():
                error=float(np.abs(v-saved[k]).max());maximum['analysis']=max(maximum['analysis'],error)
                if error>1e-5:raise ValueError('Full State independent neural/tree forward differs')
                same_stats(stats[k],r['models'][k])
            daily.append(stats);totals['full_state_days']+=1
        for role,wanted in case['state_assessment_metrics'].items():
            details,score=ref_combined([r[role] for r in daily])
            if not np.isclose(score,wanted['public_subset_regime_macro'],atol=1e-12,rtol=0):raise ValueError('Complete State regime score differs')
            for a,b in zip(details,wanted['details']):
                if a['n']!=b['n'] or any(not np.isclose(a[k],b[k],atol=1e-10,rtol=0) for k in ['rmse_speed_kmh','rmse_flow_per_lane','public_subset_state_formula']):raise ValueError('State normalized units or perregime formula differs')
        for phase,d in case['seals'].items():
            if sha256_file(base/(phase+'_sealed.json'))!=d:raise ValueError('Joint phase seal changed')
        for read in case['phase_reads']:
            prior={'decision':'weights','selection':'decision','assessment':'selection'}.get(read['phase'])
            if read['prior_seal_sha256']!=(case['seals'][prior] if prior else None):raise ValueError('Joint phase read order differs')
        cases.append({'name':case['name'],'verified':True,'tree_feature_hash_replay':'Canonical implementation reused for exact bytes;independentdouble-spatial feature semantics tolerance2e-6. No tree retraining.','full_state_days':len(daily),'joint_parameter_count':sum(v.numel() for v in next(iter(selected.values())).values())});print('joint_models_verified_case',case['name'],flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'totals':totals,'maximum_differences':maximum,'seconds':time.perf_counter()-tic,'source_files_rehashed':len(hashes),'formal_State':None,'formal_Queue':None,'formal_Physics':None,'formal_Total':None,
        'scope':'Fullnewfit/CP data proof reused,allevaluationandfullStatecontext/teachers rebuilt;weights/inference/selection/exposure audited. No independent network/tree re-optimization or formal score.'};raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8');print('joint_model_verification_COMPLETE',totals,'seconds',result['seconds'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import sys
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1]).with_suffix('.failed.json')
            if not path.exists():path.write_text(json.dumps({'status':'failed','traceback':traceback.format_exc(),'formal_Total':None},indent=2),encoding='utf-8')
        raise
