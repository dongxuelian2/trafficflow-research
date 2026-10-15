"""Independent training-exposure/checkpoint/grid/inference replay for C87."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from trafficflow.data import Release
from trafficflow.causal_queue import released_origin_rows
from trafficflow.experiments import partition_date
from trafficflow.graph_queue import CausalGraphQueue
from trafficflow.queue import QueueModel
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol
from verify_dense_pretrain_data import independent_day,independent_window
from verify_onset_rank_queue import manual_baseline,manual_metric,same_metric,paired,same_paired
from verify_rolling_onset_decoders import independent_scalar_score


ARMS=['scratch','event_warm','dense_warm'];SEEDS=[20261013,20261029]


def digest(state):
    stream=hashlib.sha256()
    for key in sorted(state):stream.update(key.encode());stream.update(state[key].detach().cpu().numpy().tobytes())
    return stream.hexdigest()


def event_sequence(a,seed,steps,batch):
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(a['ongoing']==c) for c in [False,True]];result=[]
    for _ in range(steps):result.append([rng.choice(groups[int(rng.integers(0,2))]) for _ in range(batch)])
    return np.array(result,np.int64)


def independent_exposure(a,batches,daypacks=None,origins=None,event_loss=False):
    ids=batches.ravel();counts={'windows_drawn':len(ids),'unique_window_contexts':len(set(ids.tolist()))};positive=regression=classification=0
    if daypacks is None:
        available=a['available'][ids];domain=available.copy()
        if event_loss:domain[:,:5]&=a['ongoing'][ids,None,None]
        regression=int(available.sum());classification=int(domain.sum());positive=int((a['labels'][ids]*domain).sum());unique=int(a['available'][np.unique(ids)].sum())
    else:
        covered={}
        for i in ids:
            day,at=map(int,origins[i]);p=daypacks[day];mask=p['available'][at+1:at+7];regression+=int(mask.sum());classification+=int(mask.sum());positive+=int((p['truth_queue'][at+1:at+7]*mask).sum());covered.setdefault(day,np.zeros(288,bool));covered[day][at+1:at+7]=True
        unique=sum(int((covered[d][:,None]&daypacks[d]['available']).sum()) for d in covered)
    counts.update(classification_target_occurrences=classification,regression_target_occurrences=regression,positive_class_occurrences=positive,unique_regression_teacher_cells=unique);return counts


def forward(state,a,geometry,device):
    model=CausalGraphQueue(hidden=32,use_graph=True).to(device);model.load_state_dict(state);model.eval();graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']];probs=[];speed=[]
    with torch.no_grad():
        for value in a['x']:
            logits,v=model(torch.tensor(value.transpose(2,0,1)[None].copy(),device=device),*graph);probs.append(expit(logits.cpu().numpy()[0].astype(float)-np.log(4.)));speed.append(v.cpu().numpy()[0])
    del model;return np.array(probs),np.array(speed)


def checkpoint_score(p,a):
    values=[]
    for i in range(len(p)):
        valid=a['available'][i].copy()
        if not a['ongoing'][i]:valid[:5]=False
        values.append(float(np.sum((p[i][valid]-a['labels'][i][valid])**2)/max(1,np.count_nonzero(valid))))
    return float(np.mean([np.mean(np.array(values)[a['ongoing']==c]) for c in [False,True]]))


def calibration(a,p):
    entries=[];checks=0
    for force in [False,True]:
        best={}
        for c in np.unique(a['condition']):
            use=a['condition']==c;part={k:a[k][use] for k in ['window','condition','lead_minutes']};part.update(labels=a['labels'].ravel()[use],available=a['available'].ravel()[use]);scores=[]
            for t in np.linspace(.05,.8,31):
                s={'thresholds':{'global':float(t)},'require_nonempty':force,'timing':'before'};scores.append((independent_scalar_score(part,manual_baseline(part,p[use],s)),float(t)));checks+=1
            best[str(c)]=max(scores,key=lambda r:(r[0],-abs(r[1]-.5)))[1]
        for mode in ['global','condition']:
            for thresholds in ([{'global':float(t)} for t in np.linspace(.05,.8,31)] if mode=='global' else [{'global':.5,**best}]):
                s={'mode':mode,'thresholds':thresholds,'require_nonempty':force,'timing':'before'};s['calibration_proxy_IoU']=independent_scalar_score(a,manual_baseline(a,p,s));entries.append(s);checks+=1
    return max(entries,key=lambda r:(r['calibration_proxy_IoU'],r['mode']=='global',not r['require_nonempty'])),checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite dense model audit')
    root=Path('reports/research');study_path=root/'C87_dense_dynamics_pretraining.json';study=json.loads(study_path.read_text());data_path=root/'C88_dense_pretraining_data_verification.json';data_proof=json.loads(data_path.read_text());study_folder=Path('artifacts/research')/study_path.stem
    if data_proof['registered']['protocol']['source_dataset_seal_sha256']!=sha256_file(study_folder/'all_datasets_sealed.json'):raise ValueError('Data proof binding changed')
    for path,d in data_proof['source_files_sha256'].items():
        if sha256_file(path)!=d:raise ValueError('Verified input data changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_dense_pretraining_model_audit','source_study_sha256':sha256_file(study_path),'source_data_audit_sha256':sha256_file(data_path),
        'scope':'Reuse fullfit/CP data proof explicitly;all model initialization/batch/exposure/checkpoint and optimizer snapshots,all decision grids/gates,new eval raw inputs/neural predictions and sampled portable interfaces. No independent retraining of12networks.', 'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');release=Release('data/kaggle_public');cases=[]
    totals={'network_fits_checked':0,'checkpoint_neural_window_forwards':0,'evaluation_neural_window_forwards':0,'new_eval_raw_window_occurrences':0,'decision_grid_score_checks':0,'portable_history_windows':0,'portable_role_calls':0,'additional_shuffled_calls':0,'future_history_rejections':0};maximum_probability=maximum_speed=0.
    for case in study['cases']:
        base=study_folder/case['name'];description=case['dataset_description'];fit_r=next(r for r in case['event_records'] if r['phase']=='fit');cp_r=next(r for r in case['event_records'] if r['phase']=='checkpoint');fit=dict(np.load(fit_r['path']));cp=dict(np.load(cp_r['path']));packs=[dict(np.load(r['path'])) for r in description['day_records']];origins=np.load(description['origin_record']['path'])['indices'];geometry=dict(np.load(case['geometry']['path']));initials={}
        for r in case['initial_records']:
            torch.manual_seed(r['seed']);model=CausalGraphQueue(hidden=32,use_graph=True);expected=model.state_dict();saved=torch.load(r['path'],map_location='cpu',weights_only=True)
            if sha256_file(r['path'])!=r['sha256'] or any(not torch.equal(expected[k],saved[k]) for k in expected) or digest(saved)!=r['state_digest']:raise ValueError('Common initialization differs')
            initials[r['seed']]=saved;del model
        sequences={}
        for r in case['batch_records']:
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Batch archive changed')
            a=dict(np.load(r['path']));seed=r['seed'];expected={'finetune':event_sequence(fit,seed+100003,800,4),'event_warm':event_sequence(fit,seed+500003,400,8),'dense_warm':np.random.default_rng(seed+500003).integers(0,len(origins),(400,8),dtype=np.int64)}
            if any(not np.array_equal(a[k],v) for k,v in expected.items()):raise ValueError('Per-seed matched batch sequence differs')
            sequences[seed]=a
        selected={}
        for t in case['training']:
            seq=sequences[t['seed']];arm=t['arm']
            if t['initial_state_digest']!=digest(initials[t['seed']]) or t['finetune_exposure']!=independent_exposure(fit,seq['finetune'],event_loss=True):raise ValueError('Initialization/actual finetune exposure differs')
            pre=torch.load(t['before_event']['path'],map_location='cpu',weights_only=True)
            if sha256_file(t['before_event']['path'])!=t['before_event']['sha256'] or digest(pre)!=t['before_event_state_digest']:raise ValueError('Warm-up state changed')
            if arm=='scratch':
                if t['warmup_exposure'] is not None or any(not torch.equal(pre[k],initials[t['seed']][k]) for k in pre):raise ValueError('Scratch control was warm-started')
            else:
                exposure=independent_exposure(fit,seq['event_warm']) if arm=='event_warm' else independent_exposure(None,seq['dense_warm'],packs,origins)
                if exposure!=t['warmup_exposure'] or t['warmup_steps']!=400:raise ValueError('Matched warm-up exposure differs')
            for r in t['snapshots']:
                if sha256_file(r['path'])!=r['sha256'] or sha256_file(r['checkpoint_predictions']['path'])!=r['checkpoint_predictions']['sha256'] or sha256_file(r['optimizer']['path'])!=r['optimizer']['sha256']:raise ValueError('Weight/optimizer/checkpoint predictions changed')
                state=torch.load(r['path'],map_location='cpu',weights_only=True);optimizer=torch.load(r['optimizer']['path'],map_location='cpu',weights_only=True)
                if digest(state)!=r['state_digest'] or optimizer['step']!=r['step'] or any(int(v['step'])!=r['step'] for v in optimizer['optimizer']['state'].values()):raise ValueError('Snapshot/optimizer reset/step differs')
                p,v=forward(state,cp,geometry,device);totals['checkpoint_neural_window_forwards']+=len(cp['x'])
                with np.load(r['checkpoint_predictions']['path']) as saved:ep=float(np.max(np.abs(p-saved['probability'])));ev=float(np.max(np.abs(v-saved['speed'])))
                maximum_probability=max(maximum_probability,ep);maximum_speed=max(maximum_speed,ev)
                if ep>1e-6 or ev>1e-5 or abs(checkpoint_score(p,cp)-r['checkpoint_Brier'])>1e-10:raise ValueError('Checkpoint predictions/Brier differ')
            chosen=min(t['snapshots'],key=lambda r:r['checkpoint_Brier'])
            if chosen!=t['selected']:raise ValueError('Checkpoint-only winner differs')
            selected[(arm,t['seed'])]=torch.load(chosen['path'],map_location='cpu',weights_only=True);totals['network_fits_checked']+=1
        portable={}
        for role,r in case['exports'].items():
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Portable network changed')
            model=QueueModel.load(Path(r['path']));portable[role]=model;candidate=r['candidate']
            if candidate!='incumbent':
                if model.alpha!=.25 or model.setting!=case['choices'][candidate]['setting'] or len(model.states)!=2:raise ValueError('Fixed blend/export choice differs')
                for state,seed in zip(model.states,SEEDS):
                    if any(not torch.equal(state[k],selected[(candidate,seed)][k]) for k in state):raise ValueError('Exported checkpoint state differs')
        panel=case['panel'];net,topology=release.network(panel),release.topology(panel);links=fit['links'].tolist();raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};masked_paths={partition_date(release,p):p for p in release.files(panel,'train')};source_rolling=json.loads((root/'C71_rolling_onset_threshold_refits.json').read_text());old=next(c for c in source_rolling['cases'] if c['name']==case['name'])
        @lru_cache(maxsize=8)
        def day(date):return independent_day(raw_paths[date],masked_paths[date],links,net)
        @lru_cache(maxsize=8)
        def raw_day(date):return release.read(raw_paths[date])
        evaluated={'selection':[],'assessment':[]};prediction_audits=[];poisoned=False
        for r in case['event_records']:
            phase=r['phase']
            if phase in ['fit','checkpoint']:continue
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Fresh eval cache changed')
            a=dict(np.load(r['path']))
            for i,text in enumerate(a['origins']):
                origin=pd.Timestamp(str(text));at=int((origin-pd.Timestamp(origin.date().isoformat(),tz='UTC'))/pd.Timedelta(minutes=5));expected=independent_window(day(origin.date().isoformat()),at,bool(a['ongoing'][i]))
                if any(not np.array_equal(value,a[k][i]) for value,k in zip(expected,['x','labels','truth_speed','available'])):raise ValueError('Fresh eval causal input/teacher differs')
                totals['new_eval_raw_window_occurrences']+=1
            source=next(q for q in old['cache_records'] if q['phase']==phase and q['seed']==r['seed']);f=next(q for q in source['feature_files'] if Path(q['path']).stem.endswith('_'+old['feature_arm']))
            if sha256_file(f['path'])!=f['sha256']:raise ValueError('Previously verified tree feature cache changed')
            tp=portable['incumbent'].baseline.probability(pd.read_parquet(f['path']));preds={};means={};velocities={};audit_arrays={}
            for arm in ARMS:
                member_p=[];member_v=[]
                for seed in SEEDS:
                    p,v=forward(selected[(arm,seed)],a,geometry,device);member_p.append(p);member_v.append(v);totals['evaluation_neural_window_forwards']+=len(a['x']);audit_arrays[arm+'_'+str(seed)+'_probability']=p;audit_arrays[arm+'_'+str(seed)+'_speed']=v
                means[arm]=np.mean(member_p,axis=0).ravel();velocities[arm]=np.mean(member_v,axis=0)
            if phase=='decision':
                for arm in ARMS:
                    setting,n=calibration(a,.75*tp+.25*means[arm]);totals['decision_grid_score_checks']+=n
                    if any(setting[k]!=case['choices'][arm]['setting'][k] for k in ['mode','thresholds','require_nonempty','timing']) or abs(setting['calibration_proxy_IoU']-case['choices'][arm]['setting']['calibration_proxy_IoU'])>1e-12:raise ValueError('Independent full decision grid winner differs')
            for role in ['incumbent']+ARMS:preds[role]=manual_baseline(a,tp if role=='incumbent' else .75*tp+.25*means[role],case['choices'][role]['setting'])
            if phase=='assessment':preds['selected']=preds[case['selected']]
            metrics={role:manual_metric(a,pred) for role,pred in preds.items()}
            if phase=='decision':
                for role,m in metrics.items():same_metric(m,case['choices'][role]['metrics'])
                if max(['incumbent']+ARMS,key=lambda role:metrics[role]['proxy_IoU'])!=case['decision_selected']:raise ValueError('Decision challenger differs')
            else:
                row=next(q for q in case['prediction_records'] if q['phase']==phase and q['seed']==r['seed'])
                if sha256_file(row['path'])!=row['sha256']:raise ValueError('Stored neural forecasts changed')
                with np.load(row['path']) as saved:
                    for arm in ARMS:
                        ep=float(np.abs(means[arm]-saved[arm+'_probability']).max());ev=float(np.abs(velocities[arm]-saved[arm+'_speed']).max());maximum_probability=max(maximum_probability,ep);maximum_speed=max(maximum_speed,ev)
                        if ep>1e-6 or ev>1e-5:raise ValueError('Independent selected network forecast differs')
                    if any(not np.array_equal(saved[k],pred) for k,pred in preds.items()):raise ValueError('Independent binary forecast differs')
                wanted=next(q for q in case[phase] if q['seed']==r['seed'])
                for role,m in metrics.items():same_metric(m,wanted['models'][role])
                evaluated[phase].append({'seed':r['seed'],'models':metrics})
            p=folder/(case['name']+'_'+phase+'_'+str(r['seed'])+'_members.npz');np.savez_compressed(p,**audit_arrays);prediction_audits.append({'phase':phase,'seed':r['seed'],'path':p.as_posix(),'sha256':sha256_file(p),'source_event_record':r})
            sample=set();first=set();width=6*len(links)
            for ongoing in [False,True]:
                ids=np.flatnonzero(a['ongoing']==ongoing);sample.update([int(ids[0]),int(ids[-1])]);first.add(int(ids[0]))
            for i in sorted(sample):
                origin=pd.Timestamp(str(a['origins'][i]));raw=raw_day(origin.date().isoformat());ts=pd.to_datetime(raw.timestamp,utc=True);history=raw.loc[(ts>=origin-pd.Timedelta(minutes=60))&(ts<origin)].copy();history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True);future=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]];target=pd.MultiIndex.from_product([future,links],names=['timestamp','link_id']).to_frame(index=False);offset=slice(i*width,(i+1)*width);condition='queue_ongoing' if a['ongoing'][i] else 'queue_onset';totals['portable_history_windows']+=1
                for role,model in portable.items():
                    wanted=preds[case['exports'][role]['candidate']][offset];out=model.predict_window(history,target,net,topology,origin,condition);totals['portable_role_calls']+=1
                    if not np.array_equal(out.queue_pred,wanted):raise ValueError('Portable graph warmup forecast differs')
                    if i in first:
                        order=np.random.default_rng(107).permutation(width);out=model.predict_window(history,target.iloc[order],net,topology,origin,condition);totals['additional_shuffled_calls']+=1
                        if not np.array_equal(out.queue_pred,wanted[order]):raise ValueError('Graph warmup forecast depends on target order')
                if condition=='queue_onset' and not poisoned:
                    bad=pd.concat([history,history.iloc[[0]].assign(timestamp=future[0])],ignore_index=True)
                    for model in portable.values():
                        try:model.predict_window(bad,target,net,topology,origin,condition)
                        except ValueError:totals['future_history_rejections']+=1
                        else:raise ValueError('Graph warmup model accepted future input')
                    poisoned=True
        gate=paired(evaluated['selection'],case['dates']['selection'],'incumbent',case['decision_selected']);same_paired(gate,case['selection_gate']);chosen=case['decision_selected'] if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent'
        if chosen!=case['selected']:raise ValueError('Later selection gate differs')
        for role,value in case['comparisons'].items():same_paired(paired(evaluated['assessment'],case['dates']['assessment'],'incumbent',role),value)
        for phase,digest_value in case['seals'].items():
            if sha256_file(base/(phase+'_sealed.json'))!=digest_value:raise ValueError('Choice seal changed')
        for read in case['phase_reads']:
            required={'decision':'weights','selection':'decision','assessment':'selection'}.get(read['phase'])
            if read['prior_seal_sha256']!=(case['seals'][required] if required else None):raise ValueError('Model selection phase read order differs')
        cases.append({'name':case['name'],'initialization_batches_exposures_checkpoints_grids_gates_verified':True,'prediction_audits':prediction_audits});print('dense_pretrain_models_verified_case',case['name'],flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'totals':totals,'maximum_probability_difference':maximum_probability,'maximum_normalized_speed_difference':maximum_speed,'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,
        'scope':'Fullfit/CP data proof reused;all new eval inputs reconstructed,checkpoint/inference/exposure/selection independently replayed. No independent re-optimization/training of12networks,or formal/latent physics score.'};raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8');print('dense_pretrain_model_verification_COMPLETE',totals,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
