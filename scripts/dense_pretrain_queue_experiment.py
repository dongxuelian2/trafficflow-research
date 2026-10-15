"""Matched graph Queue scratch/event-warm/dense-warm experiments."""
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
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.dense_queue_data import prepare_dense_day,dense_origins,dense_window
from trafficflow.dense_queue_loss import dense_forecast_loss
from trafficflow.graph_queue import CausalGraphQueue,queue_event_loss
from trafficflow.queue_graph_geometry import travel_time_neighborhoods
from trafficflow.probability_calibration import weighted_bce_probability
from trafficflow.full_queue_calibration import tune_full_probabilities,full_queue_decisions
from trafficflow.ensemble_queue import EnsembleFullDomainQueueModel
from trafficflow.queue import QueueModel
from trafficflow.queue_domain import score_full_forecast
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from compare_queue_events import comparison


ARMS=['scratch','event_warm','dense_warm'];SEEDS=[20261013,20261029];EVENT_SEEDS=[2909,3253,4001]
PRETRAIN_STEPS=400;FINETUNE_STEPS=800


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def record(path):return {'path':Path(path).as_posix(),'sha256':sha256_file(path)}


def state_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Dataset:
    def __init__(self,release,case,calibration,folder):
        self.release=release;self.case=case;self.calibration=calibration;self.folder=folder;self.panel=case['panel'];self.net=release.network(self.panel);self.topology=release.topology(self.panel)
        self.raw_paths={partition_date(release,p):p for p in release.files(self.panel,'train',False)};self.masked_paths={partition_date(release,p):p for p in release.files(self.panel,'train')};self.source_files={};self.day_records=[];self.event_records=[];self.phase_reads=[]
        fit_record=next(r for r in case['cache_records'] if r['phase']=='fit');self.links=np.load(fit_record['path'])['links'].tolist();self.packs=[];self.fit_day_index={};self.origins=[]

    def read_day(self,date):
        raw,masked=self.raw_paths[date],self.masked_paths[date]
        for p in [raw,masked]:self.source_files[p.as_posix()]=sha256_file(p)
        return prepare_dense_day(self.release.read(raw),self.release.read(masked),self.links,self.net)

    @lru_cache(maxsize=8)
    def held_day(self,date):return self.read_day(date)

    def prepare_fit(self):
        for i,date in enumerate(self.case['dates']['fit']):
            pack=self.read_day(date);path=self.folder/('fit_day_'+date+'.npz');np.savez_compressed(path,**pack);self.fit_day_index[date]=i;self.packs.append(pack);self.origins.extend((i,int(at)) for at in dense_origins(pack));self.day_records.append({'date':date,'phase':'fit',**record(path),'raw':record(self.raw_paths[date]),'masked':record(self.masked_paths[date])})
        self.origins=np.array(self.origins,np.int32);p=self.folder/'dense_origins.npz';np.savez_compressed(p,indices=self.origins);self.origin_record=record(p)
        return self.load_event('fit',20261013)

    def load_event(self,phase,seed):
        prior={'decision':'weights','selection':'decision','assessment':'selection'}.get(phase)
        if prior is not None and not (self.folder/(prior+'_sealed.json')).exists():raise ValueError('Earlier seal required before '+phase)
        source=self.calibration['calibration_record'] if phase=='checkpoint' else next(r for r in self.case['cache_records'] if r['phase']==phase and r['seed']==seed)
        if sha256_file(source['path'])!=source['sha256']:raise ValueError('Frozen event metadata changed')
        a=dict(np.load(source['path']));xs=[];ys=[];speeds=[];available=[]
        for i,text in enumerate(a['origins']):
            origin=pd.Timestamp(str(text));date=origin.date().isoformat();pack=self.packs[self.fit_day_index[date]] if phase=='fit' else self.held_day(date);at=int((origin-pd.Timestamp(date,tz='UTC'))/pd.Timedelta(minutes=5));window=dense_window(pack,at,ongoing=bool(a['ongoing'][i]))
            if window['inferred_ongoing']!=bool(a['ongoing'][i]) or not np.array_equal(window['labels'],a['labels'][i]) or not np.array_equal(window['available'],a['available'][i]):raise ValueError('Fresh event contract differs')
            if 'x' in a and not np.array_equal(window['x'],a['x'][i]):raise ValueError('Fresh neural history differs from old event cache')
            if 'truth_speed' in a and not np.array_equal(window['truth_speed'],a['truth_speed'][i]):raise ValueError('Fresh speed teacher differs')
            xs.append(window['x']);ys.append(window['labels']);speeds.append(window['truth_speed']);available.append(window['available'])
        a.update(x=np.stack(xs),labels=np.stack(ys),truth_speed=np.stack(speeds),available=np.stack(available));path=self.folder/(phase+'_'+str(seed)+'_neural.npz');np.savez_compressed(path,**a);r={'phase':phase,'seed':seed,**record(path),'source_record':source,'n_windows':len(xs)};self.event_records.append(r);self.phase_reads.append({'phase':phase,'seed':seed,'prior_seal_sha256':sha256_file(self.folder/(prior+'_sealed.json')) if prior else None});return a

    def dense_batch(self,ids):
        windows=[dense_window(self.packs[int(self.origins[i,0])],int(self.origins[i,1])) for i in ids]
        return {k:np.stack([w[k] for w in windows]) for k in ['x','labels','truth_speed','available']}


def event_batches(a,seed,steps,batch):
    rng=np.random.default_rng(seed);groups={c:np.flatnonzero(a['ongoing']==c) for c in [False,True]}
    return np.array([[rng.choice(groups[bool(rng.integers(2))]) for _ in range(batch)] for _ in range(steps)],np.int64)


def exposure(data,batches,dense=False,event_loss=False):
    ids=np.unique(batches);counts={'windows_drawn':int(batches.size),'unique_window_contexts':len(ids),'classification_target_occurrences':0,'regression_target_occurrences':0,'positive_class_occurrences':0};unique_masks={}
    for indices in batches:
        a=data.dense_batch(indices) if dense else {k:data[k][indices] for k in ['labels','truth_speed','available','ongoing']};domain=a['available'].copy()
        if event_loss:domain[:,:5]&=a['ongoing'][:,None,None]
        counts['classification_target_occurrences']+=int(domain.sum());counts['regression_target_occurrences']+=int(a['available'].sum());counts['positive_class_occurrences']+=int((a['labels']*domain).sum())
    if dense:
        for i in ids:
            day,at=map(int,data.origins[i]);unique_masks.setdefault(day,np.zeros(data.packs[day]['available'].shape,bool));unique_masks[day][at+1:at+7]|=data.packs[day]['available'][at+1:at+7]
        counts['unique_regression_teacher_cells']=sum(int(m.sum()) for m in unique_masks.values())
    else:counts['unique_regression_teacher_cells']=int(data['available'][ids].sum())
    return counts


def predict(model,a,graph,device):
    probs=[];speed=[];model.eval()
    with torch.no_grad():
        for x in a['x']:
            logits,v=model(torch.from_numpy(x.transpose(2,0,1)[None].copy()).to(device),*graph);probs.append(weighted_bce_probability(logits.cpu().numpy()[0],4.));speed.append(v.cpu().numpy()[0])
    return np.stack(probs),np.stack(speed)


def brier(p,a):
    domain=a['available'].copy();domain[:,:5]&=a['ongoing'][:,None,None];loss=((p-a['labels'])**2*domain).sum(axis=(1,2))/domain.sum(axis=(1,2)).clip(min=1)
    return float(np.mean([loss[a['ongoing']==c].mean() for c in [False,True]]))


def fit_model(arm,seed,initial,data,fit,checkpoint,graph,device,folder,finetune_batches,pretrain_batches):
    model=CausalGraphQueue(hidden=32,use_graph=True).to(device);model.load_state_dict(initial);initial_digest=state_digest(model.state_dict());tic=time.perf_counter();stage_history=[]
    pre_exposure=None
    if arm!='scratch':
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);pre_exposure=exposure(data if arm=='dense_warm' else fit,pretrain_batches,dense=arm=='dense_warm');model.train();losses=[]
        for step,indices in enumerate(pretrain_batches,1):
            a=data.dense_batch(indices) if arm=='dense_warm' else {k:fit[k][indices] for k in ['x','labels','truth_speed','available']};x=torch.from_numpy(a['x'].transpose(0,3,1,2).copy()).to(device);targets=[torch.from_numpy(a[k]).to(device) for k in ['labels','truth_speed','available']];optimizer.zero_grad(set_to_none=True);logits,v=model(x,*graph);loss=dense_forecast_loss(logits,v,*targets)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite warm-up loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
            if step%100==0:stage_history.append({'stage':'warm','step':step,'loss':float(np.mean(losses[-100:]))});print('dense_pretrain',data.case['name'],arm,seed,step,flush=True)
        del optimizer
    pre_path=folder/(arm+'_'+str(seed)+'_before_event.pt');torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},pre_path);pre_digest=state_digest(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);snapshots=[];best=float('inf');selected=None;losses=[]
    for step,indices in enumerate(finetune_batches,1):
        model.train();x=torch.from_numpy(fit['x'][indices].transpose(0,3,1,2).copy()).to(device);targets=[torch.from_numpy(fit[k][indices]).to(device) for k in ['labels','truth_speed','available','ongoing']];optimizer.zero_grad(set_to_none=True);logits,v=model(x,*graph);loss=queue_event_loss(logits,v,*targets)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite fine-tune loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
        if step%200==0:
            p,s=predict(model,checkpoint,graph,device);value=brier(p,checkpoint);path=folder/(arm+'_'+str(seed)+'_step'+str(step)+'.pt');torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path);optimizer_path=path.with_suffix('.optimizer.pt');torch.save({'optimizer':optimizer.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device.type=='cuda' else None},optimizer_path)
            pred_path=path.with_suffix('.checkpoint.npz');np.savez_compressed(pred_path,probability=p,speed=s);row={'step':step,'checkpoint_Brier':value,'mean_training_loss':float(np.mean(losses[-200:])),**record(path),'state_digest':state_digest(model.state_dict()),'optimizer':record(optimizer_path),'checkpoint_predictions':record(pred_path)};snapshots.append(row)
            if value<best:best=value;selected=row
            print('dense_finetune',data.case['name'],arm,seed,step,'checkpoint_Brier',value,flush=True)
    result={'arm':arm,'seed':seed,'initial_state_digest':initial_digest,'before_event':record(pre_path),'before_event_state_digest':pre_digest,'warmup_steps':0 if arm=='scratch' else PRETRAIN_STEPS,'finetune_steps':FINETUNE_STEPS,'warmup_exposure':pre_exposure,'finetune_exposure':exposure(fit,finetune_batches,event_loss=True),'history':stage_history,'snapshots':snapshots,'selected':selected,'seconds':time.perf_counter()-tic};del optimizer,model;return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();report=Path('reports/research')/(args.output.name+'.json')
    if report.exists() or args.output.exists():raise FileExistsError('Never overwrite dense pretraining study')
    root=Path('reports/research');source_path=root/'C71_rolling_onset_threshold_refits.json';source=json.loads(source_path.read_text());cal_path=root/'C74_probabilistic_onset_sets.json';cal=json.loads(cal_path.read_text());population_path=root/'C85_dense_forecast_fit_population.json';population=json.loads(population_path.read_text());jobs=[c for c in source['cases'] if c['name'] in ['I5S_September','I10W_September']]
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Matched_dense_dynamics_warmup_and_event_finetuning','previous_goal_turn':'progress:ab60160 rejected decoder screen,audited dense support','source_rolling_study_sha256':sha256_file(source_path),'source_checkpoint_metadata_sha256':sha256_file(cal_path),'source_population_sha256':sha256_file(population_path),'jobs':[{'name':c['name'],'panel':c['panel'],'dates':c['dates']} for c in jobs],
        'arms':ARMS,'seeds':SEEDS,'architecture':'Unchanged CausalGraphQueue hidden32/use_graphTrue,13slots/9channels,6speed+6queue outputs. Identical initialization per seed across3arms.',
        'warmup':'scratch0;event_warm/dense_warm400steps batch8,AdamW.001/wd1e-4/clip1,same all6step posweight4BCE+.1L1speed. Event population balanced bycondition;dense uniform all5minute legal FIT origins. Data/sampling differ,not claimed pure architecture gain.',
        'finetune':'All3arms800steps batch4,identical per-seed balanced-event batches and original event loss. Reset optimizer before common fine-tune. Preserve actual exposures and batch arrays.',
        'checkpoint':'Existing15day buffer event metadata;fresh legal input/teacher reconstruction,minimum condition/window Brier at200/400/600/800. Analytic weighted-BCE odds correction only,no fitted probability calibration.',
        'decision':'Existing30days. Frozen strong tree vs3seed-means,eachfixed.75tree+.25NN;complete-domain threshold calibration. Preferincumbent/scratch/event_warm/dense_warm in tie;individualseeds diagnostics,never primary choices.',
        'selection':'15days/3eventseeds,gain>=.01andpaired7dayblock95CI lower>0;sealbefore30dayassessment.',
        'inputs':'Pure indexed<=T history/rawpast12 plus published maskedT;future labels/speed/quality loss/evaluation only. Dense history-only condition proxy differs from selection guarantee,then event finetune.',
        'scope':'Previously viewed public observation-proxy development dates,not latent/final/private. No new formal score/submission/platform operations;finalsim seeds unopened.'})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    release=Release('data/kaggle_public');prepared=[];dataset_descriptions=[]
    for case in jobs:
        folder=args.output/case['name'];folder.mkdir();old_cal=next(c for c in cal['cases'] if c['name']==case['name']);data=Dataset(release,case,old_cal,folder);fit=data.prepare_fit();checkpoint=data.load_event('checkpoint',4129);expected=next(c for c in population['cases'] if c['name']==case['name'])['summaries']['5']['n_history_valid_origins']
        if len(data.origins)!=expected:raise ValueError('Dense origin population differs')
        lane=data.net.lanes.reindex(data.links).to_numpy(float);geometry={'incoming':travel_time_neighborhoods(data.links,data.net,data.topology,30.)[0],'outgoing':travel_time_neighborhoods(data.links,data.net,data.topology,30.)[1],'geometry':np.stack([data.net.free_speed_kmh.reindex(data.links),data.net.capacity_vph.reindex(data.links)/lane,data.net.k_jam.reindex(data.links)/lane,data.net.v_cut.reindex(data.links)]).astype(np.float32)};p=folder/'geometry.npz';np.savez_compressed(p,**geometry)
        description={'name':case['name'],'panel':case['panel'],'dates':case['dates'],'day_records':data.day_records,'origin_record':data.origin_record,'event_records':list(data.event_records),'source_files_sha256':dict(data.source_files),'geometry':record(p)};save(folder/'dataset_sealed.json',description);dataset_descriptions.append(description);prepared.append((data,fit,checkpoint,geometry))
        print('dense_datasets_prepared',case['name'],'origins',len(data.origins),'fit_events',len(fit['x']),flush=True)
    save(args.output/'all_datasets_sealed.json',{'datasets':dataset_descriptions});cases=[]
    for data,fit,checkpoint,geometry in prepared:
        folder=data.folder;graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']];training=[];batch_records=[];initial_records=[]
        for seed in SEEDS:
            torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            model=CausalGraphQueue(hidden=32,use_graph=True);initial={k:v.detach().clone() for k,v in model.state_dict().items()};del model;p=folder/('common_initial_'+str(seed)+'.pt');torch.save(initial,p);initial_records.append({'seed':seed,**record(p),'state_digest':state_digest(initial)})
            common=event_batches(fit,seed+100003,FINETUNE_STEPS,4);event_pre=event_batches(fit,seed+500003,PRETRAIN_STEPS,8);rng=np.random.default_rng(seed+500003);dense_pre=rng.integers(0,len(data.origins),(PRETRAIN_STEPS,8),dtype=np.int64);bp=folder/('batch_sequences_'+str(seed)+'.npz');np.savez_compressed(bp,finetune=common,event_warm=event_pre,dense_warm=dense_pre);batch_records.append({'seed':seed,**record(bp)})
            for arm in ARMS:training.append(fit_model(arm,seed,initial,data,fit,checkpoint,graph,device,folder,common,event_pre if arm=='event_warm' else dense_pre))
        save(folder/'weights_sealed.json',{'training':training,'batch_records':batch_records,'initial_records':initial_records});reference_record=data.case['exports']['incumbent'] if 'exports' in data.case else data.case['exported_models']['incumbent']
        # C71 stores the portable strong reference in exports.
        if sha256_file(reference_record['path'])!=reference_record['sha256']:raise ValueError('Frozen reference changed')
        reference=QueueModel.load(Path(reference_record['path']));decision=data.load_event('decision',4133);feature_record=next(r for r in data.case['cache_records'] if r['phase']=='decision');arm_name=data.case['feature_arm'];f=next(f for f in feature_record['feature_files'] if Path(f['path']).stem.endswith('_'+arm_name));features=pd.read_parquet(f['path']);tree_p=reference.baseline.probability(features)
        def neural(a):
            probs={};speed={}
            for arm in ARMS:
                members=[];velocities=[]
                for t in training:
                    if t['arm']!=arm:continue
                    net=CausalGraphQueue(hidden=32,use_graph=True).to(device);net.load_state_dict(torch.load(t['selected']['path'],map_location=device,weights_only=True));p,v=predict(net,a,graph,device);members.append(p);velocities.append(v);del net
                probs[arm]=np.mean(members,axis=0).ravel();speed[arm]=np.mean(velocities,axis=0)
            return probs,speed
        def score(a,p,setting):
            pred=full_queue_decisions(p,a['window'],a['condition'],a['lead_minutes'],setting['thresholds'],setting['require_nonempty'],setting['timing']);return pred,score_full_forecast(a['labels'].ravel(),pred,a['window'],a['condition'],a['available'].ravel())
        dp,dv=neural(decision);choices={'incumbent':{'setting':reference.setting,'metrics':score(decision,tree_p,reference.setting)[1]}}
        for arm in ARMS:
            p=.75*tree_p+.25*dp[arm];setting=tune_full_probabilities(p,decision['window'],decision['condition'],decision['lead_minutes'],decision['labels'].ravel(),decision['available'].ravel())['selected'];choices[arm]={'setting':setting,'metrics':score(decision,p,setting)[1]}
        challenger=max(['incumbent']+ARMS,key=lambda n:choices[n]['metrics']['proxy_IoU']);save(folder/'decision_sealed.json',{'choices':choices,'challenger':challenger});prediction_records=[]
        def evaluate(phase,selected=None):
            result={'dates':{'assessment':data.case['dates'][phase]},'assessment':[]}
            for seed in EVENT_SEEDS:
                a=data.load_event(phase,seed);r=next(r for r in data.case['cache_records'] if r['phase']==phase and r['seed']==seed);f=next(f for f in r['feature_files'] if Path(f['path']).stem.endswith('_'+arm_name));tp=reference.baseline.probability(pd.read_parquet(f['path']));p_nn,v_nn=neural(a);pred={};metrics={}
                for role in ['incumbent']+ARMS:pred[role],metrics[role]=score(a,tp if role=='incumbent' else .75*tp+.25*p_nn[role],choices[role]['setting'])
                if selected is not None:pred['selected']=pred[selected];metrics['selected']=metrics[selected]
                path=folder/(phase+'_'+str(seed)+'_predictions.npz');np.savez_compressed(path,tree_probability=tp,**{arm+'_probability':p_nn[arm] for arm in ARMS},**{arm+'_speed':v_nn[arm] for arm in ARMS},**pred);prediction_records.append({'phase':phase,'seed':seed,**record(path)});result['assessment'].append({'seed':seed,'models':metrics})
            return result
        selection=evaluate('selection');gate=comparison(selection,'incumbent',challenger);selected=challenger if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent';save(folder/'selection_sealed.json',{'selected':selected,'decision_selected':challenger,'gate':gate});portable={'incumbent':reference}
        for arm in ARMS:
            states=[torch.load(t['selected']['path'],map_location='cpu',weights_only=True) for t in training if t['arm']==arm];portable[arm]=EnsembleFullDomainQueueModel(reference.baseline,data.links,choices[arm]['setting'],.25,states,geometry)
        portable['decision_selected']=portable[challenger];portable['selected']=portable[selected];exports={}
        for role,model in portable.items():
            p=folder/(role+'.pkl');model.save(p);exports[role]={**record(p),'candidate':challenger if role=='decision_selected' else selected if role=='selected' else role}
        assessment=evaluate('assessment',selected);case={'name':data.case['name'],'panel':data.panel,'dates':data.case['dates'],'dataset_description':dataset_descriptions[len(cases)],'geometry':record(folder/'geometry.npz'),'initial_records':initial_records,'batch_records':batch_records,'training':training,'choices':choices,'decision_selected':challenger,'selected':selected,'selection_gate':gate,'selection':selection['assessment'],'assessment':assessment['assessment'],'comparisons':{role:comparison(assessment,'incumbent',role) for role in ARMS+['selected']},'event_records':data.event_records,'phase_reads':data.phase_reads,'source_files_sha256':data.source_files,'prediction_records':prediction_records,'exports':exports,'seals':{n:sha256_file(folder/(n+'_sealed.json')) for n in ['dataset','weights','decision','selection']}}
        cases.append(case);save(folder/'case_completed.json',case);print('dense_pretraining_case_COMPLETE',data.case['name'],'decision',challenger,'selected',selected,flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'new_network_fits':12,'seconds':time.perf_counter()-start,'source':evidence(gpu_used=device.type=='cuda'),'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,'formal_Queue':None,'formal_Total':None,'formal_model_promoted':False,'new_complete_submission':False,'scope':'Public observed-label proxy,not latent/final/private;matched warm compute control and finetune batches,extra dense contexts and objective/sampling exposure recorded. No formal score/promotion.'};save(report,result);save(args.output/'experiment.json',result);print('dense_pretraining_COMPLETE','seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
