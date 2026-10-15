"""Matched flow supervision / residual latent propagation on public data."""
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
from lightgbm import LGBMRegressor
from trafficflow.data import Release
from trafficflow.joint_queue_data import prepare_joint_day,joint_window,state_window,latest_state,reconstruction_features
from trafficflow.joint_graph_queue import JointGraphQueue,joint_dynamics_loss
from trafficflow.joint_queue_model import JointDynamicsQueueModel
from trafficflow.joint_dynamics_eval import state_sufficient,combine_state,forecast_sufficient
from trafficflow.probability_calibration import weighted_bce_probability
from trafficflow.full_queue_calibration import tune_full_probabilities,full_queue_decisions
from trafficflow.queue_domain import score_full_forecast
from trafficflow.queue import QueueModel
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from dense_pretrain_queue_experiment import Dataset,save,record,state_digest,brier
from compare_queue_events import comparison


ARMS=['direct_no_flow','direct_flow','latent_flow'];SEEDS=[20261013,20261029];EVENT_SEEDS=[2909,3253,4001]
ARRAYS=['x','state_x','labels','truth_speed','truth_flow','available','ongoing','truth_current','current_domain','current_regime','current_available']
TENSORS=['labels','truth_speed','truth_flow','available','ongoing','truth_current','current_domain']


class JointDataset(Dataset):
    def read_day(self,date):
        raw,masked=self.raw_paths[date],self.masked_paths[date]
        for p in [raw,masked]:self.source_files[p.as_posix()]=sha256_file(p)
        return prepare_joint_day(self.release.read(raw),self.release.read(masked),self.links,self.net)

    def load_event(self,phase,seed):
        prior={'decision':'weights','selection':'decision','assessment':'selection'}.get(phase)
        if prior and not (self.folder/(prior+'_sealed.json')).exists():raise ValueError('Prior seal missing')
        source=self.calibration['calibration_record'] if phase=='checkpoint' else next(r for r in self.case['cache_records'] if r['phase']==phase and r['seed']==seed)
        if sha256_file(source['path'])!=source['sha256']:raise ValueError('Original event metadata changed')
        a=dict(np.load(source['path']));rows=[]
        for i,text in enumerate(a['origins']):
            origin=pd.Timestamp(str(text));date=origin.date().isoformat();pack=self.packs[self.fit_day_index[date]] if phase=='fit' else self.held_day(date);at=int((origin-pd.Timestamp(date,tz='UTC'))/pd.Timedelta(minutes=5));w=joint_window(pack,at,bool(a['ongoing'][i]))
            for k in ['x','labels','truth_speed','available']:
                if k in a and not np.array_equal(w[k],a[k][i]):raise ValueError('Old event tensor differs: '+k)
            if w['inferred_ongoing']!=bool(a['ongoing'][i]):raise ValueError('History-only event condition differs')
            rows.append(w)
        a.update({k:np.stack([w[k] for w in rows]) for k in ARRAYS});p=self.folder/(phase+'_'+str(seed)+'_joint.npz');np.savez_compressed(p,**a);self.event_records.append({'phase':phase,'seed':seed,**record(p),'source_record':source,'n_windows':len(rows)});self.phase_reads.append({'phase':phase,'seed':seed,'prior_seal_sha256':sha256_file(self.folder/(prior+'_sealed.json')) if prior else None});return a

    def dense_batch(self,ids):
        rows=[joint_window(self.packs[int(self.origins[i,0])],int(self.origins[i,1])) for i in ids]
        return {k:np.stack([w[k] for w in rows]) for k in ARRAYS}


def tensors(a,device):return {k:torch.tensor(a[k],device=device) for k in TENSORS}


def inference(model,a,graph,device):
    result={k:[] for k in ['probability','speed','flow','analysis']};model.eval()
    with torch.no_grad():
        for i,x in enumerate(a['x']):
            out=list(model(torch.tensor(x.transpose(2,0,1)[None].copy(),device=device),*graph));out[3]=model.analyze(torch.tensor(a['state_x'][i].transpose(2,0,1)[None].copy(),device=device),*graph)
            values=[v.cpu().numpy()[0] for v in out];values[0]=weighted_bce_probability(values[0],4.)
            for k,v in zip(result,values):result[k].append(v)
    return {k:np.stack(v) for k,v in result.items()}


def exposure(data,batches,dense,event):
    fields={'windows_drawn':int(batches.size),'unique_contexts':len(np.unique(batches)),'classification_cells':0,'positive_class_cells':0,'forecast_speed_flow_cells':0,'current_masked_speed_cells':0,'current_masked_flow_cells':0}
    for ids in batches:
        a=data.dense_batch(ids) if dense else {k:data[k][ids] for k in ARRAYS};mask=a['available'].copy()
        if event:mask[:,:5]&=a['ongoing'][:,None,None]
        fields['classification_cells']+=int(mask.sum());fields['positive_class_cells']+=int((a['labels']*mask).sum());fields['forecast_speed_flow_cells']+=int(a['available'].sum());fields['current_masked_speed_cells']+=int(a['current_domain'][:,0].sum());fields['current_masked_flow_cells']+=int(a['current_domain'][:,1].sum())
    return fields


def fit_network(arm,seed,initial,data,fit,checkpoint,geometry,graph,device,batches,folder):
    model=JointGraphQueue(rollout=arm=='latent_flow').to(device);model.load_state_dict(initial);tic=time.perf_counter();weight=0. if arm=='direct_no_flow' else .1;history=[];snapshots=[]
    for stage in ['warm','event']:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);losses=[]
        for step,ids in enumerate(batches[stage],1):
            model.train();a=data.dense_batch(ids) if stage=='warm' else {k:fit[k][ids] for k in ARRAYS};x=torch.tensor(a['x'].transpose(0,3,1,2).copy(),device=device);sx=torch.tensor(a['state_x'].transpose(0,3,1,2).copy(),device=device);optimizer.zero_grad(set_to_none=True);out=list(model(x,*graph));out[3]=model.analyze(sx,*graph);loss=joint_dynamics_loss(out,tensors(a,device),weight,event=stage=='event')
            if not torch.isfinite(loss):raise ValueError('Nonfinite joint loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
            if step%200==0:
                history.append({'stage':stage,'step':step,'mean_loss':float(np.mean(losses[-200:]))});print('joint_dynamics',data.case['name'],arm,seed,stage,step,flush=True)
                if stage=='event':
                    pred=inference(model,checkpoint,graph,device);path=folder/(arm+'_'+str(seed)+'_step'+str(step)+'.pt');torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path);op=path.with_suffix('.optimizer.pt');torch.save({'optimizer':optimizer.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device.type=='cuda' else None},op);pp=path.with_suffix('.checkpoint.npz');np.savez_compressed(pp,**pred);snapshots.append({'step':step,'checkpoint_Brier':brier(pred['probability'],checkpoint),**record(path),'state_digest':state_digest(model.state_dict()),'optimizer':record(op),'checkpoint_predictions':record(pp)})
        if stage=='warm':
            path=folder/(arm+'_'+str(seed)+'_before_event.pt');torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path);before_event=record(path)
        del optimizer
    result={'arm':arm,'seed':seed,'rollout':arm=='latent_flow','flow_loss_weight':weight,'initial_state_digest':state_digest(initial),'before_event':before_event,'warm_steps':400,'finetune_steps':800,'history':history,'snapshots':snapshots,'selected':min(snapshots,key=lambda r:r['checkpoint_Brier']),'seconds':time.perf_counter()-tic};del model;return result


def hash_array(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()


def fit_state_trees(data,geometry,folder):
    tic=time.perf_counter();xs=[];ys=[];ds=[];coordinates=[];n_origins=0
    for day,p in enumerate(data.packs):
        for start in range(12,288,48):
            ats=list(range(start,min(288,start+48)));a={k:np.stack([state_window(p,at)[k] for at in ats]) for k in ['state_x','truth_current','current_domain']};features=reconstruction_features(a['state_x'],geometry);use=a['current_domain'].any(axis=1).ravel();xs.append(features[use]);ys.append(a['truth_current'].transpose(0,2,1).reshape(-1,2)[use]);ds.append(a['current_domain'].transpose(0,2,1).reshape(-1,2)[use]);coordinates.append(np.array([(day,at,j) for at in ats for j in range(len(data.links))],np.int32)[use]);n_origins+=len(ats)
    x=np.concatenate(xs);y=np.concatenate(ys);domain=np.concatenate(ds);coords=np.concatenate(coordinates);del xs,ys,ds,coordinates;models=[];records=[];sha={k:hash_array(v) for k,v in [('features',x),('labels',y),('domain',domain),('coordinates',coords)]};cp=folder/'state_tree_training_coordinates.npz';np.savez_compressed(cp,coordinates=coords,truth=y,domain=domain)
    params={'n_estimators':400,'num_leaves':63,'min_child_samples':60,'learning_rate':.03,'reg_lambda':1.,'n_jobs':4,'random_state':20261015,'deterministic':True,'force_col_wise':True,'verbosity':-1}
    for field in range(2):
        model=LGBMRegressor(**params);use=domain[:,field];model.fit(x[use],y[use,field]);path=folder/('causal_state_'+['speed','flow'][field]+'.txt');model.booster_.save_model(str(path));models.append(model.booster_);records.append({'field':['speed','flow'][field],'n_teacher_rows':int(use.sum()),**record(path)})
    return models,{'records':records,'fit_state_origins':n_origins,'n_feature_rows':len(x),'n_features':x.shape[1],'training_arrays_sha256':sha,'coordinates':record(cp),'parameters':params,'seconds':time.perf_counter()-tic,'scope':'Strong causal State comparator uses ALL eligible masked current cells with fully masked past13slots. More unique current labels than neural warm draws;not equal-compute architecture ablation. No Queue future labels or hidden data.'}


def tree_analysis(models,a,geometry):
    x=reconstruction_features(a['state_x'],geometry);values=np.stack([m.predict(x,num_threads=4) for m in models],axis=-1).reshape(len(a['state_x']),len(geometry['geometry'][0]),2).transpose(0,2,1).astype(np.float32).clip(min=0);visible=a['state_x'][:,-1,:,2:4].transpose(0,2,1)>.5
    return np.where(visible,a['state_x'][:,-1,:,:2].transpose(0,2,1),values)


def full_state_assessment(data,training,trees,geometry,graph,device):
    folder=data.folder;seal=sha256_file(folder/'selection_sealed.json');members={}
    for t in training:
        model=JointGraphQueue(rollout=t['rollout']).to(device);model.load_state_dict(torch.load(t['selected']['path'],map_location=device,weights_only=True));model.eval();members[t['arm']+'_'+str(t['seed'])]=model
    records=[]
    for date in data.case['dates']['assessment']:
        pack=data.held_day(date);values={};predictions={};starts=[]
        for start in range(12,288,48):
            ats=list(range(start,min(288,start+48)));rows=[state_window(pack,at) for at in ats];a={k:np.stack([w[k] for w in rows]) for k in ['state_x','truth_current','current_domain','current_regime','current_available']};starts.extend(ats)
            for k in ['truth_current','current_domain','current_regime','current_available']:values.setdefault(k,[]).append(a[k])
            base,_=latest_state(a['state_x'],geometry['geometry']);p={'causal_carry':base,'causal_tree':tree_analysis(trees,a,geometry)}
            with torch.no_grad():
                tensor=torch.tensor(a['state_x'].transpose(0,3,1,2).copy(),device=device)
                for key,model in members.items():p[key]=model.analyze(tensor,*graph).cpu().numpy()
            for arm in ARMS:p[arm]=np.mean([p[arm+'_'+str(seed)] for seed in SEEDS],axis=0)
            for k,v in p.items():predictions.setdefault(k,[]).append(v)
        values={k:np.concatenate(v) for k,v in values.items()};p={k:np.concatenate(v) for k,v in predictions.items()};stats={k:state_sufficient(values,v) for k,v in p.items()};path=folder/('state_assessment_'+date+'.npz');np.savez_compressed(path,**values,**p,origins=pack['timestamps'][starts]);records.append({'date':date,'phase':'assessment_state','n_contexts':len(starts),**record(path),'prior_selection_seal_sha256':seal,'source_raw':record(data.raw_paths[date]),'source_masked':record(data.masked_paths[date]),'models':stats})
    del members;return records,{k:combine_state([r['models'][k] for r in records]) for k in records[0]['models']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=Path('reports/research');report=root/(args.output.name+'.json')
    if report.exists() or args.output.exists():raise FileExistsError('Never overwrite joint dynamics experiment')
    old_path=root/'C87_dense_dynamics_pretraining.json';old=json.loads(old_path.read_text());source_path=root/'C71_rolling_onset_threshold_refits.json';source=json.loads(source_path.read_text());cal_path=root/'C74_probabilistic_onset_sets.json';cal=json.loads(cal_path.read_text());jobs=[c for c in source['cases'] if c['name'] in [o['name'] for o in old['cases']]]
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Joint_continuous_flow_supervision_and_residual_latent_propagation','previous_goal_turn':'progress:13a982c C87-C90 matched dense pretrain,neither primary promoted','source_dense_study_sha256':sha256_file(old_path),'source_event_metadata_sha256':sha256_file(source_path),'source_CP_metadata_sha256':sha256_file(cal_path),'jobs':[{'name':c['name'],'dates':c['dates']} for c in jobs],
        'arms':ARMS,'optimizer_seeds':SEEDS,'shared_data_updates':'SameC87 dense400x8 warm and800x4event batches peroptimizerseed. All arms share oldencoder initialstates and identical additional zero-output heads/GRU initialstates. ResetAdamW before event. Extra latent rollout uses more computation;recordactualtimes,not equalFLOPs claim.',
        'architecture':'Old13slot9channel graphencoder + explicit6flow and2currentanalysis heads. latent_flow adds6step sharedGRU hidden graph propagation and predicted(v,q) feedback as residual correction to common direct heads. All allocated params identical;unused GRU in direct arms. Learned latent state,not CTM/conservation/hidden-boundary physics.',
        'loss':'Original warm/event speed+queue loss,plus.1maskedcurrent speed L1 inALLarms. Flowloss0indirect_no_flow,.1*(futureflowL1+maskedcurrentflowL1) inother2. Speednorm25,flow/lane600.',
        'separate_input_contract':'Queue uses lawfulrawpast12+maskedT;State currentanalysis usesONLYmaskedpast13slots with condition derivedONLYfrommaskedhistory. Separate encoder passes/sharedweights;rawpast heldout values never enter State inputs. Raw current/future targets quality loss/evaluator only.',
        'selection':'Same15day CP mincondition-windowBrier at200/400/600/800;noState-basedweightselection. Eacharm fixed.75strongtree+.25two-seedNN,decisionthresholdgrid;later15daygain>=.01and7daypairedCIlower>0,then30dayassessment. No posthoc threshold/alpha choices.',
        'state_evaluation':'Afterselectionseal,ALL276within-daycausalcurrentorigins12..287/day for30assessmentdays. Both-fields-missing,raweligible,publicR1/R2/R3. Firsthour omitted;no future maskedcontext. Causal carry+fixed400tree fullfitmaskedcurrent comparator;not equal labelpopulation/compute toNN. Perregimepointformula andblockuncertainty diagnostics,not officialState.',
        'forecast_strata':'Allfutureeligible,teacherqueue,>=20km/hdrop/recovery relative toeligible rawT;teacher-onlyevaluator masks,not futureinputs/knownincidents.',
        'scope':'Previouslyviewedpublicnoisydevelopmentdates;not blind/final/private.12newNN+4stateboosters. No promotion based on assessment,no fullsubmission/formalscores/platform actions;finalsimseedsunopened.'})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');tic=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    release=Release('data/kaggle_public');prepared=[];descriptions=[]
    for case in jobs:
        folder=args.output/case['name'];folder.mkdir();oc=next(c for c in old['cases'] if c['name']==case['name']);data=JointDataset(release,case,next(c for c in cal['cases'] if c['name']==case['name']),folder);fit=data.prepare_fit();cp=data.load_event('checkpoint',4129);geometry=dict(np.load(oc['geometry']['path']));gp=folder/'geometry.npz';np.savez_compressed(gp,**geometry)
        for new,previous in zip(data.day_records,oc['dataset_description']['day_records']):
            with np.load(new['path']) as a,np.load(previous['path']) as b:
                if any(not np.array_equal(a[k],b[k]) for k in b.files):raise ValueError('Jointfit changed prior verified dense inputs')
        description={'name':case['name'],'panel':case['panel'],'dates':case['dates'],'day_records':data.day_records,'origin_record':data.origin_record,'event_records':list(data.event_records),'source_files_sha256':dict(data.source_files),'geometry':record(gp)};save(folder/'dataset_sealed.json',description);prepared.append((data,fit,cp,geometry,oc));descriptions.append(description)
    save(args.output/'all_datasets_sealed.json',{'datasets':descriptions});cases=[]
    for data,fit,cp,geometry,oc in prepared:
        graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']];training=[];initials=[];batch_records=[];exposures=[];folder=data.folder;trees,tree_record=fit_state_trees(data,geometry,folder);print('joint_state_trees_COMPLETE',data.case['name'],'rows',tree_record['n_feature_rows'],flush=True)
        for seed in SEEDS:
            ir=next(r for r in oc['initial_records'] if r['seed']==seed);br=next(r for r in oc['batch_records'] if r['seed']==seed)
            if sha256_file(ir['path'])!=ir['sha256'] or sha256_file(br['path'])!=br['sha256']:raise ValueError('Matched initial/batch source changed')
            torch.manual_seed(seed+101);base=JointGraphQueue();base.load_state_dict(torch.load(ir['path'],map_location='cpu',weights_only=True),strict=False);initial={k:v.detach().clone() for k,v in base.state_dict().items()};del base;ip=folder/('common_joint_initial_'+str(seed)+'.pt');torch.save(initial,ip);initials.append({'seed':seed,**record(ip),'state_digest':state_digest(initial),'source_old_initial':ir})
            seq=dict(np.load(br['path']));batches={'warm':seq['dense_warm'],'event':seq['finetune']};bp=folder/('matched_batches_'+str(seed)+'.npz');np.savez_compressed(bp,**batches);batch_records.append({'seed':seed,**record(bp),'source_old_batches':br});exposures.append({'seed':seed,'warm_available':exposure(data,batches['warm'],True,False),'event_available':exposure(fit,batches['event'],False,True),'flow_optimized':'direct_no_flow0;otherarmsallavailableflowcells'})
            for arm in ARMS:training.append(fit_network(arm,seed,initial,data,fit,cp,geometry,graph,device,batches,folder))
        save(folder/'weights_sealed.json',{'training':training,'initials':initials,'batches':batch_records,'exposures':exposures,'state_trees':tree_record});reference=QueueModel.load(Path(data.case['exports']['incumbent']['path']));decision=data.load_event('decision',4133);feature_arm=data.case['feature_arm'];pred_records=[]
        def tree_probability(phase,seed):
            r=next(r for r in data.case['cache_records'] if r['phase']==phase and r['seed']==seed);f=next(f for f in r['feature_files'] if Path(f['path']).stem.endswith('_'+feature_arm))
            if sha256_file(f['path'])!=f['sha256']:raise ValueError('Frozen Queue tree feature cache changed')
            return reference.baseline.probability(pd.read_parquet(f['path']))
        def neural(a):
            means={};members={}
            for arm in ARMS:
                rows=[]
                for t in training:
                    if t['arm']!=arm:continue
                    model=JointGraphQueue(rollout=t['rollout']).to(device);model.load_state_dict(torch.load(t['selected']['path'],map_location=device,weights_only=True));r=inference(model,a,graph,device);rows.append(r);members.update({arm+'_'+str(t['seed'])+'_'+k:v for k,v in r.items()});del model
                means[arm]={k:np.mean([r[k] for r in rows],axis=0) for k in rows[0]}
            return means,members
        def score(a,p,s):
            binary=full_queue_decisions(p,a['window'],a['condition'],a['lead_minutes'],s['thresholds'],s['require_nonempty'],s['timing']);return binary,score_full_forecast(a['labels'].ravel(),binary,a['window'],a['condition'],a['available'].ravel())
        def evaluate(a,phase,seed,choices,selected=None):
            tp=tree_probability(phase,seed);nn,members=neural(a);metrics={};values={'tree_probability':tp,**members}
            for role in ['incumbent']+ARMS:
                values[role],metrics[role]=score(a,tp if role=='incumbent' else .75*tp+.25*nn[role]['probability'].ravel(),choices[role]['setting'])
            if selected is not None:values['selected']=values[selected];metrics['selected']=metrics[selected]
            values.update({arm+'_'+k:v for arm,r in nn.items() for k,v in r.items()});path=folder/(phase+'_'+str(seed)+'_predictions.npz');np.savez_compressed(path,**values);pred_records.append({'phase':phase,'seed':seed,**record(path)})
            return {'seed':seed,'models':metrics,'neural_forecast_diagnostics':{arm:forecast_sufficient(a,r['speed'],r['flow']) for arm,r in nn.items()}}
        tp=tree_probability('decision',4133);nn,_=neural(decision);choices={'incumbent':{'setting':reference.setting,'metrics':score(decision,tp,reference.setting)[1]}}
        for arm in ARMS:
            p=.75*tp+.25*nn[arm]['probability'].ravel();setting=tune_full_probabilities(p,decision['window'],decision['condition'],decision['lead_minutes'],decision['labels'].ravel(),decision['available'].ravel())['selected'];choices[arm]={'setting':setting,'metrics':score(decision,p,setting)[1]}
        challenger=max(['incumbent']+ARMS,key=lambda k:choices[k]['metrics']['proxy_IoU']);save(folder/'decision_sealed.json',{'choices':choices,'challenger':challenger});evaluate(decision,'decision',4133,choices)
        selection={'dates':{'assessment':data.case['dates']['selection']},'assessment':[evaluate(data.load_event('selection',seed),'selection',seed,choices) for seed in EVENT_SEEDS]};gate=comparison(selection,'incumbent',challenger);selected=challenger if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent';save(folder/'selection_sealed.json',{'selected':selected,'decision_selected':challenger,'gate':gate});exports={};portable={'incumbent':reference}
        for arm in ARMS:portable[arm]=JointDynamicsQueueModel(reference.baseline,data.links,choices[arm]['setting'],.25,[torch.load(t['selected']['path'],map_location='cpu',weights_only=True) for t in training if t['arm']==arm],geometry,arm=='latent_flow')
        portable['decision_selected']=portable[challenger];portable['selected']=portable[selected]
        for role,model in portable.items():
            path=folder/(role+'.pkl');model.save(path);exports[role]={**record(path),'candidate':challenger if role=='decision_selected' else selected if role=='selected' else role}
        assessment={'dates':{'assessment':data.case['dates']['assessment']},'assessment':[evaluate(data.load_event('assessment',seed),'assessment',seed,choices,selected) for seed in EVENT_SEEDS]};state_records,state_metrics=full_state_assessment(data,training,trees,geometry,graph,device)
        case={'name':data.case['name'],'panel':data.panel,'dates':data.case['dates'],'dataset_description':descriptions[len(cases)],'geometry':record(folder/'geometry.npz'),'initial_records':initials,'batch_records':batch_records,'actual_exposures':exposures,'training':training,'state_trees':tree_record,'choices':choices,'decision_selected':challenger,'selected':selected,'selection_gate':gate,'selection':selection['assessment'],'assessment':assessment['assessment'],'comparisons':{arm:comparison(assessment,'incumbent',arm) for arm in ARMS+['selected']},'state_assessment_records':state_records,'state_assessment_metrics':state_metrics,'event_records':data.event_records,'prediction_records':pred_records,'phase_reads':data.phase_reads,'source_files_sha256':data.source_files,'exports':exports,'seals':{k:sha256_file(folder/(k+'_sealed.json')) for k in ['dataset','weights','decision','selection']}};save(folder/'case_completed.json',case);cases.append(case);print('joint_dynamics_case_COMPLETE',data.case['name'],'decision',challenger,'selected',selected,flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'new_network_fits':12,'new_State_tree_fits':4,'seconds':time.perf_counter()-tic,'source':evidence(gpu_used=device.type=='cuda'),'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,'formal_State':None,'formal_Queue':None,'formal_Physics':None,'formal_Total':None,'formal_model_promoted':False,'new_complete_submission':False,'scope':'Matched three-armpublicnoisydevelopment study;propermaskedState inputs and lawfulQueue history differ deliberately. No formal/private/final score,promotion or platform actions.'};save(report,result);save(args.output/'experiment.json',result);print('joint_dynamics_COMPLETE','seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
