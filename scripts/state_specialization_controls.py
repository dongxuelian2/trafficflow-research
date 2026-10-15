"""Separate State selection, task sharing and actual teacher-population effects."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.joint_queue_data import state_window,prepare_joint_day,reconstruction_features,latest_state,joint_window
from trafficflow.joint_graph_queue import JointGraphQueue
from trafficflow.graph_queue import queue_event_loss
from trafficflow.dense_queue_loss import dense_forecast_loss
from trafficflow.state_control_data import state_coordinates,context_exposure,nominal_teacher_weights,current_weighted_mse
from trafficflow.state_control_model import state_only_loss,StateControlModel
from trafficflow.state_control_eval import paired_state_bootstrap
from trafficflow.joint_dynamics_eval import state_sufficient,combine_state
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from dense_pretrain_queue_experiment import save,record,state_digest
from joint_dynamics_queue_experiment import hash_array


SEEDS=[20261013,20261029];FIELDS=['state_x','truth_current','current_domain','current_regime','current_available'];TREE_ARMS=['matched_l2','matched_l2_weighted','matched_l1_weighted'];NN_ROLES=['joint_queue_cp','joint_state_event_cp','joint_state_full_cp','state_only_event_cp','state_only'];CANDIDATES=['joint_state_full_cp','state_only']+TREE_ARMS


class CaseData:
    def __init__(self,release,old,folder,root_folder):
        for r in [old['geometry'],old['dataset_description']['origin_record'],next(r for r in old['event_records'] if r['phase']=='fit'),next(r for r in old['event_records'] if r['phase']=='checkpoint')]:
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Frozen State source metadata/geometry changed')
        self.release=release;self.old=old;self.folder=folder;self.root_folder=root_folder;self.panel=old['panel'];self.net=release.network(self.panel);self.links=np.load(next(r for r in old['event_records'] if r['phase']=='fit')['path'])['links'].tolist();self.geometry=dict(np.load(old['geometry']['path']));self.packs=[];self.source_files={};self.phase_reads=[];self.held_records=[]
        original_inputs=json.loads(Path('reports/research/C92_joint_dynamics_data_verification.json').read_text())['source_files_sha256']
        for path in (release.root/'corridors'/old['panel']/'network').glob('*.csv'):
            digest=sha256_file(path)
            if original_inputs.get(path.as_posix())!=digest:raise ValueError('Frozen public network changed')
            self.source_files[path.as_posix()]=digest
        self.dates={k:list(v) for k,v in old['dates'].items()};self.dates['late_transfer']=pd.date_range('2031-01-15','2031-02-13').strftime('%Y-%m-%d').tolist();self.raw_paths={partition_date(release,p):p for p in release.files(self.panel,'train',False)};self.masked_paths={partition_date(release,p):p for p in release.files(self.panel,'train')}
        if any(date not in self.raw_paths or date not in self.masked_paths for dates in self.dates.values() for date in dates):raise ValueError('Public TRAIN calendar incomplete')
        for r in old['dataset_description']['day_records']:
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Audited FIT day changed')
            self.packs.append(dict(np.load(r['path'])))
            for key in ['raw','masked']:self.source_files[r[key]['path']]=r[key]['sha256']
        self.origin_record=old['dataset_description']['origin_record'];self.dense_origins=np.load(self.origin_record['path'])['indices'];self.fit_record=next(r for r in old['event_records'] if r['phase']=='fit')
        with np.load(self.fit_record['path']) as a:self.event_origins=a['origins'].copy()
        self.cp_event_record=next(r for r in old['event_records'] if r['phase']=='checkpoint')
        with np.load(self.cp_event_record['path']) as a:self.cp_event={k:a[k] for k in FIELDS};self.cp_event['origins']=a['origins'].copy()

    @lru_cache(maxsize=4)
    def held_day(self,date,phase):
        if date not in self.dates[phase]:raise ValueError('State phase calendar violation')
        prior={'decision':'weights','selection':'decision','assessment':'selection','late_transfer':'selection'}.get(phase);seal=self.root_folder/(prior+'_sealed.json') if prior else None
        if seal and not seal.exists():raise ValueError('Global earlier phase must be sealed before '+phase)
        self.phase_reads.append({'date':date,'phase':phase,'prior_seal_sha256':sha256_file(seal) if seal else None})
        for path in [self.raw_paths[date],self.masked_paths[date]]:self.source_files[path.as_posix()]=sha256_file(path)
        self.held_records.append({'date':date,'phase':phase,'raw':record(self.raw_paths[date]),'masked':record(self.masked_paths[date])})
        return prepare_joint_day(self.release.read(self.raw_paths[date]),self.release.read(self.masked_paths[date]),self.links,self.net)

    def batch(self,coordinates):
        rows=[state_window(self.packs[int(day)],int(at)) for day,at in coordinates];return {k:np.stack([r[k] for r in rows]) for k in FIELDS}

    def checkpoint(self):
        rows=[];origins=[]
        for date in self.dates['checkpoint']:
            p=self.held_day(date,'checkpoint')
            for at in range(12,288):rows.append(state_window(p,at));origins.append(p['timestamps'][at])
        a={k:np.stack([r[k] for r in rows]) for k in FIELDS};a['origins']=np.array(origins);path=self.folder/'full_checkpoint_state.npz';np.savez_compressed(path,**a);self.cp_full_record=record(path);return a


def predict_state(state,a,geometry,device,batch=48):
    model=JointGraphQueue().to(device);model.load_state_dict(state);model.eval();graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']];values=[]
    with torch.no_grad():
        for start in range(0,len(a['state_x']),batch):values.append(model.analyze(torch.tensor(a['state_x'][start:start+batch].transpose(0,3,1,2).copy(),device=device),*graph).cpu().numpy())
    return np.concatenate(values)


def fit_state_only(seed,initial,schedule,data,cp,device):
    model=JointGraphQueue().to(device);model.load_state_dict(initial);graph=[torch.tensor(data.geometry[k],device=device) for k in ['incoming','outgoing','geometry']];snapshots=[];history=[];tic=time.perf_counter()
    for stage in ['warm','event']:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);losses=[]
        for step,coordinates in enumerate(schedule[stage],1):
            model.train();a=data.batch(coordinates);optimizer.zero_grad(set_to_none=True);x=torch.tensor(a['state_x'].transpose(0,3,1,2).copy(),device=device);truth=torch.tensor(a['truth_current'],device=device);domain=torch.tensor(a['current_domain'],device=device);loss=state_only_loss(model.analyze(x,*graph),truth,domain)
            if not torch.isfinite(loss):raise ValueError('Nonfinite State-only loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
            if step%200==0:
                history.append({'stage':stage,'step':step,'mean_current_L1_loss':float(np.mean(losses[-200:]))});print('state_only',data.old['name'],seed,stage,step,flush=True)
                if stage=='event':
                    path=data.folder/('state_only_'+str(seed)+'_step'+str(step)+'.pt');state={k:v.detach().cpu() for k,v in model.state_dict().items()};torch.save(state,path);op=path.with_suffix('.optimizer.pt');torch.save({'step':step,'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device.type=='cuda' else None},op);full=predict_state(state,cp,data.geometry,device);event=predict_state(state,data.cp_event,data.geometry,device);pp=path.with_suffix('.state_CP.npz');np.savez_compressed(pp,analysis_full=full,analysis_event=event);snapshots.append({'step':step,**record(path),'state_digest':state_digest(state),'optimizer':record(op),'checkpoint_predictions':record(pp),'full_current_MSE':current_weighted_mse(cp,full),'event_current_MSE':current_weighted_mse(data.cp_event,event)})
        if stage=='warm':
            path=data.folder/('state_only_'+str(seed)+'_before_event.pt');torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path);warm_record=record(path)
        del optimizer
    return {'seed':seed,'initial_state_digest':state_digest(initial),'snapshots':snapshots,'before_event':warm_record,'selected_full':min(snapshots,key=lambda r:r['full_current_MSE']),'selected_event':min(snapshots,key=lambda r:r['event_current_MSE']),'history':history,'seconds':time.perf_counter()-tic}


def matched_trees(seed,schedule,data):
    tic=time.perf_counter();coordinates,counts,mass=context_exposure(schedule);a=data.batch(coordinates);raw,normalized=nominal_teacher_weights(mass,a['current_domain']);features=reconstruction_features(a['state_x'],data.geometry);use=a['current_domain'].any(axis=1).ravel();x=features[use];y=a['truth_current'].transpose(0,2,1).reshape(-1,2)[use];domain=a['current_domain'].transpose(0,2,1).reshape(-1,2)[use];weights=normalized.transpose(0,2,1).reshape(-1,2)[use];raw_weights=raw.transpose(0,2,1).reshape(-1,2)[use];row_coords=np.concatenate([np.repeat(coordinates,len(data.links),axis=0),np.tile(np.arange(len(data.links)),len(coordinates))[:,None]],axis=1)[use].astype(np.int32)
    path=data.folder/('matched_teacher_'+str(seed)+'.npz');np.savez_compressed(path,coordinates=row_coords,truth=y,domain=domain,normalized_weights=weights,nominal_weights=raw_weights,context_coordinates=coordinates,context_stage_counts=counts,context_mass=mass)
    arrays={'features':x,'truth':y,'domain':domain,'coordinates':row_coords,'normalized_weights':weights,'nominal_weights':raw_weights};hashes={k:hash_array(v) for k,v in arrays.items()};models={};records={}
    for arm in TREE_ARMS:
        pair=[];rows=[];params=dict(data.old['state_trees']['parameters']);params['objective']='regression_l1' if arm=='matched_l1_weighted' else 'regression'
        for field in range(2):
            valid=domain[:,field];model=LGBMRegressor(**params);model.fit(x[valid],y[valid,field],sample_weight=weights[valid,field] if arm!='matched_l2' else None);path=data.folder/(arm+'_'+str(seed)+'_'+['speed','flow'][field]+'.txt');model.booster_.save_model(str(path));pair.append(path.read_text());rows.append({'field':field,'n_teacher_rows':int(valid.sum()),**record(path)})
        models[arm]=pair;records[arm]={'parameters':params,'records':rows}
    return models,{'seed':seed,'n_unique_contexts':len(coordinates),'n_feature_rows':len(x),'n_teacher_rows':domain.sum(axis=0).tolist(),'feature_count':x.shape[1],'teacher_archive':record(data.folder/('matched_teacher_'+str(seed)+'.npz')),'arrays_sha256':hashes,'trees':records,'seconds':time.perf_counter()-tic,
        'scope':'Exactcurrentteacher union from6400originalcontextdraws. Nominalweights=sum1/(batchsize*nmaskedfield) perupdate,normalizedtofieldmean1 forfixedregularization. Doesnot reproduce optimizersequence/reset/clipping. Threefixedtreecontrols separate coverage,weighting andL1/L2.'}


def joint_checkpoint_choices(seed,data,cp,device):
    source=next(t for t in data.old['training'] if t['arm']=='direct_flow' and t['seed']==seed);rows=[]
    for s in source['snapshots']:
        if sha256_file(s['path'])!=s['sha256'] or sha256_file(s['checkpoint_predictions']['path'])!=s['checkpoint_predictions']['sha256']:raise ValueError('Frozen joint checkpoint changed')
        state=torch.load(s['path'],map_location='cpu',weights_only=True);full=predict_state(state,cp,data.geometry,device)
        with np.load(s['checkpoint_predictions']['path']) as old:event=old['analysis'].copy()
        path=data.folder/('joint_'+str(seed)+'_step'+str(s['step'])+'_state_CP.npz');np.savez_compressed(path,analysis_full=full,analysis_event=event);rows.append({'source_snapshot':s,'full_current_MSE':current_weighted_mse(cp,full),'event_current_MSE':current_weighted_mse(data.cp_event,event),'checkpoint_predictions':record(path)})
    choices={'joint_queue_cp':source['selected'],'joint_state_event_cp':min(rows,key=lambda r:r['event_current_MSE'])['source_snapshot'],'joint_state_full_cp':min(rows,key=lambda r:r['full_current_MSE'])['source_snapshot']}
    return choices,{'seed':seed,'source_training':source,'snapshots':rows,'choices':choices}


def gradient_diagnostics(seed,schedule,data,device):
    source=next(t for t in data.old['training'] if t['arm']=='direct_flow' and t['seed']==seed);fit_path=data.fit_record['path'];graph=[torch.tensor(data.geometry[k],device=device) for k in ['incoming','outgoing','geometry']];rows=[]
    with np.load(fit_path) as f:fit={k:f[k] for k in ['x','state_x','labels','truth_speed','truth_flow','available','ongoing','truth_current','current_domain']}
    source_br=next(r for r in data.old['batch_records'] if r['seed']==seed);br=dict(np.load(source_br['path']));old_batches=dict(np.load(source_br['source_old_batches']['path']))
    for stage,state_record in [('warm',source['before_event']),('event',source['snapshots'][-1])]:
        if sha256_file(state_record['path'])!=state_record['sha256']:raise ValueError('Gradient diagnostic weight changed')
        model=JointGraphQueue().to(device);model.load_state_dict(torch.load(state_record['path'],map_location=device,weights_only=True));model.eval();parameters=[p for name,p in model.named_parameters() if name.startswith(('input.','geometry.','blocks.'))]
        for index in range(4):
            if stage=='warm':
                windows=[joint_window(data.packs[int(day)],int(at)) for day,at in schedule['warm'][index]];a={k:np.stack([w[k] for w in windows]) for k in fit}
            else:a={k:v[old_batches['finetune'][index]] for k,v in fit.items()}
            x=torch.tensor(a['x'].transpose(0,3,1,2).copy(),device=device);sx=torch.tensor(a['state_x'].transpose(0,3,1,2).copy(),device=device);t={k:torch.tensor(a[k],device=device) for k in a if k not in ['x','state_x']};logits,v,q,_=model(x,*graph);analysis=model.analyze(sx,*graph);current=state_only_loss(analysis,t['truth_current'],t['current_domain'])
            aux=queue_event_loss(logits,v,t['labels'],t['truth_speed'],t['available'],t['ongoing']) if stage=='event' else dense_forecast_loss(logits,v,t['labels'],t['truth_speed'],t['available']);aux=aux+.1*(((q-t['truth_flow']).abs()*t['available']).sum(dim=(1,2))/t['available'].sum(dim=(1,2)).clamp_min(1)).mean();g1=torch.cat([g.reshape(-1) for g in torch.autograd.grad(current,parameters,retain_graph=True)]).detach().cpu().numpy().astype(float);g2=torch.cat([g.reshape(-1) for g in torch.autograd.grad(aux,parameters)]).detach().cpu().numpy().astype(float);n1=float(np.linalg.norm(g1));n2=float(np.linalg.norm(g2));dot=float(np.dot(g1,g2));rows.append({'stage':stage,'batch_index':index,'weight_record':state_record,'State_gradient_norm':n1,'auxiliary_gradient_norm':n2,'dot':dot,'cosine':dot/n1/n2 if n1>0 and n2>0 else None})
    return {'seed':seed,'rows':rows,'scope':'FIT-onlyposthocsharedencodergradients,first4actualwarm/eventbatches atfixedwarm400/event800weights. Auxiliary=Queue+futurev/q,current=StateL1. Gradientcosine describes localalignment,not proof of causation or heldout improvement.'}


def phase_evaluation(data,phase,predictors):
    records=[]
    for date in data.dates[phase]:
        pack=data.held_day(date,phase);fields={};preds={}
        for start in range(12,288,48):
            ats=list(range(start,min(288,start+48)));rows=[state_window(pack,at) for at in ats];a={k:np.stack([r[k] for r in rows]) for k in FIELDS};out={};base,_=latest_state(a['state_x'],data.geometry['geometry']);out['causal_carry']=base;cache={}
            for role,model in predictors.items():
                signature=model._control_signature
                if signature not in cache:cache[signature]=model.predict_tensor(a['state_x'])
                out[role]=cache[signature]
            for role in NN_ROLES+TREE_ARMS:out[role]=np.mean([out[role+'_'+str(seed)] for seed in SEEDS],axis=0)
            for k,v in out.items():preds.setdefault(k,[]).append(v)
            for k in FIELDS[1:]:fields.setdefault(k,[]).append(a[k])
        fields={k:np.concatenate(v) for k,v in fields.items()};out={k:np.concatenate(v) for k,v in preds.items()};path=data.folder/(phase+'_'+date+'.npz');np.savez_compressed(path,**fields,**out,origins=pack['timestamps'][12:]);records.append({'date':date,'phase':phase,**record(path),'n_contexts':276,'models':{k:state_sufficient(fields,v) for k,v in out.items()}})
    return {'dates':data.dates[phase],'records':records,'metrics':{k:combine_state([r['models'][k] for r in records]) for k in records[0]['models']}}


def decision_screen(cases):
    roles=['full_tree']+CANDIDATES;points={role:float(np.mean([c['decision']['metrics'][role]['public_subset_regime_macro'] for c in cases])) for role in roles};winner=max(roles,key=lambda k:points[k]);candidate=winner if winner!='full_tree' else max(CANDIDATES,key=lambda k:points[k]);gains=[c['decision']['metrics'][candidate]['public_subset_regime_macro']-c['decision']['metrics']['full_tree']['public_subset_regime_macro'] for c in cases];r3=[c['decision']['metrics'][candidate]['details'][2]['public_subset_state_formula']-c['decision']['metrics']['full_tree']['details'][2]['public_subset_state_formula'] for c in cases];halves=[]
    for sl in [slice(0,15),slice(15,30)]:
        value={role:float(np.mean([combine_state([r['models'][role] for r in c['decision']['records'][sl]])['public_subset_regime_macro'] for c in cases])) for role in ['full_tree',candidate]};halves.append(value[candidate]-value['full_tree'])
    gain=points[candidate]-points['full_tree'];shortlisted=bool(winner!='full_tree' and gain>=.002 and min(gains)>=0 and min(r3)>=-.002 and min(halves)>0)
    return {'points':points,'decision_best':winner,'challenger':candidate,'gain':gain,'case_gains':gains,'R3_case_gains':r3,'half_calendar_gains':halves,'shortlisted':shortlisted,'rule':'Globalgain>=.002,bothcasegains>=0,bothR3gains>=-.002,both15dayhalvesgain>0. Ties preferfull_tree. Failure stops all new later labels.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=Path('reports/research');report=root/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Never overwrite State specialization controls')
    source_path=root/'C91_joint_flow_latent_dynamics.json';source=json.loads(source_path.read_text());proof_path=root/'C93b_joint_dynamics_model_verification.json';proof=json.loads(proof_path.read_text())
    if proof['registered']['protocol']['source_study_sha256']!=sha256_file(source_path):raise ValueError('Audited source differs')
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'State_specialization_selection_and_teacher_exposure_controls','previous_goal_turn':'progress:28aa20c C91-C94 jointcontrols completed','source_joint_study_sha256':sha256_file(source_path),'source_joint_audit_sha256':sha256_file(proof_path),'seeds':SEEDS,
        'new_fits':'4StateonlyNN,24matchedtreefieldboosters. Existing4direct_flowfits onlyreselected,not newtraining. No newQueue fit/prediction or complete submission.',
        'State_only':'Sameallparametersinitial/400x8warm+800x4eventcoordinates/AdamW.001wd1e-4clip1/resetasC91direct_flow. Analyze(masked13slots)only,.1currentvL1+.1currentqL1 exactlysourcecurrentcomponent. Future/Queuelossremoved;changesgradientdirection,clippingandcompute,not claimnegativecosinecausalproof.',
        'selection_controls':'OriginaljointQueueBrierselection;sameoldCPeventStateMSEselection;full15dayCPStateMSEselection. NewStateonly event/fullCPMSE variants;primaryfull. FixednormalizedcurrentMSE=.54speedMSE+.46flowperlaneMSE. Perregimepointformulasdiagnostic only. Coverage/criterionchanges separated.',
        'matched_trees':'Exactuniquecurrentteacherunion fromoriginalNN6400contextdraws,129features,same400tree/63leaf/min60/lr.03/lambda1params. L2unweighted/L2weighted/L1weighted. Nominalweight=sum1/(batchsize*nmaskedcurrentfield),normalizefieldmean1. Staticlossmeasure not optimizer-sequence equivalence. FullL2causaltreesfrozenfromC91;perseedandmean comparisons separate ensemble effects.',
        'gradient_diagnostic':'32FIT-onlysharedencodergradientpairs:2cases*2seeds*2fixedstages*first4actualbatches. SharedcurrentState vsQueue+futurev/q,atwarm400/event800. Never heldout gradients,not selectionfeature.',
        'decision_screen':'Full30dayOct01..30State subset276origins/day. Oneglobalfamilyamongjoint_state_full_cp,state_only,3matchedtreevariants versusfull_tree. Gain>=.002,bothcasegains>=0,R3drop<=.002percase,bothcalendarhalvespositive. Tiesbaseline;failuredoNOTreadnewselection/assessment.',
        'selection_if_screen_passes':'Original15dayOct31..Nov14,onefrozenfamily,pairedsynchronized7day4000blockCIgain>=.002lower>0. Failure stops laterlabels.',
        'assessment_only_if_both_pass':'OriginalNov15..Dec14 plus fixedlate_transfer2031Jan15..Feb13 with no retraining. Datespublicdevelopment,not blind final/private. No laterread on rejection.',
        'scope':'PublicnoisycausalState bothfields-missingeligible subset,firsthour omitted;not fullTask1 or formalState/Physics/Total. Goal.950unmet;finalsimseedsunopened;local models/Git only.'})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');tic=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    release=Release('data/kaggle_public');prepared=[];descriptions=[]
    for old in source['cases']:
        folder=args.output/old['name'];folder.mkdir();data=CaseData(release,old,folder,args.output);cp=data.checkpoint();schedules={};schedule_records=[]
        for seed in SEEDS:
            sr=next(r for r in old['batch_records'] if r['seed']==seed)
            if sha256_file(sr['path'])!=sr['sha256']:raise ValueError('Original batch archive changed')
            schedule=state_coordinates(data.dense_origins,data.event_origins,data.dates['fit'],dict(np.load(sr['path'])));p=folder/('State_schedule_'+str(seed)+'.npz');np.savez_compressed(p,**schedule);schedules[seed]=schedule;schedule_records.append({'seed':seed,**record(p),'source_record':sr})
        description={'name':old['name'],'panel':old['panel'],'dates':data.dates,'fit_day_records':old['dataset_description']['day_records'],'fit_event_record':data.fit_record,'checkpoint_event_record':data.cp_event_record,'checkpoint_full_record':data.cp_full_record,'schedule_records':schedule_records,'geometry':old['geometry'],'fit_and_CP_sources':dict(data.source_files),'held_records':list(data.held_records)};save(folder/'dataset_sealed.json',description);descriptions.append(description);prepared.append((data,cp,schedules,description))
    save(args.output/'all_datasets_sealed.json',{'datasets':descriptions});cases=[];all_predictors=[]
    for data,cp,schedules,description in prepared:
        predictors={};training=[];trees=[];joint=[];gradients=[];choices={};exports={};initials=[]
        if any(sha256_file(r['path'])!=r['sha256'] for r in data.old['state_trees']['records']):raise ValueError('Full State reference changed')
        full_pair=[Path(r['path']).read_text() for r in data.old['state_trees']['records']];predictors['full_tree']=StateControlModel(data.links,data.geometry,trees=[full_pair]);predictors['full_tree']._control_signature='full_tree'
        for seed in SEEDS:
            initial_record=next(r for r in data.old['initial_records'] if r['seed']==seed)
            if sha256_file(initial_record['path'])!=initial_record['sha256']:raise ValueError('Sharedinitial changed')
            initial=torch.load(initial_record['path'],map_location='cpu',weights_only=True);initials.append(initial_record);old_choices,joint_record=joint_checkpoint_choices(seed,data,cp,device);joint.append(joint_record);fit=fit_state_only(seed,initial,schedules[seed],data,cp,device);training.append(fit);tree_models,tree_record=matched_trees(seed,schedules[seed],data);trees.append(tree_record);gradients.append(gradient_diagnostics(seed,schedules[seed],data,device))
            chosen={**old_choices,'state_only_event_cp':fit['selected_event'],'state_only':fit['selected_full']};choices[str(seed)]=chosen
            for role,r in chosen.items():
                state=torch.load(r['path'],map_location='cpu',weights_only=True);model=StateControlModel(data.links,data.geometry,states=[state]);model._control_signature=r['sha256'];predictors[role+'_'+str(seed)]=model
            for role,pair in tree_models.items():
                model=StateControlModel(data.links,data.geometry,trees=[pair]);model._control_signature=role+'_'+str(seed);predictors[role+'_'+str(seed)]=model
        for role in ['full_tree']+NN_ROLES+TREE_ARMS:
            model=predictors['full_tree'] if role=='full_tree' else StateControlModel(data.links,data.geometry,states=[predictors[role+'_'+str(seed)].states[0] for seed in SEEDS]) if role in NN_ROLES else StateControlModel(data.links,data.geometry,trees=[predictors[role+'_'+str(seed)].trees[0] for seed in SEEDS]);path=data.folder/(role+'.pkl');model.save(path);exports[role]=record(path)
        case={'name':data.old['name'],'panel':data.panel,'dates':data.dates,'dataset_description':description,'initial_records':initials,'training':training,'matched_trees':trees,'joint_reselection':joint,'gradient_diagnostics':gradients,'choices':choices,'exports':exports,'phase_records':{}};save(data.folder/'weights_sealed.json',case);case['weights_seal_sha256']=sha256_file(data.folder/'weights_sealed.json');cases.append(case);all_predictors.append(predictors);print('State_control_weights_COMPLETE',case['name'],flush=True)
    save(args.output/'weights_sealed.json',{'cases':[{'name':c['name'],'sha256':c['weights_seal_sha256']} for c in cases]})
    for (data,_,_,_),case,predictors in zip(prepared,cases,all_predictors):case['decision']=phase_evaluation(data,'decision',predictors);case['phase_records']['decision']=case['decision']['records']
    screen=decision_screen(cases);save(args.output/'decision_sealed.json',screen);selection_gate=None;selected='full_tree';later_reads=[]
    if screen['shortlisted']:
        for (data,_,_,_),case,predictors in zip(prepared,cases,all_predictors):case['selection']=phase_evaluation(data,'selection',predictors);case['phase_records']['selection']=case['selection']['records']
        selection_gate=paired_state_bootstrap([c['selection'] for c in cases],'full_tree',screen['challenger']);passed=selection_gate['gain'] is not None and selection_gate['gain']>=.002 and selection_gate['ci95'] is not None and selection_gate['ci95'][0]>0;selected=screen['challenger'] if passed else 'full_tree';save(args.output/'selection_sealed.json',{'selected':selected,'gate':selection_gate})
        if passed:
            for phase in ['assessment','late_transfer']:
                for (data,_,_,_),case,predictors in zip(prepared,cases,all_predictors):case[phase]=phase_evaluation(data,phase,predictors);case['phase_records'][phase]=case[phase]['records']
                later_reads.append(phase)
    for (data,_,_,_),case in zip(prepared,cases):case.update(source_files_sha256=data.source_files,phase_reads=data.phase_reads,held_records=data.held_records);save(data.folder/'case_completed.json',case)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'decision_screen':screen,'selection_gate':selection_gate,'selected':selected,'later_phases_read':later_reads,'new_network_fits':4,'new_tree_field_fits':24,'reselected_existing_direct_flow_fits':4,'seconds':time.perf_counter()-tic,'source':evidence(gpu_used=device.type=='cuda'),'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,'formal_State':None,'formal_Queue':None,'formal_Physics':None,'formal_Total':None,'formal_model_promoted':False,'new_complete_submission':False,'scope':'Statecausalpublicdevelopmentfactorcontrols;weakmodelsstopbeforeadditionalheldperiods. NovelNNweightsandmatchedtrees,existingweightsreselectedexplicitly. No hidden/private/latent/formalscore or platform operations.'};save(report,result);save(args.output/'experiment.json',result);print('State_controls_COMPLETE','screen',screen['shortlisted'],'selected',selected,'seconds',result['seconds'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import sys
        if '--output' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output')+1]);failed=Path('reports/research')/(out.name+'.failed.json')
            if not failed.exists():failed.write_text(json.dumps({'status':'failed','traceback':traceback.format_exc(),'formal_Total':None},indent=2),encoding='utf-8')
        raise
