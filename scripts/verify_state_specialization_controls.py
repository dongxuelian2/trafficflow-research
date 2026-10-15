"""Independent State exposure/selection replay, plus one NN and tree-pair refit."""
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
from trafficflow.joint_graph_queue import JointGraphQueue
from trafficflow.state_control_model import StateControlModel
from trafficflow.joint_queue_data import reconstruction_features
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol
from verify_joint_dynamics_data import reference_day,reference_state,reference_joint
from verify_joint_dynamics_models import reference_features,ref_sufficient,ref_combined,same_stats
from verify_dense_pretrain_models import digest
from dense_pretrain_queue_experiment import save,record
from joint_dynamics_queue_experiment import hash_array


SEEDS=[20261013,20261029];NN_ROLES=['joint_queue_cp','joint_state_event_cp','joint_state_full_cp','state_only_event_cp','state_only'];TREES=['matched_l2','matched_l2_weighted','matched_l1_weighted'];PRIMARY=['joint_state_full_cp','state_only']+TREES;FIELDS=['state_x','truth_current','current_domain','current_regime','current_available']


def independent_schedule(old,seed,dates,origins,event_origins):
    source=next(r for r in old['batch_records'] if r['seed']==seed);b=dict(np.load(source['path']));events=[]
    for text in event_origins:
        t=pd.Timestamp(str(text));events.append([dates.index(t.date().isoformat()),t.hour*12+t.minute//5])
    return {'warm':np.asarray(origins,np.int32)[b['warm']],'event':np.asarray(events,np.int32)[b['event']]}


def independent_teacher(schedule,packs,geometry):
    table={}
    for col,stage in enumerate(['warm','event']):
        for batch in schedule[stage]:
            for coordinate in batch:
                key=tuple(map(int,coordinate));table.setdefault(key,[0,0])[col]+=1
    coordinates=np.array(sorted(table),np.int32);counts=np.array([table[tuple(c)] for c in coordinates],np.int64);mass=counts[:,0]/8.+counts[:,1]/4.;rows=[reference_state(packs[int(day)],int(at)) for day,at in coordinates];a={k:np.stack([r[k] for r in rows]) for k in FIELDS};canonical=reconstruction_features(a['state_x'],geometry);independent,_=reference_features(a['state_x'],geometry);maximum=float(np.abs(canonical-independent).max())
    if maximum>2e-6:raise ValueError('Independent teacher feature semantics differ')
    nominal=np.zeros(a['current_domain'].shape,float)
    for i,domain in enumerate(a['current_domain']):
        for field in range(2):
            n=int(domain[field].sum())
            if n:nominal[i,field,domain[field]]=mass[i]/n
    normalized=nominal.copy()
    for field in range(2):
        selected=a['current_domain'][:,field]
        if selected.any():normalized[:,field][selected]/=nominal[:,field][selected].mean()
    use=a['current_domain'].any(axis=1).ravel();row_coords=np.array([(int(day),int(at),j) for day,at in coordinates for j in range(a['state_x'].shape[2])],np.int32)[use]
    arrays={'features':canonical[use],'truth':a['truth_current'].transpose(0,2,1).reshape(-1,2)[use],'domain':a['current_domain'].transpose(0,2,1).reshape(-1,2)[use],'coordinates':row_coords,'normalized_weights':normalized.transpose(0,2,1).reshape(-1,2)[use],'nominal_weights':nominal.transpose(0,2,1).reshape(-1,2)[use]}
    extra={'context_coordinates':coordinates,'context_stage_counts':counts,'context_mass':mass};return arrays,extra,maximum


def mse(a,p):
    values=[]
    for field,w in enumerate([.54,.46]):
        mask=a['current_domain'][:,field];error=p[:,field].astype(float)-a['truth_current'][:,field].astype(float)
        if not mask.any():return None
        v=error[mask];values.append(w*float(np.dot(v,v))/len(v))
    return float(sum(values))


def run_model(state,a,geometry,device,batch,full_precision=False):
    model=JointGraphQueue().to(device);model.load_state_dict(state);model.eval();graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']];values=[]
    with torch.no_grad(),torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,benchmark=False,deterministic=torch.backends.cudnn.deterministic,allow_tf32=not full_precision):
        for start in range(0,len(a['state_x']),batch):values.append(model.analyze(torch.tensor(a['state_x'][start:start+batch].transpose(0,3,1,2).copy(),device=device),*graph).cpu().numpy())
    return np.concatenate(values)


def research_predictor(model,x):
    """Replay original C95 TF32-enabled inference, preserving frozen arrays."""
    if model.states is None:return model.predict_tensor(x)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not hasattr(model,'_research_networks'):
        model._research_networks=[]
        for state in model.states:
            net=JointGraphQueue().to(device);net.load_state_dict(state);net.eval();model._research_networks.append(net)
    graph=[torch.tensor(model.geometry[k],device=device) for k in ['incoming','outgoing','geometry']]
    with torch.no_grad(),torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,benchmark=False,deterministic=torch.backends.cudnn.deterministic,allow_tf32=True):
        tensor=torch.tensor(x.transpose(0,3,1,2).copy(),device=device);return np.mean([net.analyze(tensor,*graph).cpu().numpy() for net in model._research_networks],axis=0)


def same_prediction(actual,wanted,maximum):
    error=float(np.abs(actual-wanted).max());maximum['prediction']=max(maximum['prediction'],error)
    if error>1e-5:raise ValueError('Independent State prediction differs')


def independent_current_loss(prediction,truth,mask):
    error=(prediction-truth).abs();count=mask.sum(dim=2).clamp_min(1);field=(error*mask).sum(dim=2)/count
    return .1*field[:,0].mean()+.1*field[:,1].mean()


def refit_one_network(initial,schedule,packs,geometry,training,device,folder):
    model=JointGraphQueue().to(device);model.load_state_dict(initial);graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']];maximum=0.;snapshots=[]
    for stage in ['warm','event']:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        for step,coordinates in enumerate(schedule[stage],1):
            model.train();rows=[reference_state(packs[int(day)],int(at)) for day,at in coordinates];x=np.stack([r['state_x'] for r in rows]);truth=np.stack([r['truth_current'] for r in rows]);mask=np.stack([r['current_domain'] for r in rows]);optimizer.zero_grad(set_to_none=True);out=model.analyze(torch.tensor(x.transpose(0,3,1,2).copy(),device=device),*graph);loss=independent_current_loss(out,torch.tensor(truth,device=device),torch.tensor(mask,device=device));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if stage=='event' and step%200==0:
                source=next(r for r in training['snapshots'] if r['step']==step);expected=torch.load(source['path'],map_location='cpu',weights_only=True);state={k:v.detach().cpu() for k,v in model.state_dict().items()};difference=max(float((state[k]-expected[k]).abs().max()) for k in state);maximum=max(maximum,difference)
                if difference>1e-6:raise ValueError('Independent State-only optimization differs')
                path=folder/('independent_State_refit_step'+str(step)+'.pt');torch.save(state,path);snapshots.append({'step':step,**record(path),'state_digest':digest(state),'maximum_tensor_difference':difference})
    return {'seed':training['seed'],'snapshot_records':snapshots,'maximum_weight_difference':maximum,'scope':'OnepredeclaredI5SfirstseedNNindependentlyreoptimizedusingreferenceStateinput/teacherassembly andseparateanalyticcurrentloss. Other3NNonlyinference/exposure audited. RNG archive bytes not compared;traininghasnostochasticlayers.'}


def gradient_check(rows,schedule,packs,geometry,device):
    graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']];maximum=0.
    for row in rows:
        stage=row['stage'];model=JointGraphQueue().to(device);model.load_state_dict(torch.load(row['weight_record']['path'],map_location=device,weights_only=True));model.eval();parameters=[p for name,p in model.named_parameters() if name.startswith(('input.','geometry.','blocks.'))];windows=[]
        for day,at in schedule[stage][row['batch_index']]:
            p=packs[int(day)];at=int(at);ongoing=bool((p['truth_queue'][at-12:at].sum(axis=0)>=2).any());windows.append(reference_joint(p,at,ongoing))
        a={k:np.stack([w[k] for w in windows]) for k in ['x','state_x','labels','truth_speed','truth_flow','available','ongoing','truth_current','current_domain']};t={k:torch.tensor(v,device=device) for k,v in a.items() if k not in ['x','state_x']};x=torch.tensor(a['x'].transpose(0,3,1,2).copy(),device=device);sx=torch.tensor(a['state_x'].transpose(0,3,1,2).copy(),device=device);logits,v,q,_=model(x,*graph);state=model.analyze(sx,*graph);current=independent_current_loss(state,t['truth_current'],t['current_domain']);domain=t['available'].clone()
        if stage=='event':domain[:,:5]&=t['ongoing'][:,None,None]
        bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,t['labels'],reduction='none',pos_weight=torch.tensor(4.,device=device));classification=(bce*domain).sum(dim=(1,2))/domain.sum(dim=(1,2)).clamp_min(1);speed=((v-t['truth_speed']).abs()*t['available']).sum(dim=(1,2))/t['available'].sum(dim=(1,2)).clamp_min(1);flow=((q-t['truth_flow']).abs()*t['available']).sum(dim=(1,2))/t['available'].sum(dim=(1,2)).clamp_min(1);aux=(classification+.1*speed).mean()+.1*flow.mean();g1=torch.cat([v.reshape(-1) for v in torch.autograd.grad(current,parameters,retain_graph=True)]).detach().cpu().numpy().astype(float);g2=torch.cat([v.reshape(-1) for v in torch.autograd.grad(aux,parameters)]).detach().cpu().numpy().astype(float);n1=np.linalg.norm(g1);n2=np.linalg.norm(g2);cos=float(np.dot(g1,g2)/n1/n2) if n1>0 and n2>0 else None
        if cos is None or row['cosine'] is None:
            if cos!=row['cosine']:raise ValueError('Zero-norm gradient status differs')
        else:
            maximum=max(maximum,abs(cos-row['cosine']))
            if abs(cos-row['cosine'])>1e-6:raise ValueError('Independent task-gradient cosine differs')
        if not np.allclose([n1,n2],[row['State_gradient_norm'],row['auxiliary_gradient_norm']],rtol=1e-5,atol=1e-8):raise ValueError('Independent task-gradient norms differ')
    return maximum


def metrics_from_stats(records):
    result={}
    for role in records[0]['models']:
        details,point=ref_combined([r['models'][role] for r in records]);result[role]={'details':details,'point':point}
    return result


def independent_screen(cases):
    roles=['full_tree']+PRIMARY;points={role:float(np.mean([c['verified_phase_metrics']['decision'][role]['point'] for c in cases])) for role in roles};winner=max(roles,key=lambda k:points[k]);candidate=winner if winner!='full_tree' else max(PRIMARY,key=lambda k:points[k]);gains=[c['verified_phase_metrics']['decision'][candidate]['point']-c['verified_phase_metrics']['decision']['full_tree']['point'] for c in cases];r3=[c['verified_phase_metrics']['decision'][candidate]['details'][2]['public_subset_state_formula']-c['verified_phase_metrics']['decision']['full_tree']['details'][2]['public_subset_state_formula'] for c in cases];halves=[]
    for sl in [slice(0,15),slice(15,30)]:
        v={role:float(np.mean([ref_combined([r['models'][role] for r in c['verified_phases']['decision'][sl]])[1] for c in cases])) for role in ['full_tree',candidate]};halves.append(v[candidate]-v['full_tree'])
    gain=points[candidate]-points['full_tree'];shortlisted=winner!='full_tree' and gain>=.002 and min(gains)>=0 and min(r3)>=-.002 and min(halves)>0
    return {'points':points,'decision_best':winner,'challenger':candidate,'gain':gain,'case_gains':gains,'R3_case_gains':r3,'half_calendar_gains':halves,'shortlisted':bool(shortlisted)}


def independent_selection_interval(cases,candidate):
    n=len(cases[0]['verified_phases']['selection']);rng=np.random.default_rng(20261015);starts=rng.integers(0,n,(4000,int(np.ceil(n/7))));ix=((starts[:,:,None]+np.arange(7))%n).reshape(4000,-1)[:,:n];values=[]
    for indices in ix:
        gains=[]
        for case in cases:
            selected=[case['verified_phases']['selection'][int(i)] for i in indices];a=ref_combined([r['models']['full_tree'] for r in selected])[1];b=ref_combined([r['models'][candidate] for r in selected])[1]
            if a is None or b is None:break
            gains.append(b-a)
        if len(gains)==len(cases):values.append(float(np.mean(gains)))
    point=float(np.mean([c['verified_phase_metrics']['selection'][candidate]['point']-c['verified_phase_metrics']['selection']['full_tree']['point'] for c in cases]));return {'gain':point,'ci95':np.quantile(values,[.025,.975]).tolist() if values else None,'valid_resamples':len(values)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite State control verification')
    root=Path('reports/research');study_path=root/'C95_state_specialization_controls.json';study=json.loads(study_path.read_text());prior_path=root/'C93b_joint_dynamics_model_verification.json';prior=json.loads(prior_path.read_text());old_path=root/'C91_joint_flow_latent_dynamics.json';old_study=json.loads(old_path.read_text());study_folder=Path('artifacts/research')/study_path.stem
    if study['registered']['protocol']['source_joint_study_sha256']!=sha256_file(old_path) or study['registered']['protocol']['source_joint_audit_sha256']!=sha256_file(prior_path):raise ValueError('State source proof binding differs')
    for name in ['src/trafficflow/joint_queue_data.py','src/trafficflow/joint_graph_queue.py']:
        expected=next(r['sha256'] for r in prior['source_snapshot']['captured_files'] if r['path']==name);actual=next(r['sha256'] for r in study['source_snapshot']['captured_files'] if r['path']==name)
        if expected!=actual:raise ValueError('Reused fullFIT data/architecture proof does not bind unchanged source')
    sources={}
    for case in study['cases']:sources.update(case['source_files_sha256'])
    for path,d in sources.items():
        if sha256_file(path)!=d:raise ValueError('Public control input changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_State_selection_exposure_training_and_screen','source_study_sha256':sha256_file(study_path),'source_joint_proof_sha256':sha256_file(prior_path),
        'numerical_profiles':'OriginalC95TF32-enabledarrays replayedseparatelyfromcorrectedfullFP32portableprofile. Original1e-5predictionchecksunchanged. FullCPcheckpointchoicesandglobalfamilygatemustremainunchanged;fullFP32newphasearrayssavedseparately. PriorC96failure retained,notdeclaredpassed.',
        'scope':'Reuse originalfullFITmaskedStateproof;independentcontext schedules/nominalweights/allmatchedtreefeatures,currentCPcriteria,allnewphaseinputs/predictions/Stateformulas/gradientpairs/portableAPIs. OneI5SfirstseedNN anditsweightedL2treepair independentlyretrained. OtherNN/treesnotindependentlyreoptimized. No formal/private score.'})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');tic=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=True;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');release=Release('data/kaggle_public');totals={k:0 for k in ['schedules','matched_teacher_rows','matched_tree_field_models','new_networks','new_NN_snapshots','old_joint_snapshot_CP_records','full_CP_contexts','event_CP_contexts','CP_member_forwards','CP_full_FP32_member_forwards','phase_contexts','phase_days','phase_member_role_contexts','phase_unique_full_FP32_NN_contexts','gradient_pairs','portable_history_windows','portable_role_calls','portable_shuffle_calls','portable_timezone_calls','portable_future_rejections','independent_NN_refits','independent_tree_field_refits']};maximum={'prediction':0.,'feature_semantics':0.,'gradient_cosine':0.};portable_maximum={'prediction':0.};profile_change=np.zeros(2);cases=[];refit_record=None;refit_trees=[]
    for case in study['cases']:
        old=next(c for c in old_study['cases'] if c['name']==case['name']);description=case['dataset_description'];packs=[dict(np.load(r['path'])) for r in description['fit_day_records']];geometry=dict(np.load(description['geometry']['path']));cp=dict(np.load(description['checkpoint_full_record']['path']));cp_event={};
        for r in description['fit_day_records']:
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Audited FIT pack changed')
        for r in [description['geometry'],description['checkpoint_full_record'],description['checkpoint_event_record'],description['fit_event_record']]:
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('State dataset archive changed')
        with np.load(description['checkpoint_event_record']['path']) as a:cp_event={k:a[k] for k in FIELDS};cp_event['origins']=a['origins'].copy()
        with np.load(description['fit_event_record']['path']) as a:event_origins=a['origins'].copy()
        dense=np.load(old['dataset_description']['origin_record']['path'])['indices'];panel=case['panel'];net=release.network(panel);links=np.load(description['fit_event_record']['path'])['links'].tolist();raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};masked_paths={partition_date(release,p):p for p in release.files(panel,'train')}
        @lru_cache(maxsize=4)
        def day(date):return reference_day(raw_paths[date],masked_paths[date],links,net)
        if any(case['dates'][k]!=old['dates'][k] for k in old['dates']) or case['dates']['late_transfer']!=pd.date_range('2031-01-15','2031-02-13').strftime('%Y-%m-%d').tolist():raise ValueError('Predeclared State calendars differ')
        expected_full=[];expected_origins=[]
        for date in case['dates']['checkpoint']:
            p=day(date)
            for at in range(12,288):expected_full.append(reference_state(p,at));expected_origins.append(p['timestamps'][at])
        if not np.array_equal(cp['origins'],expected_origins):raise ValueError('Full State CP coverage differs')
        for k in FIELDS:
            expected=np.stack([a[k] for a in expected_full])
            if not np.array_equal(cp[k],expected):raise ValueError('Independent full State CP inputs/teachers differ')
        totals['full_CP_contexts']+=len(cp['origins']);totals['event_CP_contexts']+=len(cp_event['origins']);predictors={};member_paths={};initials={r['seed']:torch.load(r['path'],map_location='cpu',weights_only=True) for r in case['initial_records']}
        for i,text in enumerate(cp_event['origins']):
            t=pd.Timestamp(str(text));at=t.hour*12+t.minute//5;expected=reference_state(day(t.date().isoformat()),at)
            if any(not np.array_equal(expected[k],cp_event[k][i]) for k in FIELDS):raise ValueError('Independent event State CP differs')
        for sr in description['schedule_records']:
            seed=sr['seed'];schedule=independent_schedule(old,seed,case['dates']['fit'],dense,event_origins);saved_schedule=dict(np.load(sr['path']))
            if sha256_file(sr['path'])!=sr['sha256'] or any(not np.array_equal(saved_schedule[k],v) for k,v in schedule.items()):raise ValueError('Matched State schedules differ')
            totals['schedules']+=1;teacher,extra,error=independent_teacher(schedule,packs,geometry);maximum['feature_semantics']=max(maximum['feature_semantics'],error);tr=next(r for r in case['matched_trees'] if r['seed']==seed)
            if {k:hash_array(v) for k,v in teacher.items()}!=tr['arrays_sha256']:raise ValueError('Exactmatchedteacher/weight/feature hashes differ')
            with np.load(tr['teacher_archive']['path']) as a:
                if any(not np.array_equal(a[k],v) for k,v in {**{k:v for k,v in teacher.items() if k!='features'},**extra}.items()):raise ValueError('Independent teacher weights or coordinates differ')
            totals['matched_teacher_rows']+=len(teacher['truth'])
            for arm,tree in tr['trees'].items():
                pair=[]
                for field,r in enumerate(tree['records']):
                    if sha256_file(r['path'])!=r['sha256']:raise ValueError('Matched State tree changed')
                    pair.append(Path(r['path']).read_text());totals['matched_tree_field_models']+=1
                    if case['name']==study['cases'][0]['name'] and seed==SEEDS[0] and arm=='matched_l2_weighted':
                        valid=teacher['domain'][:,field];model=LGBMRegressor(**tree['parameters']);model.fit(teacher['features'][valid],teacher['truth'][valid,field],sample_weight=teacher['normalized_weights'][valid,field]);path=folder/('independent_weighted_tree_'+str(field)+'.txt');model.booster_.save_model(str(path))
                        if path.read_text()!=pair[-1]:raise ValueError('Independent weighted-tree optimization differs')
                        refit_trees.append(record(path));totals['independent_tree_field_refits']+=1
                predictors[arm+'_'+str(seed)]=StateControlModel(links,geometry,trees=[pair]);predictors[arm+'_'+str(seed)]._audit_signature=arm+'_'+str(seed)
            training=next(t for t in case['training'] if t['seed']==seed);precise_new=[]
            if training['initial_state_digest']!=digest(initials[seed]):raise ValueError('State-only initial state differs')
            for snap in training['snapshots']:
                for r in [snap,snap['optimizer'],snap['checkpoint_predictions']]:
                    if sha256_file(r['path'])!=r['sha256']:raise ValueError('State-only snapshot changed')
                state=torch.load(snap['path'],map_location='cpu',weights_only=True);optimizer=torch.load(snap['optimizer']['path'],map_location='cpu',weights_only=True)
                if digest(state)!=snap['state_digest'] or optimizer['step']!=snap['step'] or any(int(v['step'])!=snap['step'] for v in optimizer['optimizer']['state'].values()):raise ValueError('State optimizer/reset step differs')
                if any(not torch.equal(v,initials[seed][key]) for key,v in state.items() if not key.startswith(('input.','geometry.','blocks.','analysis_output.'))):raise ValueError('State-only unexpectedly updated forecasting heads')
                full=run_model(state,cp,geometry,device,48);event=run_model(state,cp_event,geometry,device,48);totals['CP_member_forwards']+=len(full)+len(event)
                with np.load(snap['checkpoint_predictions']['path']) as a:same_prediction(full,a['analysis_full'],maximum);same_prediction(event,a['analysis_event'],maximum)
                if not np.allclose([mse(cp,full),mse(cp_event,event)],[snap['full_current_MSE'],snap['event_current_MSE']],atol=1e-12,rtol=0):raise ValueError('Independent State-only CP criterion differs')
                fp=run_model(state,cp,geometry,device,48,True);ep=run_model(state,cp_event,geometry,device,48,True);totals['CP_full_FP32_member_forwards']+=len(fp)+len(ep);precise_new.append({'record':snap,'full':mse(cp,fp),'event':mse(cp_event,ep)})
                totals['new_NN_snapshots']+=1
            if min(training['snapshots'],key=lambda r:r['full_current_MSE'])!=training['selected_full'] or min(training['snapshots'],key=lambda r:r['event_current_MSE'])!=training['selected_event']:raise ValueError('State-only CP selection differs')
            if refit_record is None:refit_record=refit_one_network(initials[seed],schedule,packs,geometry,training,device,folder);totals['independent_NN_refits']+=1
            if min(precise_new,key=lambda r:r['full'])['record']!=training['selected_full'] or min(precise_new,key=lambda r:r['event'])['record']!=training['selected_event']:raise ValueError('FullFP32 changes a State-only checkpoint choice;report this before deployment')
            totals['new_networks']+=1;joint=next(r for r in case['joint_reselection'] if r['seed']==seed);precise_joint=[]
            for row in joint['snapshots']:
                snap=row['source_snapshot'];state=torch.load(snap['path'],map_location='cpu',weights_only=True);full=run_model(state,cp,geometry,device,48);event=run_model(state,cp_event,geometry,device,1);totals['CP_member_forwards']+=len(full)+len(event)
                with np.load(row['checkpoint_predictions']['path']) as a:same_prediction(full,a['analysis_full'],maximum);same_prediction(event,a['analysis_event'],maximum)
                if not np.allclose([mse(cp,full),mse(cp_event,event)],[row['full_current_MSE'],row['event_current_MSE']],atol=1e-12,rtol=0):raise ValueError('Independent joint State CP criterion differs')
                fp=run_model(state,cp,geometry,device,48,True);ep=run_model(state,cp_event,geometry,device,1,True);totals['CP_full_FP32_member_forwards']+=len(fp)+len(ep);precise_joint.append({'record':snap,'full':mse(cp,fp),'event':mse(cp_event,ep)})
                totals['old_joint_snapshot_CP_records']+=1
            choices={'joint_queue_cp':joint['source_training']['selected'],'joint_state_event_cp':min(joint['snapshots'],key=lambda r:r['event_current_MSE'])['source_snapshot'],'joint_state_full_cp':min(joint['snapshots'],key=lambda r:r['full_current_MSE'])['source_snapshot'],'state_only_event_cp':training['selected_event'],'state_only':training['selected_full']}
            if min(precise_joint,key=lambda r:r['full'])['record']!=choices['joint_state_full_cp'] or min(precise_joint,key=lambda r:r['event'])['record']!=choices['joint_state_event_cp']:raise ValueError('FullFP32 changes a joint State checkpoint choice;report this before deployment')
            if choices!=case['choices'][str(seed)]:raise ValueError('Frozen State checkpoint family choices differ')
            for role,r in choices.items():
                model=StateControlModel(links,geometry,states=[torch.load(r['path'],map_location='cpu',weights_only=True)]);model._audit_signature=r['sha256'];predictors[role+'_'+str(seed)]=model;member_paths[role+'_'+str(seed)]=r
            gradients=next(g for g in case['gradient_diagnostics'] if g['seed']==seed);maximum['gradient_cosine']=max(maximum['gradient_cosine'],gradient_check(gradients['rows'],schedule,packs,geometry,device));totals['gradient_pairs']+=len(gradients['rows'])
        pair=[]
        for r in old['state_trees']['records']:
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Full strong State tree changed')
            pair.append(Path(r['path']).read_text())
        predictors['full_tree']=StateControlModel(links,geometry,trees=[pair]);predictors['full_tree']._audit_signature='full_tree';portable={}
        for role,r in case['exports'].items():
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('State portable model changed')
            model=StateControlModel.load(r['path']);portable[role]=model
            if model._networks is not None or model._boosters is not None:raise ValueError('Portable predictor retained live inference cache')
            if role in NN_ROLES:
                for state,seed in zip(model.states,SEEDS):
                    expected=predictors[role+'_'+str(seed)].states[0]
                    if any(not torch.equal(state[k],expected[k]) for k in state):raise ValueError('Portable State NN weights differ')
            elif model.trees!=([pair] if role=='full_tree' else [predictors[role+'_'+str(seed)].trees[0] for seed in SEEDS]):raise ValueError('Portable State tree ensemble differs')
        verified={};verified_metrics={};precise_phases={};poisoned=False
        for phase,records in case['phase_records'].items():
            if [r['date'] for r in records]!=case['dates'][phase]:raise ValueError('Complete State phase calendar differs')
            output=[];precise_records=[]
            for ri,r in enumerate(records):
                if sha256_file(r['path'])!=r['sha256']:raise ValueError('State phase predictions changed')
                saved=dict(np.load(r['path']));p=day(r['date']);predictions={};precise_predictions={};teachers={}
                for start in range(12,288,48):
                    ats=list(range(start,min(288,start+48)));rows=[reference_state(p,at) for at in ats];a={k:np.stack([w[k] for w in rows]) for k in FIELDS};_,base=reference_features(a['state_x'],geometry);out={'causal_carry':base};precise_out={'causal_carry':base};cache={};precise_cache={}
                    for role,model in predictors.items():
                        if model._audit_signature not in cache:cache[model._audit_signature]=research_predictor(model,a['state_x'])
                        out[role]=cache[model._audit_signature];totals['phase_member_role_contexts']+=len(rows)
                        if model._audit_signature not in precise_cache:
                            precise_cache[model._audit_signature]=model.predict_tensor(a['state_x']) if model.states is not None else out[role]
                            if model.states is not None:totals['phase_unique_full_FP32_NN_contexts']+=len(rows)
                        precise_out[role]=precise_cache[model._audit_signature]
                    for role in NN_ROLES+TREES:out[role]=np.mean([out[role+'_'+str(seed)] for seed in SEEDS],axis=0)
                    for role in NN_ROLES+TREES:precise_out[role]=np.mean([precise_out[role+'_'+str(seed)] for seed in SEEDS],axis=0)
                    for k,v in out.items():predictions.setdefault(k,[]).append(v)
                    for k,v in precise_out.items():precise_predictions.setdefault(k,[]).append(v)
                    for k in FIELDS[1:]:teachers.setdefault(k,[]).append(a[k])
                    totals['phase_contexts']+=len(rows)
                a={k:np.concatenate(v) for k,v in teachers.items()};out={k:np.concatenate(v) for k,v in predictions.items()};precise_out={k:np.concatenate(v) for k,v in precise_predictions.items()}
                if not np.array_equal(saved['origins'],p['timestamps'][12:]) or any(not np.array_equal(saved[k],v) for k,v in a.items()):raise ValueError('Independent State phase teacher or coverage differs')
                stats={k:ref_sufficient(a,v) for k,v in out.items()}
                for k,v in out.items():same_prediction(v,saved[k],maximum);same_stats(stats[k],r['models'][k])
                for k,v in precise_out.items():profile_change=np.maximum(profile_change,np.abs(v-out[k]).max(axis=(0,2)))
                precise_path=folder/(case['name']+'_'+phase+'_'+r['date']+'_FP32.npz');np.savez_compressed(precise_path,**a,**precise_out,origins=p['timestamps'][12:]);precise_records.append({'date':r['date'],**record(precise_path),'source_TF32_record':r,'models':{k:ref_sufficient(a,v) for k,v in precise_out.items()}})
                output.append({'date':r['date'],'models':stats});totals['phase_days']+=1
                if ri in [0,len(records)-1]:
                    frame=release.read(masked_paths[r['date']]);timestamp=pd.to_datetime(frame.timestamp,utc=True)
                    for at in [12,150,287]:
                        origin=pd.Timestamp(str(p['timestamps'][at]));history=frame.loc[(timestamp>=origin-pd.Timedelta(minutes=60))&(timestamp<=origin)].copy();expected_x=reference_state(p,at)['state_x'];totals['portable_history_windows']+=1
                        for role,model in portable.items():
                            actual=model.predict_current(history,net,origin);same_prediction(actual,precise_out[role][at-12],portable_maximum);totals['portable_role_calls']+=1
                            if at==12:
                                same_prediction(model.predict_current(history.sample(frac=1,random_state=17),net,origin),actual,portable_maximum);totals['portable_shuffle_calls']+=1
                                same_prediction(model.predict_current(history,net,origin.tz_convert('Asia/Shanghai')),actual,portable_maximum);totals['portable_timezone_calls']+=1
                        from trafficflow.state_control_data import causal_state_input
                        if not np.array_equal(causal_state_input(history,links,net,origin),expected_x):raise ValueError('Portable pureCPU masked State input differs')
                        if not poisoned:
                            bad=pd.concat([history,history.iloc[[0]].assign(timestamp=origin+pd.Timedelta(minutes=5))],ignore_index=True)
                            for model in portable.values():
                                try:model.predict_current(bad,net,origin)
                                except ValueError:totals['portable_future_rejections']+=1
                                else:raise ValueError('State predictor accepted future input')
                            poisoned=True
            verified[phase]=output;verified_metrics[phase]=metrics_from_stats(output);precise_phases[phase]={'dates':case['dates'][phase],'records':precise_records,'metrics':metrics_from_stats(precise_records)}
            for role,m in verified_metrics[phase].items():
                expected=case[phase]['metrics'][role]
                if not np.isclose(m['point'],expected['public_subset_regime_macro'],atol=1e-12,rtol=0):raise ValueError('State point formula differs')
        for read in case['phase_reads']:
            prior_phase={'decision':'weights','selection':'decision','assessment':'selection','late_transfer':'selection'}.get(read['phase']);expected=sha256_file(study_folder/(prior_phase+'_sealed.json')) if prior_phase else None
            if read['prior_seal_sha256']!=expected:raise ValueError('State stage read-order seal differs')
        cases.append({'name':case['name'],'verified_phases':verified,'verified_phase_metrics':verified_metrics,'full_FP32_phases':precise_phases});print('State_controls_verified_case',case['name'],flush=True)
    screen=independent_screen(cases);wanted=study['decision_screen']
    precise_screen=independent_screen([{'verified_phases':{k:v['records'] for k,v in c['full_FP32_phases'].items()},'verified_phase_metrics':{k:v['metrics'] for k,v in c['full_FP32_phases'].items()}} for c in cases])
    if any(precise_screen[k]!=screen[k] for k in ['decision_best','challenger','shortlisted']):raise ValueError('FullFP32 changes State family decision;do not silently reuse TF32 policy')
    for k,v in screen.items():
        if isinstance(v,(float,list,dict)):
            if isinstance(v,dict):
                if any(abs(v[j]-wanted[k][j])>1e-12 for j in v):raise ValueError('Independent State decision points differ')
            elif not np.allclose(v,wanted[k],atol=1e-12,rtol=0):raise ValueError('Independent State screen values differ')
        elif v!=wanted[k]:raise ValueError('Independent State screen gate differs')
    if not screen['shortlisted']:
        if study['selection_gate'] is not None or study['selected']!='full_tree' or study['later_phases_read'] or any(set(c['phase_records'])!={'decision'} for c in study['cases']):raise ValueError('Rejected State screen accessed new later labels')
        if (study_folder/'selection_sealed.json').exists():raise ValueError('Rejected screen unexpectedly created later selection seal')
    else:
        interval=independent_selection_interval(cases,screen['challenger']);wanted=study['selection_gate']
        if not np.isclose(interval['gain'],wanted['gain'],atol=1e-12,rtol=0) or interval['valid_resamples']!=wanted['valid_resamples'] or not np.allclose(interval['ci95'],wanted['ci95'],atol=1e-12,rtol=0):raise ValueError('Independent selection block interval differs')
        passed=interval['gain']>=.002 and interval['ci95'] is not None and interval['ci95'][0]>0;expected=screen['challenger'] if passed else 'full_tree'
        if study['selected']!=expected or study['later_phases_read']!=(['assessment','late_transfer'] if passed else []):raise ValueError('Independent State selection stopping policy differs')
    result={'registered':registered,'source_snapshot':snapshot,'cases':[{'name':c['name'],'phase_dates':{phase:[r['date'] for r in rows] for phase,rows in c['verified_phases'].items()},'full_FP32_phases':c['full_FP32_phases'],'verified':True} for c in cases],'totals':totals,'maximum_differences':maximum,'portable_full_FP32_maximum_difference':portable_maximum['prediction'],'TF32_to_full_FP32_normalized_maximum_change_by_field':profile_change.tolist(),'TF32_to_FP32_State_point_change_bound':float(.54*profile_change[0]+.46*profile_change[1]),'full_FP32_checkpoint_choices_unchanged':True,'full_FP32_decision_screen':precise_screen,'independent_NN_refit':refit_record,'independent_tree_refit_records':refit_trees,'independent_decision_screen':screen,'seconds':time.perf_counter()-tic,'source_files_rehashed':len(sources),'formal_State':None,'formal_Queue':None,'formal_Physics':None,'formal_Total':None,
        'scope':'Allnewinputs/currentteachers/schedules/nominalweights/CPselection/phaseforecasts/formulas rechecked;1NNand2fieldtreesrefitted independently. Remaining3NN/22treefieldfits not independentlyreoptimized. Reused C93bfullFITinputs proof and originalQueueCPwinnerexplicitly. No blind/official/private score.'};raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8');print('State_controls_verification_COMPLETE',totals,'seconds',result['seconds'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import sys
        if '--output' in sys.argv:
            failed=Path(sys.argv[sys.argv.index('--output')+1]).with_suffix('.failed.json')
            if not failed.exists():failed.write_text(json.dumps({'status':'failed','traceback':traceback.format_exc(),'formal_Total':None},indent=2),encoding='utf-8')
        raise
