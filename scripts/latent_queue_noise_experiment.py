"""Paired hard/soft/oracle teachers on independently generated CTM Queue.

Known synthetic latent labels are diagnostic only. No competition model is
promoted from these scores; no organizer latent truth is read or estimated.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from lightgbm import LGBMRegressor,Booster
from trafficflow.digital_twin import Scenario,simulate
from trafficflow.ctm_scenarios import CASES
from trafficflow.latent_queue import causal_queue_features,gaussian_queue_targets,latent_queue_decisions,window_ious
from trafficflow.metrics import queue_iou
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence


def build_dataset(folder,phase,seeds,cases,noises,phase_reads,seal=None):
    if phase!='fit' and (seal is None or not Path(seal).is_file()):raise ValueError('Prior choice must be sealed')
    phase_reads.append({'phase':phase,'prior_seal_sha256':sha256_file(seal) if seal else None})
    xs=[];ys={k:[] for k in ['observed','soft','latent','available']};ongoing=[];scene_ids=[];origins=[];records=[];previous={}
    for seed in seeds:
        for case,parameters in cases.items():
            for sigma in noises:
                config=Scenario(seed=seed,days=2,mask_rate=.5,speed_noise=sigma,**parameters);sim=simulate(config)
                latent=sim['latent_speed'];truth=sim['queue_truth'];quality=sim['eligible'];cut=.6*config.free_speed
                digest=hashlib.sha256(latent.tobytes()+quality.tobytes()).hexdigest();key=(seed,case)
                if key in previous and digest!=previous[key]:raise ValueError('Noise counterfactual changed latent state or quality')
                previous[key]=digest;scene=len(records);selected=[]
                for at in range(12,len(latent)-6):
                    past=slice(at-12,at);future=slice(at+1,at+7)
                    if quality[past].mean()<.7:continue
                    condition=bool(((truth[past]&quality[past]).sum(axis=0)>=2).any());target=truth[future]
                    if not target.any() or (not condition and (target[:5].any() or not target[5].any())):continue
                    last=np.full(config.cells,config.free_speed)
                    for j in range(12):last=np.where(quality[past][j],latent[past][j],last)
                    if queue_iou(target,np.broadcast_to(last<=cut,target.shape))>.9:continue
                    # Protected past slots use raw noisy observations; T uses released-like masking.
                    hv=sim['observation_speed'][at-12:at+1].copy();hq=sim['observation_flow'][at-12:at+1].copy();valid=quality[at-12:at+1].copy()
                    hv[-1]=sim['masked_speed'][at];hq[-1]=sim['masked_flow'][at];valid[-1]&=np.isfinite(hv[-1])&np.isfinite(hq[-1])
                    x=causal_queue_features(hv,hq,valid,np.arange(config.cells)*config.length_km,config.free_speed,config.capacity,cut,condition)
                    observed=sim['observation_speed'][future]
                    xs.append(x);ys['observed'].append(observed<=cut);ys['soft'].append(gaussian_queue_targets(observed,cut,sigma));ys['latent'].append(target);ys['available'].append(quality[future])
                    ongoing.append(condition);scene_ids.append(scene);origins.append(at);selected.append(at)
                path=folder/f'{phase}_{scene:03d}_scene.npz'
                np.savez_compressed(path,**{k:sim[k] for k in ['latent_speed','observation_speed','observation_flow','masked_speed','masked_flow','eligible','queue_truth','failure']})
                records.append({'seed':seed,'case':case,'sigma':sigma,'configuration':asdict(config),'scene':scene,'origins':selected,'n_windows':len(selected),
                    'path':path.as_posix(),'sha256':sha256_file(path),'latent_quality_sha256':digest,'max_conservation_error_vehicles':sim['max_conservation_error_vehicles']})
        print('latent_queue_data',phase,seed,'windows',len(xs),flush=True)
    if not xs or set(ongoing)!={False,True}:raise ValueError('Both known-latent proxy conditions required')
    data={'x':np.stack(xs),**{k:np.stack(v) for k,v in ys.items()},'ongoing':np.array(ongoing),'scene':np.array(scene_ids),'at':np.array(origins)}
    path=folder/(phase+'_dataset.npz');np.savez_compressed(path,**data)
    return data,{'phase':phase,'path':path.as_posix(),'sha256':sha256_file(path),'scenes':records,'n_windows':len(xs),
        'conditions':{'onset':int((~data['ongoing']).sum()),'ongoing':int(data['ongoing'].sum())},
        'n_zero_observed_positive_windows':int((~(data['observed']&data['available']).any(axis=(1,2))).sum())}


def score(p,data,setting,contract):
    decisions=latent_queue_decisions(p,data['ongoing'],setting['thresholds'],setting['force']);iou=window_ious(data[contract],decisions,data['available'])
    by_condition={name:float(iou[data['ongoing']==value].mean()) for name,value in [('onset',False),('ongoing',True)]}
    return {'macro':float(np.mean(list(by_condition.values()))),'conditions':by_condition,'window_IoU':iou.tolist()}


def calibrate(p,data,contract):
    rows=[]
    for force in [False,True]:
        for onset in [.1,.25,.4,.55,.7]:
            for ongoing in [.1,.25,.4,.55,.7]:
                setting={'thresholds':{'onset':onset,'ongoing':ongoing},'force':force};metric=score(p,data,setting,contract)
                rows.append({'setting':setting,'score':metric['macro'],'conditions':metric['conditions']})
    selected=max(rows,key=lambda r:(r['score'],r['setting']['thresholds']['onset']==r['setting']['thresholds']['ongoing'],not r['setting']['force'],-sum(abs(t-.5) for t in r['setting']['thresholds'].values())))
    return {'selected':selected,'grid':rows,'label_contract':contract}


def probabilities(data,exports):
    x=data['x'].reshape(-1,data['x'].shape[-1]);shape=data['latent'].shape;out={}
    for arm,members in exports.items():
        values=[]
        for member in members:
            if sha256_file(member['path'])!=member['sha256']:raise ValueError('Frozen model changed')
            values.append(Booster(model_file=member['path']).predict(x,num_threads=4).reshape(shape))
        out[arm]=np.clip(np.mean(values,axis=0),0,1)
    # Fixed causal persistence sees only current/last valid normalized speed and public cutoff.
    out['persistence']=(data['x'][:,:,39]<=data['x'][:,:,56]).reshape(shape).astype(float)
    return out


def scene_summaries(scores,data,record):
    result=[]
    for scene in record['scenes']:
        ids=np.flatnonzero(data['scene']==scene['scene'])
        if not len(ids):continue
        condition_means={name:float(np.array(scores['window_IoU'])[ids[data['ongoing'][ids]==value]].mean())
            for name,value in [('onset',False),('ongoing',True)] if (data['ongoing'][ids]==value).any()}
        result.append({'seed':scene['seed'],'case':scene['case'],'sigma':scene['sigma'],'scene':scene['scene'],'condition_IoU':condition_means,'macro_available_conditions':float(np.mean(list(condition_means.values()))),'n_windows':len(ids)})
    return result


def seed_paired_diagnostic(base,candidate):
    a={(r['seed'],r['case'],r['sigma']):r for r in base};b={(r['seed'],r['case'],r['sigma']):r for r in candidate}
    if set(a)!=set(b):raise ValueError('Paired scenes differ')
    seeds=sorted({key[0] for key in a});delta=[]
    for seed in seeds:delta.append(float(np.mean([b[k]['macro_available_conditions']-a[k]['macro_available_conditions'] for k in a if k[0]==seed])))
    rng=np.random.default_rng(947211);sample=np.array(delta)[rng.integers(0,len(delta),(4000,len(delta)))].mean(axis=1)
    return {'seed_mean_gain':float(np.mean(delta)),'ci95_seed_bootstrap':np.quantile(sample,[.025,.975]).tolist(),'seed_gains':dict(zip(map(str,seeds),delta)),
        'independent_seed_count':len(seeds),'note':'Only four independently generated seeds,shared cases/noise counterfactuals and overlapping origins;coarse diagnostic,not official Queue confidence.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();report=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Never overwrite Queue label-noise experiment')
    cases={k:{a:b for a,b in v.items() if a!='speed_noise'} for k,v in CASES.items()}
    stress={'shifted_up':dict(incident_cell=4,capacity_retained=.4,demand_multiplier=1.05,failure_fraction=.25,free_speed=95.,capacity=6400.,length_km=1.),
        'shifted_down':dict(incident_cell=24,capacity_retained=.65,demand_multiplier=1.25,failure_fraction=.2),
        'high_demand':dict(incident_cell=15,capacity_retained=.6,demand_multiplier=1.4,relative_flow_noise=.08),
        'long_failures':dict(incident_cell=12,capacity_retained=.35,demand_multiplier=1.15,failure_fraction=.35,failure_steps=60)}
    split={'fit':[947101,947107,947119,947129],'calibration':[947137,947149],'assessment':[947161,947167,947177,947183]}
    arms=['observation_hard','observation_soft','latent_supervised'];model_seeds=[20261013,20261029]
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Paired_known_CTM_Queue_hard_soft_latent_teacher_study','seeds':split,'cases':cases,'stress_cases':stress,
        'fit_cal_noise':[1.,3.,8.],'assessment_noise':[0.,3.,8.,16.],'model_seeds':model_seeds,'arms':arms,
        'features':'58 fixed causal features,exact13past/current slots,normalized speed/flow,validity,local summaries,within5km positional context,public condition/cutoff/lead. No scene/cell/time identity,noise sigma or latent input.',
        'sampling':'Own latent selected events;history eligible>=.7,ongoing latent>=2 same cell in prior12slots,horizon queue,onset first atT+30,persistenceIoU<=.9. All overlapping origins,not organizer selector.',
        'history':'Past12slots use eligible raw noisy observations;T uses eligible released-like masking. Future eligible mask only training/scoring.',
        'teachers':'Hard threshold of noisy observations;Gaussian ndtr((cut-observed)/known synthetic sigma) flat-prior soft teacher;known latent only synthetic upper reference. No claim soft teacher is true posterior.',
        'model':'All6 LightGBM cross_entropy,300trees/31leaves/lr.035/min_child40/lambda10,feature_fraction.9/bagging.9/freq1/4threads;identical X/rows/weights for each arm and paired optimizer seed.',
        'weights':'Equal condition/equal window,within-window equal legal supervised cells;early onset steps omitted per public timing. No class-dependent weights.',
        'decision':'Fixed equal two-seed probabilities;50 condition thresholds/force settings fitted separately on observed cal and latent cal. Latent-cal settings diagnostic only.',
        'primary':'Observed-cal soft only if gain>=.005 and each condition loss<=.01,else hard. Latent-trained arm not eligible for public pipeline promotion.',
        'heldout':'Assessment/stress generation only after model and decision seals. Same four assessment seeds across stress/noise;not independent new seeds.',
        'uncertainty':'Equal scene available-condition macros then seed means;paired bootstrap4independent seeds,4000resamples. Overlapping windows never independent.',
        'final_simulator_seeds_generated':False,'formal_Queue':None,'formal_Total':None,'public_model_or_submission_changed':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter();reads=[];records=[]
    fit,record=build_dataset(args.output,'fit',split['fit'],cases,[1.,3.,8.],reads);records.append(record)
    allowed=fit['available'].copy();allowed[~fit['ongoing'],:5]=False;count=allowed.sum(axis=(1,2));weight=np.zeros(allowed.shape,float)
    for condition in [False,True]:
        rows=fit['ongoing']==condition;weight[rows]=allowed[rows]/np.maximum(count[rows,None,None],1)/rows.sum()
    use=allowed.ravel();weights=weight.ravel()[use];weights/=weights.mean();x=fit['x'].reshape(-1,58)[use];exports={};training=[]
    targets={'observation_hard':fit['observed'],'observation_soft':fit['soft'],'latent_supervised':fit['latent']}
    for arm in arms:
        exports[arm]=[]
        for seed in model_seeds:
            model=LGBMRegressor(objective='cross_entropy',n_estimators=300,num_leaves=31,learning_rate=.035,min_child_samples=40,reg_lambda=10.,n_jobs=4,
                random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True,colsample_bytree=.9,subsample=.9,subsample_freq=1)
            tic=time.perf_counter();model.fit(x,targets[arm].ravel()[use].astype(float),sample_weight=weights)
            path=args.output/(arm+'_'+str(seed)+'.txt');model.booster_.save_model(path)
            row={'path':path.as_posix(),'sha256':sha256_file(path),'seed':seed};exports[arm].append(row)
            training.append({'arm':arm,'seed':seed,'seconds':time.perf_counter()-tic,'n_rows':len(x),'n_trees':model.booster_.num_trees(),'model':row})
            print('latent_queue_fit',arm,seed,'rows',len(x),'seconds',training[-1]['seconds'],flush=True)
    fit_evidence=args.output/'fit_exposure.npz';np.savez_compressed(fit_evidence,flat_indices=np.flatnonzero(use),weights=weights)
    model_seal=args.output/'models_sealed.json';model_seal.write_text(json.dumps(exports,indent=2,allow_nan=False),encoding='utf-8')
    del fit,x,targets,weight,weights
    cal,record=build_dataset(args.output,'calibration',split['calibration'],cases,[1.,3.,8.],reads,model_seal);records.append(record);cp=probabilities(cal,exports)
    choices={arm:{contract:calibrate(p,cal,contract) for contract in ['observed','latent']} for arm,p in cp.items() if arm!='persistence'}
    left,right=[choices[a]['observed']['selected'] for a in ['observation_hard','observation_soft']]
    primary='observation_soft' if right['score']-left['score']>=.005 and min(right['conditions'][c]-left['conditions'][c] for c in ['onset','ongoing'])>=-.01 else 'observation_hard'
    selection={'primary':primary,'choices':choices,'calibration_observed_gain_soft_vs_hard':right['score']-left['score']}
    seal=args.output/'selection_sealed.json';seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8')
    ppath=args.output/'calibration_probabilities.npz';np.savez_compressed(ppath,**cp);probability_files=[{'phase':'calibration','path':ppath.as_posix(),'sha256':sha256_file(ppath)}]
    print('latent_queue_PRIMARY_SEALED',primary,flush=True);results={};comparisons={}
    for phase,configs in [('assessment',cases),('stress',stress)]:
        data,record=build_dataset(args.output,phase,split['assessment'],configs,[0.,3.,8.,16.],reads,seal);records.append(record);ps=probabilities(data,exports);scores={}
        for arm,p in ps.items():
            policies={'observed_cal':{'thresholds':{'onset':.5,'ongoing':.5},'force':False}} if arm=='persistence' else {contract+'_cal':choices[arm][contract]['selected']['setting'] for contract in ['observed','latent']}
            scores[arm]={}
            for policy,setting in policies.items():
                scores[arm][policy]={}
                for contract in ['observed','latent']:
                    measured=score(p,data,setting,contract);measured['scenes']=scene_summaries(measured,data,record);scores[arm][policy][contract]=measured
        results[phase]=scores;comparisons[phase]={}
        for arm in ['observation_soft','latent_supervised']:
            comparisons[phase][arm]={contract:seed_paired_diagnostic(scores['observation_hard']['observed_cal'][contract]['scenes'],scores[arm]['observed_cal'][contract]['scenes']) for contract in ['observed','latent']}
        path=args.output/(phase+'_probabilities.npz');np.savez_compressed(path,**ps);probability_files.append({'phase':phase,'path':path.as_posix(),'sha256':sha256_file(path)})
    result={'registered':registered,'source_snapshot':snapshot,'training':training,'exports':exports,'fit_exposure':{'path':fit_evidence.as_posix(),'sha256':sha256_file(fit_evidence)},
        'selection':selection,'selection_seal_sha256':sha256_file(seal),'model_seal_sha256':sha256_file(model_seal),'phase_read_order':reads,
        'datasets':records,'probability_files':probability_files,'assessment':results,'comparisons':comparisons,'seconds':time.perf_counter()-start,'source':evidence(),
        'formal_Queue':None,'formal_Total':None,'limits':'Own CTM law and known synthetic noise/latent labels;no organizer latent truth or public-data model transfer. Flat-prior Gaussian soft teacher is an assumption. Four heldout simulator seeds only;counterfactuals/windows correlated. No production promotion.'}
    raw=json.dumps(result,indent=2,allow_nan=False);report.write_text(raw,encoding='utf-8');(args.output/'experiment.json').write_text(raw,encoding='utf-8')
    print('latent_queue_COMPLETE',primary,'seconds',result['seconds'],'latent_gains',{p:comparisons[p]['observation_soft']['latent']['seed_mean_gain'] for p in comparisons},flush=True)


if __name__=='__main__':main()
