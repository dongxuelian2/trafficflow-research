"""Independent CTM regeneration, row-based features, targets and model replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from lightgbm import Booster
from scipy.special import erfc
from trafficflow.digital_twin import Scenario,simulate
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol


def manual_features(v,q,mask,cfg,ongoing):
    n=cfg['cells'];p=np.arange(n)*cfg['length_km'];locals=[];summary=[]
    for j in range(n):
        valid=mask[:,j];vr=v[:,j]/cfg['free_speed'];qr=q[:,j]/cfg['capacity'];ids=np.flatnonzero(valid);recent=ids[ids>=6]
        def slope(value):
            if len(recent)<2:return 0.
            t=recent.astype(float)-12;z=value[recent];return float(np.dot(t-t.mean(),z-z.mean())/np.dot(t-t.mean(),t-t.mean()))
        current=float(vr[ids[-1]]) if len(ids) else 1.;sv=slope(vr);coverage=len(ids)/13
        mean=float(vr[ids].mean()) if len(ids) else 1.;std=float(np.sqrt(max(float(np.mean(vr[ids]**2))-mean**2,0.))) if len(ids) else 0.
        fields=[current,mean,float(vr[ids].min()) if len(ids) else 1.,std,sv,(12-ids[-1])/12 if len(ids) else 13/12,coverage,
            float(qr[ids[-1]]) if len(ids) else 0.,float(qr[ids].mean()) if len(ids) else 0.,slope(qr)]
        locals.append(list(np.where(valid,vr,0.))+list(np.where(valid,qr,0.))+valid.astype(float).tolist()+fields);summary.append([current,sv,coverage])
    rows=[]
    for j in range(n):
        row=locals[j]
        for direction in [-1,1]:
            ids=[i for i in range(n) if 0<direction*(p[i]-p[j])<=5]
            row+=np.mean(np.array(summary)[ids],axis=0).tolist() if ids else [1.,0.,0.]
        rows.append(row+[float(ongoing),.6])
    return np.concatenate([np.column_stack([rows,np.full(n,lead/30)]) for lead in [5,10,15,20,25,30]]).astype(np.float32)


def decisions(p,ongoing,setting):
    out=np.zeros(p.shape,bool)
    for i,c in enumerate(ongoing):
        threshold=setting['thresholds']['ongoing' if c else 'onset'];out[i]=p[i]>=threshold
        allowed=np.ones(p[i].shape,bool)
        if not c:out[i,:5]=False;allowed[:5]=False
        if setting['force'] and not out[i].any():out[i].ravel()[np.argmax(np.where(allowed,p[i],-np.inf))]=True
    return out


def measure(labels,pred,mask,ongoing):
    values=[]
    for y,p,m in zip(labels,pred,mask):
        union=np.count_nonzero((y|p)&m);values.append(np.count_nonzero(y&p&m)/union if union else float(m.any()))
    values=np.array(values);conditions={name:float(values[np.asarray(ongoing)==c].mean()) for name,c in [('onset',False),('ongoing',True)]}
    return {'macro':float(np.mean(list(conditions.values()))),'conditions':conditions,'window_IoU':values.tolist()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite latent Queue verification')
    study=json.loads(args.study.read_text());cfg=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem
    for name,digest,key in [('models',study['model_seal_sha256'],'exports'),('selection',study['selection_seal_sha256'],'selection')]:
        p=source/(name+'_sealed.json')
        if sha256_file(p)!=digest or json.loads(p.read_text())!=study[key]:raise ValueError('Sealed decision changed')
    if study['phase_read_order']!=[{'phase':'fit','prior_seal_sha256':None},{'phase':'calibration','prior_seal_sha256':study['model_seal_sha256']},
        {'phase':'assessment','prior_seal_sha256':study['selection_seal_sha256']},{'phase':'stress','prior_seal_sha256':study['selection_seal_sha256']}]:raise ValueError('Heldout read order differs')
    sets=[set(s) for s in cfg['seeds'].values()]
    if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]) or any(set([7907,8017,9109])&a for a in sets):raise ValueError('Simulation seed split/final seal violated')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_all_CTM_scenes_features_teachers_decisions_and_deployment_replay',
        'source_study_sha256':sha256_file(args.study),'scope':'Regenerate all268scenes;independent row feature and selector/teacher/mask/weight implementation;6portable models replay;all300cal grid entries and heldout decisions/scores. No independent tree refit.',
        'feature_absolute_tolerance':2e-6,'score_probability_absolute_tolerance':1e-12,'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();datasets={};scene_count=window_count=0;maximum_feature=0.;maximum_soft=0.;scene_checks=[]
    for record in study['datasets']:
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Dataset changed')
        data=dict(np.load(record['path']));datasets[record['phase']]=data;seen=set();counterfactual={}
        for scene in record['scenes']:
            path=Path(scene['path'])
            if sha256_file(path)!=scene['sha256']:raise ValueError('Raw synthetic scene changed')
            config=scene['configuration'];regenerated=simulate(Scenario(**config))
            with np.load(path) as saved:
                if any(not np.array_equal(saved[k],regenerated[k],equal_nan=True) for k in saved.files):raise ValueError('Independent simulator regeneration differs')
            latent,quality,truth=[regenerated[k] for k in ['latent_speed','eligible','queue_truth']];cut=.6*config['free_speed']
            digest=hashlib.sha256(latent.tobytes()+quality.tobytes()).hexdigest();key=(scene['seed'],scene['case'])
            if digest!=scene['latent_quality_sha256'] or (key in counterfactual and counterfactual[key]!=digest):raise ValueError('Noise changed latent/quality counterfactual')
            counterfactual[key]=digest;valid_origins=[]
            for at in range(12,len(latent)-6):
                prior=quality[at-12:at];ongoing=bool(((truth[at-12:at]&prior).sum(axis=0)>=2).any());future=truth[at+1:at+7]
                if prior.sum()/prior.size<.7 or not future.any() or (not ongoing and (future[:5].any() or not future[5].any())):continue
                last=[]
                for j in range(config['cells']):
                    ids=np.flatnonzero(prior[:,j]);last.append(latent[at-12+ids[-1],j] if len(ids) else config['free_speed'])
                persistence=np.broadcast_to(np.array(last)<=cut,future.shape);union=np.count_nonzero(future|persistence)
                if (np.count_nonzero(future&persistence)/union if union else 1.)>.9:continue
                valid_origins.append(at)
            if valid_origins!=scene['origins']:raise ValueError('Independent latent selector differs')
            ids=np.flatnonzero(data['scene']==scene['scene'])
            if data['at'][ids].tolist()!=valid_origins:raise ValueError('Dataset scene/window order differs')
            for i,at in zip(ids,valid_origins):
                hv=regenerated['observation_speed'][at-12:at+1].copy();hq=regenerated['observation_flow'][at-12:at+1].copy();valid=quality[at-12:at+1].copy()
                hv[-1]=regenerated['masked_speed'][at];hq[-1]=regenerated['masked_flow'][at];valid[-1]&=np.isfinite(hv[-1])&np.isfinite(hq[-1])
                condition=bool(((truth[at-12:at]&quality[at-12:at]).sum(axis=0)>=2).any())
                x=manual_features(hv,hq,valid,config,condition);error=float(np.abs(x-data['x'][i]).max());maximum_feature=max(maximum_feature,error)
                if error>2e-6 or condition!=data['ongoing'][i]:raise ValueError('Independent causal feature/condition differs')
                obs=regenerated['observation_speed'][at+1:at+7];target=truth[at+1:at+7];mask=quality[at+1:at+7]
                sigma=scene['sigma'];soft=(obs<=cut).astype(float) if sigma==0 else .5*erfc((obs-cut)/(sigma*math.sqrt(2)))
                soft_error=float(np.abs(soft.astype(np.float32)-data['soft'][i]).max());maximum_soft=max(maximum_soft,soft_error)
                if not np.array_equal(data['observed'][i],obs<=cut) or not np.array_equal(data['latent'][i],target) or not np.array_equal(data['available'][i],mask) or soft_error>1e-7:raise ValueError('Independent teacher or eligibility differs')
                seen.add(int(i));window_count+=1
            scene_count+=1;scene_checks.append({'phase':record['phase'],'seed':scene['seed'],'case':scene['case'],'sigma':scene['sigma'],'n_windows':len(ids),'regeneration_selector_features_teachers_exact_within_tolerance':True})
        if len(seen)!=len(data['x']):raise ValueError('Unverified dataset windows')
        print('latent_queue_verified_data',record['phase'],len(seen),'scenes',len(record['scenes']),flush=True)
    fit=datasets['fit'];allowed=fit['available'].copy();allowed[~fit['ongoing'],:5]=False;expected=np.zeros(allowed.shape,float)
    for c in [False,True]:
        ids=np.flatnonzero(fit['ongoing']==c)
        for i in ids:expected[i]=allowed[i]/max(int(allowed[i].sum()),1)/len(ids)
    weights=expected.ravel()[allowed.ravel()];weights/=weights.mean();record=study['fit_exposure']
    with np.load(record['path']) as a:
        if sha256_file(record['path'])!=record['sha256'] or not np.array_equal(a['flat_indices'],np.flatnonzero(allowed)) or not np.allclose(a['weights'],weights,rtol=0,atol=1e-14):raise ValueError('Training exposure/weights differ')
    probability={};calls=0;maximum_probability=0.
    for record in study['probability_files']:
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Saved probabilities changed')
        data=datasets[record['phase']];x=data['x'].reshape(-1,58);shape=data['latent'].shape;probability[record['phase']]={}
        with np.load(record['path']) as saved:
            for arm,members in study['exports'].items():
                predictions=[]
                for member in members:
                    if sha256_file(member['path'])!=member['sha256']:raise ValueError('Portable model changed')
                    model=Booster(model_file=member['path'])
                    if model.num_feature()!=58 or model.num_trees()!=300:raise ValueError('Model budget differs')
                    predictions.append(model.predict(x,num_threads=4).reshape(shape));calls+=1
                p=np.clip(sum(predictions)/len(predictions),0,1);error=float(np.abs(p-saved[arm]).max());maximum_probability=max(maximum_probability,error)
                if error>1e-12:raise ValueError('Independent model deployment differs')
                probability[record['phase']][arm]=p
            persistence=(data['x'][:,:,39]<=data['x'][:,:,56]).reshape(shape).astype(float)
            if not np.array_equal(persistence,saved['persistence']):raise ValueError('Persistence baseline differs')
            probability[record['phase']]['persistence']=persistence
    cal=datasets['calibration'];grids=0
    for arm,contracts in study['selection']['choices'].items():
        for contract,choice in contracts.items():
            rows=[]
            for row in choice['grid']:
                measured=measure(cal[contract],decisions(probability['calibration'][arm],cal['ongoing'],row['setting']),cal['available'],cal['ongoing']);grids+=1
                if abs(measured['macro']-row['score'])>1e-12 or any(abs(measured['conditions'][k]-row['conditions'][k])>1e-12 for k in measured['conditions']):raise ValueError('Cal grid differs')
                rows.append(row)
            winner=max(rows,key=lambda r:(r['score'],r['setting']['thresholds']['onset']==r['setting']['thresholds']['ongoing'],not r['setting']['force'],-sum(abs(t-.5) for t in r['setting']['thresholds'].values())))
            if winner!=choice['selected']:raise ValueError('Cal threshold selection differs')
    left,right=[study['selection']['choices'][arm]['observed']['selected'] for arm in ['observation_hard','observation_soft']]
    primary='observation_soft' if right['score']-left['score']>=.005 and min(right['conditions'][c]-left['conditions'][c] for c in ['onset','ongoing'])>=-.01 else 'observation_hard'
    if primary!=study['selection']['primary']:raise ValueError('Primary selection differs')
    checks=0;maximum_score=0.
    for phase,arms in study['assessment'].items():
        data=datasets[phase]
        for arm,policies in arms.items():
            for policy,contracts in policies.items():
                setting={'thresholds':{'onset':.5,'ongoing':.5},'force':False} if arm=='persistence' else study['selection']['choices'][arm][policy.removesuffix('_cal')]['selected']['setting']
                pred=decisions(probability[phase][arm],data['ongoing'],setting)
                for contract,saved in contracts.items():
                    metric=measure(data[contract],pred,data['available'],data['ongoing']);error=max(abs(metric['macro']-saved['macro']),float(np.abs(np.array(metric['window_IoU'])-saved['window_IoU']).max()));maximum_score=max(maximum_score,error)
                    if error>1e-12:raise ValueError('Heldout window scoring differs')
                    for scene in saved['scenes']:
                        ids=np.flatnonzero(data['scene']==scene['scene']);values=np.array(metric['window_IoU']);means={name:float(values[ids[data['ongoing'][ids]==c]].mean()) for name,c in [('onset',False),('ongoing',True)] if (data['ongoing'][ids]==c).any()}
                        if any(abs(means[k]-scene['condition_IoU'][k])>1e-12 for k in means) or abs(np.mean(list(means.values()))-scene['macro_available_conditions'])>1e-12:raise ValueError('Heldout scene diagnostic differs')
                    checks+=len(saved['window_IoU'])
    result={'registered':registered,'source_snapshot':snapshot,'n_regenerated_scenes':scene_count,'n_independent_feature_teacher_windows':window_count,'scene_checks':scene_checks,
        'maximum_feature_difference':maximum_feature,'maximum_soft_teacher_difference':maximum_soft,'training_rows_and_weights_verified':int(allowed.sum()),
        'n_independent_booster_calls':calls,'maximum_probability_difference':maximum_probability,'n_cal_grid_entries_verified':grids,
        'n_heldout_window_scores_verified':checks,'maximum_score_difference':maximum_score,'primary_selection_exact':True,'seconds':time.perf_counter()-start,
        'formal_Queue':None,'formal_Total':None,'scope':'All source scenes/causal inputs/labels/weights and portable model decisions/scores verified;does not independently refit6models or validate organizer latent truth.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8')
    print('latent_queue_verified','scenes',scene_count,'windows',window_count,'booster_calls',calls,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
