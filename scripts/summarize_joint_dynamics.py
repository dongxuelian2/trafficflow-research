"""Posthoc matched Queue/causal-State/forecast diagnostics; no new selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.runtime import sha256_file,capture_source_snapshot
from summarize_rolling_onset_decoders import synchronized_comparison


ARMS=['direct_no_flow','direct_flow','latent_flow'];SEEDS=[20261013,20261029];REGIMES=['R1','R2','R3']


def scalar_state_formula(rows):
    points=[]
    for regime in REGIMES:
        count=sum(r[regime]['n'] for r in rows)
        if not count:return None
        rv=(sum(r[regime]['speed_normalized_SSE'] for r in rows)/count)**.5;rq=(sum(r[regime]['flow_normalized_SSE'] for r in rows)/count)**.5
        points.append(.54*max(0.,1-rv)+.46*max(0.,1-rq))
    return float(sum(points)/3.)


def state_comparisons(cases):
    calendars=[c['dates']['assessment'] for c in cases]
    if any(days!=calendars[0] for days in calendars):raise ValueError('This joint State bootstrap needs identical calendars')
    roles=['causal_carry','causal_tree']+ARMS;n=len(calendars[0]);count=np.array([[[r['models']['causal_carry'][g]['n'] for g in REGIMES] for r in c['state_assessment_records']] for c in cases],float)
    errors=np.array([[[[[r['models'][role][g][k] for g in REGIMES] for r in c['state_assessment_records']] for c in cases] for role in roles] for k in ['speed_normalized_SSE','flow_normalized_SSE']],float)
    rng=np.random.default_rng(20261015);starts=rng.integers(0,n,(4000,int(np.ceil(n/7))));ids=((starts[:,:,None]+np.arange(7))%n).reshape(4000,-1)[:,:n];sample_count=count[:,ids].sum(axis=2);valid=(sample_count>0).all(axis=(0,2));sample_errors=errors[:,:,:,ids].sum(axis=4)
    score=.54*np.maximum(0.,1-np.sqrt(sample_errors[0]/sample_count[None]))+.46*np.maximum(0.,1-np.sqrt(sample_errors[1]/sample_count[None]));samples=score.mean(axis=(1,3))[:,valid]
    scalar=[]
    for index in np.flatnonzero(valid):
        row=[]
        for role in roles:
            points=[]
            for c in cases:
                value=scalar_state_formula([c['state_assessment_records'][d]['models'][role] for d in ids[index]]);points.append(value)
            row.append(float(np.mean(points)))
        scalar.append(row)
    maximum=float(np.abs(samples-np.array(scalar).T).max())
    if maximum>1e-12:raise ValueError('Independent scalar State bootstrap differs')
    points={role:float(np.mean([c['state_assessment_metrics'][role]['public_subset_regime_macro'] for c in cases])) for role in roles};comparisons=[]
    for baseline,candidate in [('causal_carry','causal_tree'),('causal_carry','direct_no_flow'),('causal_tree','direct_no_flow'),('causal_tree','direct_flow'),('causal_tree','latent_flow'),('direct_no_flow','direct_flow'),('direct_flow','latent_flow')]:
        delta=samples[roles.index(candidate)]-samples[roles.index(baseline)];comparisons.append({'baseline':baseline,'candidate':candidate,'point_macro':{baseline:points[baseline],candidate:points[candidate]},'gain':points[candidate]-points[baseline],'ci95':np.quantile(delta,[.025,.975]).tolist(),'probability_positive':float((delta>0).mean())})
    return {'comparisons':comparisons,'roles':roles,'synchronized_same_calendar':True,'days':n,'block_days':7,'resamples':4000,'valid_resamples':int(valid.sum()),'independent_scalar_maximum_difference':maximum,
        'scope':'All276causalcurrentorigins/day,publicnoisybothmissingeligibleR1/R2/R3. PerregimeRMSEpointformula then equalcase mean;NOT completeofficialState. Posthocuncorrected,conditionalonfittedmodels,notoptimizerseeduncertainty or promotion. Tree has more unique labels thanNN.'}


def unique_current_exposure(case):
    packs=[dict(np.load(r['path'])) for r in case['dataset_description']['day_records']];origins=np.load(case['dataset_description']['origin_record']['path'])['indices'];fit=dict(np.load(next(r for r in case['event_records'] if r['phase']=='fit')['path']));dates=case['dates']['fit'];result=[]
    for record in case['batch_records']:
        seq=dict(np.load(record['path']));covered=np.zeros((len(dates),288,len(fit['links'])),bool)
        for i in np.unique(seq['warm']):
            day,at=origins[i];covered[day,at]=True
        for i in np.unique(seq['event']):
            text=str(fit['origins'][i]);date=text[:10];at=int(text[11:13])*12+int(text[14:16])//5;covered[dates.index(date),at]=True
        counts=[]
        for field,mask in [(0,2),(1,3)]:counts.append(sum(int((covered[i]&p['available']&~(p['masked_channels'][:,:,mask]>.5)).sum()) for i,p in enumerate(packs)))
        result.append({'seed':record['seed'],'unique_current_speed_teacher_cells':counts[0],'unique_current_flow_available_cells':counts[1],'flow_optimized':'direct_no_flow0,other2allavailable','strong_tree_teacher_cells':[r['n_teacher_rows'] for r in case['state_trees']['records']]})
    return result


def forecast_diagnostics(case):
    totals={arm:{key:{'n':0,'v':0.,'q':0.} for key in ['all','drop20','recovery20','queue_teacher']} for arm in ARMS};seen={}
    for r in case['event_records']:
        if r['phase']!='assessment':continue
        pr=next(p for p in case['prediction_records'] if p['phase']=='assessment' and p['seed']==r['seed'])
        if sha256_file(r['path'])!=r['sha256'] or sha256_file(pr['path'])!=pr['sha256']:raise ValueError('Frozen forecast input changed')
        a=dict(np.load(r['path']));p=dict(np.load(pr['path']))
        for i,origin in enumerate(a['origins']):
            fingerprint=tuple(p[arm+'_probability'][i].tobytes() for arm in ARMS)
            if str(origin) in seen:
                if seen[str(origin)]!=fingerprint:raise ValueError('Repeated forecast origin differs')
                continue
            seen[str(origin)]=fingerprint;delta=a['truth_speed'][i]-a['truth_current'][i,0];valid=a['available'][i];ov=a['current_available'][i][None];masks={'all':valid,'drop20':valid&ov&(delta<=-.8),'recovery20':valid&ov&(delta>=.8),'queue_teacher':valid&(a['labels'][i]>0)}
            for arm in ARMS:
                for key,mask in masks.items():
                    row=totals[arm][key];ev=(p[arm+'_speed'][i].astype(float)-a['truth_speed'][i].astype(float))[mask];eq=(p[arm+'_flow'][i].astype(float)-a['truth_flow'][i].astype(float))[mask];row['n']+=int(mask.sum());row['v']+=float(np.dot(ev,ev));row['q']+=float(np.dot(eq,eq))
    return {'n_unique_origins':len(seen),'metrics':{arm:{key:{'n':v['n'],'speed_RMSE_kmh':float(np.sqrt(v['v']/v['n'])*25.) if v['n'] else None,'flow_RMSE_per_lane':float(np.sqrt(v['q']/v['n'])*600.) if v['n'] else None} for key,v in rows.items()} for arm,rows in totals.items()},
        'scope':'Origin-dedup,cellpooled publicobservedfutureteacher forecast diagnostics. drop/recovery20km/h relative toeligible rawT are evaluation strata,neverinputs;not knownincidents,State or officialQueue/Physics.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite joint dynamics summary')
    root=Path('reports/research');study_path=root/'C91_joint_flow_latent_dynamics.json';study=json.loads(study_path.read_text());audit=json.loads(args.audit.read_text())
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(study_path):raise ValueError('Audit source binding differs')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');tic=time.perf_counter();comparisons=[synchronized_comparison(study['cases'],a,b) for a,b in [('incumbent','selected'),('incumbent','direct_no_flow'),('incumbent','direct_flow'),('incumbent','latent_flow'),('direct_no_flow','direct_flow'),('direct_no_flow','latent_flow'),('direct_flow','latent_flow')]]
    cases=[]
    for c in study['cases']:
        cases.append({'name':c['name'],'decision_selected':c['decision_selected'],'selected':c['selected'],'selection_gate':c['selection_gate'],'actual_current_teacher_exposure':unique_current_exposure(c),'state_metrics':c['state_assessment_metrics'],'forecast_diagnostics':forecast_diagnostics(c),
            'fit_timings':{'state_tree_seconds':c['state_trees']['seconds'],'networks':[{'arm':t['arm'],'seed':t['seed'],'seconds':t['seconds'],'selected_step':t['selected']['step']} for t in c['training']]}})
    result={'source_study_sha256':sha256_file(study_path),'independent_audit_sha256':sha256_file(args.audit),'source_snapshot':snapshot,'Queue_joint_comparisons':comparisons,'State_joint_diagnostics':state_comparisons(study['cases']),'cases':cases,'seconds':time.perf_counter()-tic,'formal_State':None,'formal_Queue':None,'formal_Physics':None,'formal_Total':None,
        'scope':'Posthoc sourcebound diagnostics of frozen joint models;no model/threshold/loss/alpha choice,formal promotion or platform operations. State and Queue domains differ and cannotbe added intoformalTotal.'};raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'experiment.json').write_text(raw,encoding='utf-8');print('joint_dynamics_summary_COMPLETE','seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
