"""Independent FIT bank replay and direct prefix-set matrix evaluation."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import Booster
from threadpoolctl import threadpool_limits
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.causal_queue import released_origin_rows
from trafficflow.queue import QueueModel
from trafficflow.expected_iou_queue import independent_iou_set
from trafficflow.joint_queue_scenarios import bank_prefix_set
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol
from verify_onset_rank_queue import manual_baseline,manual_metric,same_metric,paired,same_paired
from verify_rolling_onset_decoders import manual_threshold


LAWS=[('independent_full','independent','full'),('independent_empirical','independent','empirical'),('copula_full','copula','full'),('copula_empirical','copula','empirical')]
ORDER=['incumbent','threshold_raw','iou_cal_nonempty']+[r[0] for r in LAWS]


def rebuild_bank(fit):
    onset=np.flatnonzero(~fit['ongoing']);y=fit['labels'][onset,-1];m=fit['available'][onset];n=len(onset);nodes=len(fit['links']);rates=[];positive=[];eligible=[]
    for j in range(nodes):
        values=y[:,j][m[:,-1,j]];positive.append(int(values.sum()));eligible.append(len(values));rates.append(int(values.sum())/len(values) if len(values) else .5)
    rates=np.array(rates);uniforms=[];coupled=[];choices=[];members=[];last=[];evidence=[]
    for member,seed in enumerate([20261013,20261029]):
        rng=np.random.default_rng(seed);ids=rng.integers(0,n,size=2048);u=rng.random(2048*nodes).reshape(2048,nodes);joint=u.copy()
        for j in range(nodes):
            known=m[ids,-1,j];ones=known&(y[ids,j]==1);zeros=known&(y[ids,j]==0);joint[ones,j]=u[ones,j]*rates[j];joint[zeros,j]=rates[j]+u[zeros,j]*(1-rates[j])
        uniforms.append(u);coupled.append(joint);choices.append(ids);members.append(np.repeat(member,2048));last.append(m[ids,-1]);evidence.append(np.array([bool(m[i].any()) for i in ids]))
    return {'independent_uniform':np.vstack(uniforms),'copula_uniform':np.vstack(coupled),'last_eligible':np.vstack(last),'evidence':np.concatenate(evidence),'template_indices':np.concatenate(choices),'member':np.concatenate(members),
        'fitted_rates':rates,'fit_positive_counts':np.array(positive),'fit_eligible_counts':np.array(eligible),'seeds':np.array([20261013,20261029]),'draws_per_seed':np.array(2048),'links':fit['links'],'fit_onset_indices':onset}


def matrix_prefix(probability,links,bank,dependence,availability,member=None):
    """Direct intersections/unions with every prefix indicator, no cumsum."""
    p=np.asarray(probability);order=np.lexsort((np.asarray(links).astype(str),-p));use=np.ones(len(bank['member']),bool) if member is None else bank['member']==member;y=bank[dependence+'_uniform'][use]<p[None,:]
    m=bank['last_eligible'][use] if availability=='empirical' else np.ones(y.shape,bool);e=bank['evidence'][use] if availability=='empirical' else np.ones(len(y),bool)
    prefix=(np.arange(len(p))[:,None]<np.arange(len(p)+1)[None,:]).astype(float);positive=(y&m)[:,order].astype(float);coverage=m[:,order].astype(float);intersection=positive@prefix;predicted=coverage@prefix;union=predicted+positive.sum(axis=1)[:,None]-intersection
    scores=np.divide(intersection,union,out=np.broadcast_to(e[:,None],union.shape).astype(float).copy(),where=union>0);curve=np.mean(scores,axis=0);k=int(np.flatnonzero(curve>=curve.max()-1e-12)[0]);out=np.zeros(len(p),int);out[order[:k]]=1
    return out,curve


def calibrate(state,p):
    p=np.clip(p,1e-6,1-1e-6);linear=state['slope']*np.log(p/(1-p))+state['intercept'];return 1/(1+np.exp(-linear))


def decode(a,p,controls,bank,probes):
    preds={**controls,**{role:controls['incumbent'].copy() for role,_,_ in LAWS}};links=np.tile(a['links'],len(a['ongoing'])*6);checks=0;maximum=0.;iid_errors=[]
    if probes:
        for role,_,_ in LAWS:
            for member in [0,1]:preds[role+'_member'+str(member)]=controls['incumbent'].copy()
    for window in pd.unique(a['window']):
        ids=np.flatnonzero(a['window']==window)
        if a['condition'][ids[0]]!='queue_onset':continue
        last=ids[a['lead_minutes'][ids]==30];order=pd.Index(links[last]).get_indexer(bank['links']);last=last[order];probability=p[last]
        for role,dependence,availability in LAWS:
            for member in ([None,0,1] if probes else [None]):
                prediction,curve=matrix_prefix(probability,bank['links'],bank,dependence,availability,member);actual,detail=bank_prefix_set(probability,bank['links'],bank,dependence,availability,member);error=float(np.max(np.abs(curve-detail['curve'])));maximum=max(maximum,error);checks+=1
                if error>1e-10 or not np.array_equal(prediction,actual):raise ValueError('Independent prefix-matrix scenario decision differs')
                key=role if member is None else role+'_member'+str(member);preds[key][ids]=0;preds[key][last]=prediction
                if role=='independent_full' and member is None:
                    exact=independent_iou_set(probability,bank['links'])[1]['expected_by_size'];iid_errors.append(float(np.abs(curve-exact).max()))
    return preds,checks,maximum,iid_errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite scenario audit')
    root=Path('reports/research');study=json.loads(args.study.read_text());source_path=root/'C74_probabilistic_onset_sets.json';source=json.loads(source_path.read_text());rolling=json.loads((root/'C71_rolling_onset_threshold_refits.json').read_text());raw_proof=json.loads((root/'C72b_rolling_onset_threshold_verification.json').read_text());probability_proof=json.loads((root/'C75_probabilistic_onset_verification.json').read_text());cfg=study['registered']['protocol']
    if sha256_file(source_path)!=cfg['source_study_sha256']:raise ValueError('Probability source changed')
    for path,digest in {**raw_proof['source_files_sha256'],**probability_proof['source_files_sha256'],**cfg['unchanged_source_constructors_sha256']}.items():
        if sha256_file(path)!=digest:raise ValueError('Reused verified public inputs changed')
    recorded={r['path']:r['sha256'] for r in probability_proof['source_snapshot']['captured_files']}
    if sha256_file('src/trafficflow/expected_iou_queue.py')!=recorded['src/trafficflow/expected_iou_queue.py']:raise ValueError('Frozen calibration/IID kernel changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_fit_bank_and_prefix_matrix_audit','source_study_sha256':sha256_file(args.study),
        'scope':'Oldraw/probability proofs rehashed/reused. All fit banks reconstructed per node/RNG,all new ensemble/member prefix curves independently via indicator-matrix set counts,grids/gates/scores and fresh sampled portable histories. No new fullraw/classifier/calibration fitting.',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public');records=[];maximum=0.;iid_errors=[]
    totals={'fit_banks_verified':0,'fit_onset_window_occurrences':0,'scenario_curve_checks':0,'decision_onset_Booster_calls':0,'decision_reference_probability_calls':0,'portable_history_windows':0,'portable_role_calls':0,'additional_shuffled_calls':0,'future_history_rejections':0,'new_raw_window_rebuilds':0}
    for case,old,roll in zip(study['cases'],source['cases'],rolling['cases']):
        if case['name']!=old['name'] or case['name']!=roll['name']:raise ValueError('Case binding differs')
        path=Path('artifacts/research')/args.study.stem/case['name'];r=case['fit_record']
        if sha256_file(r['path'])!=r['sha256']:raise ValueError('Fit source changed')
        fit=dict(np.load(r['path']));expected_bank=rebuild_bank(fit);b=case['fitted_bank']
        if sha256_file(b['path'])!=b['sha256']:raise ValueError('Scenario bank changed')
        bank=dict(np.load(b['path']))
        if bank.keys()!=expected_bank.keys() or any(not np.array_equal(bank[k],expected_bank[k]) for k in expected_bank):raise ValueError('Independent fit-only bank replay differs')
        totals['fit_banks_verified']+=1;totals['fit_onset_window_occurrences']+=len(expected_bank['fit_onset_indices'])
        for name,digest in case['seals'].items():
            if sha256_file(path/(name+'_sealed.json'))!=digest:raise ValueError('Scenario policy seal changed')
        for read in case['phase_reads']:
            required={'decision':'bank','selection':'decision','assessment':'selection'}[read['phase']]
            if read['prior_seal_sha256']!=case['seals'][required]:raise ValueError('Scenario causal read order differs')
        portable={}
        for role,row in case['exports'].items():
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Portable scenario model changed')
            portable[role]=QueueModel.load(Path(row['path']))
        baseline=portable['incumbent'];arm=case['feature_arm'];boosters=[]
        for t in roll['training']:
            if t['role']=='onset' and t['arm']==arm:boosters.append(Booster(model_file=t['model']['path']))
        panel=case['panel'];net,topology=release.network(panel),release.topology(panel);raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)}
        @lru_cache(maxsize=8)
        def raw_day(date):return release.read(raw_paths[date])
        evaluated={'selection':[],'assessment':[]};checks=[];poisoned=False
        for item in case['source_records']:
            r=item['source_record'];f=item['feature_file'];phase=r['phase']
            if sha256_file(r['path'])!=r['sha256'] or sha256_file(f['path'])!=f['sha256']:raise ValueError('Verified eval cache changed')
            a=dict(np.load(r['path']))
            if phase=='decision':
                x=pd.read_parquet(f['path']);raw=np.mean([b.predict(x,num_threads=4) for b in boosters],axis=0);p=calibrate(case['calibration_state'],raw);base=manual_baseline(a,baseline.baseline.probability(x),baseline.setting);totals['decision_onset_Booster_calls']+=2;totals['decision_reference_probability_calls']+=1
                nonempty=base.copy();links=np.tile(a['links'],len(a['ongoing'])*6)
                for window in pd.unique(a['window']):
                    ids=np.flatnonzero(a['window']==window)
                    if a['condition'][ids[0]]!='queue_onset':continue
                    nonempty[ids]=0;last=ids[a['lead_minutes'][ids]==30];nonempty[last]=independent_iou_set(p[last],links[last],allow_empty=False)[0]
                controls={'incumbent':base,'threshold_raw':manual_threshold(a,raw,base,roll['threshold_choice']['setting']),'iou_cal_nonempty':nonempty}
            else:
                old_pred=next(q for q in old['prediction_records'] if q['phase']==phase and q['seed']==r['seed'])
                if sha256_file(old_pred['path'])!=old_pred['sha256']:raise ValueError('C75 verified probability file changed')
                with np.load(old_pred['path']) as q:p=q['calibrated_probability'];controls={role:q[role] for role in ORDER[:3]}
            pred,n,error,mc=decode(a,p,controls,bank,probes=phase!='decision');totals['scenario_curve_checks']+=n;maximum=max(maximum,error);iid_errors.extend(mc)
            if phase=='decision':
                for role,value in case['decision_metrics'].items():same_metric(manual_metric(a,pred[role]),value)
                if max(ORDER,key=lambda k:case['decision_metrics'][k]['proxy_IoU'])!=case['decision_selected']:raise ValueError('Scenario decision winner differs')
            else:
                if phase=='assessment':pred['selected']=pred[case['selected']]
                row=next(q for q in case['prediction_records'] if q['phase']==phase and q['seed']==r['seed'])
                if sha256_file(row['path'])!=row['sha256']:raise ValueError('New prediction file changed')
                with np.load(row['path']) as saved:
                    if any(not np.array_equal(saved[k],v) for k,v in pred.items()):raise ValueError('Independent scenario/probe forecast differs')
                metrics={k:manual_metric(a,pred[k]) for k in ORDER+(['selected'] if phase=='assessment' else [])};wanted=next(q for q in case[phase] if q['seed']==r['seed'])
                for k,v in metrics.items():same_metric(v,wanted['models'][k])
                evaluated[phase].append({'seed':r['seed'],'models':metrics})
            sample=set();first=set();width=6*len(a['links'])
            for ongoing in [False,True]:
                ids=np.flatnonzero(a['ongoing']==ongoing)
                if len(ids):sample.update([int(ids[0]),int(ids[-1])]);first.add(int(ids[0]))
            for i in sorted(sample):
                origin=pd.Timestamp(str(a['origins'][i]));raw=raw_day(origin.date().isoformat());t=pd.to_datetime(raw.timestamp,utc=True);history=raw.loc[(t>=origin-pd.Timedelta(minutes=60))&(t<origin)].copy();history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True);future=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]];target=pd.MultiIndex.from_product([future,a['links']],names=['timestamp','link_id']).to_frame(index=False);condition='queue_ongoing' if a['ongoing'][i] else 'queue_onset';offset=slice(i*width,(i+1)*width);totals['portable_history_windows']+=1
                for role,model in portable.items():
                    expected=pred[case['exports'][role]['candidate']][offset];out=model.predict_window(history,target,net,topology,origin,condition);totals['portable_role_calls']+=1
                    if not out[['timestamp','link_id']].equals(target) or not np.array_equal(out.queue_pred,expected):raise ValueError('Portable scenario forecast differs')
                    if i in first:
                        order=np.random.default_rng(101).permutation(width);out=model.predict_window(history,target.iloc[order],net,topology,origin,condition);totals['additional_shuffled_calls']+=1
                        if not np.array_equal(out.queue_pred,expected[order]):raise ValueError('Scenario forecast depends on target order')
                if condition=='queue_onset' and not poisoned:
                    bad=pd.concat([history,history.iloc[[0]].assign(timestamp=future[0])],ignore_index=True)
                    for model in portable.values():
                        try:model.predict_window(bad,target,net,topology,origin,condition)
                        except ValueError:totals['future_history_rejections']+=1
                        else:raise ValueError('Scenario model accepted future measurement')
                    poisoned=True
            checks.append({'phase':phase,'seed':r['seed'],'scenario_curves_controls_scores_and_member_probes_exact':True})
        gate=paired(evaluated['selection'],case['dates']['selection'],'incumbent',case['decision_selected']);same_paired(gate,case['selection_gate']);selected=case['decision_selected'] if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent'
        if selected!=case['selected']:raise ValueError('Scenario selection gate differs')
        for role,value in case['comparisons'].items():same_paired(paired(evaluated['assessment'],case['dates']['assessment'],'incumbent',role),value)
        records.append({'name':case['name'],'fit_bank_member_curves_policy_gates_independently_verified':True,'n_scenarios':len(bank['member']),'fit_last_eligible_fraction':float(fit['available'][expected_bank['fit_onset_indices'],-1].mean()),'bank_last_eligible_fraction':float(bank['last_eligible'].mean()),'bank_any_evidence_fraction':float(bank['evidence'].mean()),'checks':checks});print('joint_scenario_verified_case',case['name'],flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':records,'totals':totals,'maximum_matrix_cumsum_curve_difference':maximum,
        'independent_full_MonteCarlo_curve_error_vs_exact_IID':{'n_window_occurrences':len(iid_errors),'maximum':max(iid_errors),'mean':float(np.mean(iid_errors))},'reused_raw_window_proof':2196,'reused_calibration_and_probability_audit':'C75_probabilistic_onset_verification',
        'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,'scope':'Fitted bank/RNG and new prefix-set math verified;old raw/probability proofs reused explicitly. Portable sampled histories fresh;no all-window raw rebuild or new classifiers. MC error is conditional on frozen probabilities,not training/model uncertainty.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8');print('joint_scenario_verification_COMPLETE',totals,'seconds',result['seconds'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
