"""Independent local-neighborhood set matrices and decision-only screen audit."""
import argparse
from functools import lru_cache
import itertools
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
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol
from verify_onset_rank_queue import manual_baseline,manual_metric,same_metric
from verify_rolling_onset_decoders import manual_threshold
from verify_joint_scenario_onset import matrix_prefix,calibrate
from verify_probabilistic_onset import integral_curve,curve_decision


def set_values(candidates,y,m,e,weights=None):
    candidates=np.atleast_2d(candidates).astype(float);positive=(y&m).astype(float);intersection=positive@candidates.T;union=m.astype(float)@candidates.T+positive.sum(axis=1)[:,None]-intersection
    values=np.divide(intersection,union,out=np.broadcast_to(e[:,None],union.shape).astype(float).copy(),where=union>0);w=np.ones(len(y))/len(y) if weights is None else np.asarray(weights)/np.sum(weights)
    return np.sum(w[:,None]*values,axis=0),values


def independent_search(p,links,y,m,e,weights=None):
    order=np.lexsort((np.asarray(links).astype(str),-p));prefixes=np.zeros((len(p)+1,len(p)))
    for k in range(len(p)+1):prefixes[k,order[:k]]=1
    values,_=set_values(prefixes,y,m,e,weights);k=int(np.flatnonzero(values>=values.max()-1e-12)[0]);pred=prefixes[k].astype(int);trace=[];current=float(values[k]);evaluations=0;reason='move_budget'
    for _ in range(8):
        candidates=np.tile(pred,(len(p),1));j=np.arange(len(p));candidates[j,j]=1-candidates[j,j];scores,_=set_values(candidates,y,m,e,weights);evaluations+=len(p);best=float(scores.max())
        if best<=current+1e-6:reason='single_flip_stationary';break
        tied=np.flatnonzero(scores>=best-1e-12);index=min(tied,key=lambda j:(not bool(pred[j]),str(links[j])));operation='remove' if pred[index] else 'add';before=current;pred=candidates[index];current=float(scores[index]);trace.append({'link_id':str(links[index]),'operation':operation,'before':before,'after':current,'gain':current-before})
    return prefixes[k].astype(int),pred,trace,current,evaluations,reason


def validation_bank(fit):
    onset=np.flatnonzero(~fit['ongoing']);y=fit['labels'][onset,-1];m=fit['available'][onset];counts=m[:,-1].sum(axis=0);positives=(y*m[:,-1]).sum(axis=0).astype(int);rates=np.array([int(p)/int(n) if n else .5 for p,n in zip(positives,counts)]);uniform=[];joint=[];idxs=[];members=[];masks=[];evidence=[]
    for member,seed in enumerate([20261107,20261119]):
        rng=np.random.default_rng(seed);idx=rng.integers(0,len(onset),2048);u=rng.random(2048*y.shape[1]).reshape(2048,y.shape[1]);v=u.copy()
        for j in range(y.shape[1]):
            known=m[idx,-1,j];positive=known&(y[idx,j]==1);negative=known&(y[idx,j]==0);v[positive,j]=u[positive,j]*rates[j];v[negative,j]=rates[j]+u[negative,j]*(1-rates[j])
        uniform.append(u);joint.append(v);idxs.append(idx);members.append(np.full(2048,member));masks.append(m[idx,-1]);evidence.append(np.array([m[i].any() for i in idx]))
    return {'independent_uniform':np.vstack(uniform),'copula_uniform':np.vstack(joint),'last_eligible':np.vstack(masks),'evidence':np.concatenate(evidence),'template_indices':np.concatenate(idxs),'member':np.concatenate(members),'fitted_rates':rates,'fit_positive_counts':positives,'fit_eligible_counts':counts,'seeds':np.array([20261107,20261119]),'draws_per_seed':np.array(2048),'links':fit['links'],'fit_onset_indices':onset}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite local-screen audit')
    root=Path('reports/research');path=root/'C83_bounded_search_decision_screen.json';study=json.loads(path.read_text());small_path=root/'C82_bounded_search_known_laws.json';small=json.loads(small_path.read_text());cfg=study['registered']['protocol']
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_bounded_local_decision_screen_audit','source_study_sha256':sha256_file(path),'source_small_law_sha256':sha256_file(small_path),'scope':'Separate validation-bank replay,all32small sets and allsingle-bit candidate neighborhoods via direct set matrices;hypothetical-MC acceptance,public decision controls/metrics/screen and sampled portable checks. Old raw/probability audits reused;no new selection/assessment.', 'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public');totals={'validation_banks':0,'known_law_cases':0,'known_global_sets_enumerated':0,'decision_onset_windows':0,'neighbor_candidates':0,'decision_onset_Booster_calls':0,'decision_reference_probability_calls':0,'portable_history_windows':0,'portable_role_calls':0,'additional_shuffled_calls':0,'future_history_rejections':0};records=[];maximum=0.
    stress=json.loads((root/'C78_nonempty_joint_mask_stress.json').read_text());counter=json.loads((root/'C81_joint_scenario_matched_summary.json').read_text())['restricted_prefix_counterexample'];laws=[]
    for row in stress['cases']:
        s=row['hypothetical_scenarios'];laws.append((row['name'],np.array(row['marginal_truth_probability']),np.array(s['labels'],bool),np.array(s['last_eligible'],bool),np.array(s['evidence'],bool),np.array(s['weights'])))
    s=counter['self_defined_distribution'];y=np.array(s['labels'],bool);w=np.array(s['weights']);laws.append(('nonprefix_counterexample',w@y,y,np.ones(y.shape,bool),np.ones(len(y),bool),w))
    for (name,p,y,m,e,w),saved in zip(laws,small['cases']):
        if name!=saved['name']:raise ValueError('Known law order differs')
        prefix,proposal,trace,value,evaluations,reason=independent_search(p,np.array([str(i) for i in range(len(p))]),y,m,e,w);best=max(float(set_values(np.array(candidate),y,m,e,w)[0][0]) for candidate in itertools.product([0,1],repeat=len(p)))
        if not np.array_equal(proposal,saved['prediction']) or abs(value-saved['trace']['expected_proxy_IoU'])>1e-10 or abs(best-saved['known_global_optimum_diagnostic']['expected_IoU'])>1e-12:raise ValueError('Small global/neighbor audit differs')
        totals['known_law_cases']+=1;totals['known_global_sets_enumerated']+=2**len(p)
    for case in study['cases']:
        for p,h in cfg['unchanged_kernel_sha256'].items():
            if sha256_file(p)!=h:raise ValueError('Frozen kernel changed')
        fit=dict(np.load(case['fit_record']['path']));expected=validation_bank(fit)
        if sha256_file(case['validation_bank']['path'])!=case['validation_bank']['sha256']:raise ValueError('Validation bank changed')
        bank=dict(np.load(case['validation_bank']['path']))
        if bank.keys()!=expected.keys() or any(not np.array_equal(bank[k],expected[k]) for k in expected):raise ValueError('Independent validation-bank replay differs')
        totals['validation_banks']+=1;portable={}
        for role,row in case['exports'].items():
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Portable screen model changed')
            portable[role]=QueueModel.load(Path(row['path']))
        prefix_model=portable['prefix'];r=case['source_decision_record']['source_record'];f=case['source_decision_record']['feature_file'];a=dict(np.load(r['path']));x=pd.read_parquet(f['path']);models=[Booster(model_str=s) for s in prefix_model.base.model_strings];raw=np.mean([m.predict(x,num_threads=4) for m in models],axis=0);p=calibrate(prefix_model.calibration,raw);incumbent=portable['incumbent'];base=manual_baseline(a,incumbent.baseline.probability(x),incumbent.setting);totals['decision_onset_Booster_calls']+=2;totals['decision_reference_probability_calls']+=1
        threshold_model=QueueModel.load(Path(case['source_controls']['threshold_raw']['path']));threshold=manual_threshold(a,raw,base,threshold_model.setting);nonempty=base.copy();links=np.tile(a['links'],len(a['ongoing'])*6);preds={'incumbent':base,'threshold_raw':threshold,'iou_cal_nonempty':nonempty,'prefix':base.copy(),'local_proposal':base.copy(),'validated_local':base.copy()}
        for window in pd.unique(a['window']):
            ids=np.flatnonzero(a['window']==window)
            if a['condition'][ids[0]]!='queue_onset':continue
            last=ids[a['lead_minutes'][ids]==30];lookup=pd.Index(links[last]).get_indexer(prefix_model.bank['links']);last=last[lookup];prob=p[last];order,curve=integral_curve(prob,links[last]);nonempty[ids]=0;nonempty[last]=curve_decision(order,curve,False)
            model_bank=prefix_model.bank;y=model_bank['copula_uniform']<prob[None,:];m=model_bank['last_eligible'];e=model_bank['evidence'];prefix,proposal,trace,value,n,reason=independent_search(prob,links[last],y,m,e);totals['decision_onset_windows']+=1;totals['neighbor_candidates']+=n
            vy=bank['copula_uniform']<prob[None,:];scores,per_scene=set_values(np.stack([prefix,proposal]),vy,bank['last_eligible'],bank['evidence']);delta=per_scene[:,1]-per_scene[:,0];gain=float(delta.mean());se=float(np.std(delta,ddof=1)/np.sqrt(len(delta)));ci=[gain-1.96*se,gain+1.96*se];accepted=gain>=.001 and ci[0]>0;validated=proposal if accepted else prefix;d=next(d for d in case['onset_diagnostics'] if d['window']==window)
            if not np.array_equal(prefix,d['prefix_prediction']) or len(trace)!=len(d['moves']) or reason!=d['stop_reason'] or n!=d['n_neighbor_evaluations'] or abs(value-d['expected_proxy_IoU'])>1e-10 or bool(d['accepted'])!=bool(accepted) or abs(gain-d['validation_expected_gain'])>1e-12 or not np.allclose(ci,d['validation_paired_MC_ci95'],atol=1e-12,rtol=0):raise ValueError('Independent neighborhood/MC acceptance differs')
            for actual,wanted in zip(trace,d['moves']):
                if actual['link_id']!=wanted['link_id'] or actual['operation']!=wanted['operation'] or abs(actual['after']-wanted['after'])>1e-10:raise ValueError('Independent flip trace differs')
            for role,q in [('prefix',prefix),('local_proposal',proposal),('validated_local',validated)]:preds[role][ids]=0;preds[role][last]=q
        row=case['prediction_record']
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Screen forecast file changed')
        with np.load(row['path']) as saved:
            if not np.array_equal(raw,saved['raw_probability']) or not np.allclose(p,saved['calibrated_probability'],atol=1e-15,rtol=0) or any(not np.array_equal(v,saved[k]) for k,v in preds.items()):raise ValueError('Independent screen forecast differs')
        for role,v in case['decision_metrics'].items():same_metric(manual_metric(a,preds[role]),v)
        passed=case['decision_metrics']['validated_local']['proxy_IoU']>=case['decision_metrics']['incumbent']['proxy_IoU']+.005 and case['decision_metrics']['validated_local']['proxy_IoU']>=case['decision_metrics']['prefix']['proxy_IoU']+.005
        if bool(passed)!=case['local_shortlisted'] or any(q['phase'] not in ['fit','decision'] for q in case['phase_reads']):raise ValueError('Decision-only screen scope differs')
        panel=case['panel'];net,topology=release.network(panel),release.topology(panel);raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)}
        @lru_cache(maxsize=4)
        def raw_day(date):return release.read(raw_paths[date])
        sample=set();first=set();width=6*len(a['links']);poisoned=False
        for ongoing in [False,True]:
            ids=np.flatnonzero(a['ongoing']==ongoing);sample.update([int(ids[0]),int(ids[-1])]);first.add(int(ids[0]))
        for i in sorted(sample):
            origin=pd.Timestamp(str(a['origins'][i]));raw_frame=raw_day(origin.date().isoformat());t=pd.to_datetime(raw_frame.timestamp,utc=True);history=raw_frame.loc[(t>=origin-pd.Timedelta(minutes=60))&(t<origin)].copy();history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True);future=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]];target=pd.MultiIndex.from_product([future,a['links']],names=['timestamp','link_id']).to_frame(index=False);condition='queue_ongoing' if a['ongoing'][i] else 'queue_onset';offset=slice(i*width,(i+1)*width);totals['portable_history_windows']+=1
            for role,model in portable.items():
                out=model.predict_window(history,target,net,topology,origin,condition);totals['portable_role_calls']+=1
                if not np.array_equal(out.queue_pred,preds[role][offset]):raise ValueError('Portable local screen differs')
                if i in first:
                    order=np.random.default_rng(103).permutation(width);out=model.predict_window(history,target.iloc[order],net,topology,origin,condition);totals['additional_shuffled_calls']+=1
                    if not np.array_equal(out.queue_pred,preds[role][offset][order]):raise ValueError('Local model depends on target order')
            if condition=='queue_onset' and not poisoned:
                bad=pd.concat([history,history.iloc[[0]].assign(timestamp=future[0])],ignore_index=True)
                for model in portable.values():
                    try:model.predict_window(bad,target,net,topology,origin,condition)
                    except ValueError:totals['future_history_rejections']+=1
                    else:raise ValueError('Local model accepted future measurements')
                poisoned=True
        records.append({'name':case['name'],'bank_all_neighbor_sets_controls_and_screen_exact':True});print('local_screen_verified_case',case['name'],flush=True)
    if study['campaign_local_shortlisted']!=any(c['local_shortlisted'] for c in study['cases']) or study['selection_arrays_read'] or study['assessment_arrays_read']:raise ValueError('Screen stopping rule differs')
    result={'registered':registered,'source_snapshot':snapshot,'cases':records,'totals':totals,'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,'scope':'Independent hypothetical-law numerical checks and public decision-only scoring. Existing raw/probability evidence reused;no new holdout/private/latent-score claim.'};raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8');print('local_screen_verification_COMPLETE',totals,'seconds',result['seconds'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
