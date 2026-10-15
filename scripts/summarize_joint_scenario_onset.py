"""Matched scenario contrasts, seed stability and exact selection-bias replay."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.joint_queue_scenarios import scenario_prefix_set
from summarize_rolling_onset_decoders import synchronized_comparison


ROLES=['independent_full','independent_empirical','copula_full','copula_empirical']


def scalar_nonempty_check(row):
    masks=list(itertools.product([0,1],repeat=5));truth=[(tuple(int(j==i) for j in range(5)),.2) for i in range(5)] if row['kind']=='single' else [((1,1,0,0,0),.6),((0,0,1,1,1),.4)];scenes=[]
    for y,py in truth:
        for mask in masks:
            probability=py
            for j in range(5):
                rate=row['positive_coverage'][j] if y[j] else row['negative_coverage'][j];probability*=rate if mask[j] else 1-rate
            if probability>0:scenes.append((y,mask,row['earlier_five_steps_have_evidence'] or any(mask),probability))
    supplied=row['hypothetical_scenarios'];errors=[]
    if len(scenes)!=len(supplied['weights']):raise ValueError('Known-law support differs')
    for i,(y,m,e,w) in enumerate(scenes):
        if tuple(supplied['labels'][i])!=y or tuple(supplied['last_eligible'][i])!=m or bool(supplied['evidence'][i])!=bool(e):raise ValueError('Known-law scenario differs')
        errors.append(abs(w-supplied['weights'][i]))
    def score(pred,conditional=False):
        value=weight=0.
        for y,m,e,w in scenes:
            true={j for j in range(5) if y[j] and m[j]}
            if conditional and not true:continue
            predicted={j for j in range(5) if pred[j] and m[j]};union=true|predicted;value+=w*(len(true&predicted)/len(union) if union else float(e));weight+=w
        return value/weight
    values=np.array([score(p) for p in masks]);proxy=np.array([score(p,True) for p in masks]);marginal=np.sum([np.array(y)*w for y,m,e,w in scenes],axis=0);order=np.lexsort((np.array([str(i) for i in range(5)]),-marginal));curve=[]
    for k in range(6):
        pred=np.zeros(5,int);pred[order[:k]]=1;curve.append(score(pred))
    errors.extend(np.abs(np.array(curve)-row['joint_prefix_expected_curve']).tolist())
    for policy in row['policies'].values():
        errors.extend([abs(score(policy['prediction'])-policy['latent_population_expected_scored_IoU']),abs(score(policy['prediction'],True)-policy['observed_positive_subset_expected_IoU'])])
    proxy_policy=row['policies']['observed_positive_distribution_selected']['prediction'];errors.extend([abs(values.max()-row['known_joint_global_optimum_diagnostic']['expected_IoU']),abs(score(proxy_policy,True)-proxy.max())])
    if max(errors)>1e-12 or not row['all_latent_labels_nonempty']:raise ValueError('Independent nonempty/mask/selection replay differs')
    return {'name':row['name'],'maximum_difference':max(errors),'proxy_selected_population_regret':row['proxy_selected_population_regret'],'prefix_gap':row['restricted_prefix_gap']}


def prefix_counterexample():
    # Positive population laws only; marginal ranking puts node B before C/D/E.
    y=np.array([[0,1,0,0,0],[1,0,1,1,0],[1,0,1,0,1],[1,0,0,1,1]],bool);w=np.array([.41,.59/3,.59/3,.59/3]);p=w@y;m=np.ones(y.shape,bool);e=np.ones(4,bool);pred,detail=scenario_prefix_set(p,list('ABCDE'),y,m,e,w);values=[];sets=list(itertools.product([0,1],repeat=5))
    for s in sets:
        candidate={i for i,v in enumerate(s) if v};expectation=0.
        for truth,weight in zip(y,w):
            true=set(np.flatnonzero(truth));expectation+=weight*len(candidate&true)/len(candidate|true)
        values.append(expectation)
    best=int(np.argmax(values));gap=values[best]-detail['expected_proxy_IoU']
    if abs(gap-.0065)>1e-12:raise ValueError('Restricted-prefix counterexample differs')
    return {'self_defined_distribution':{'labels':y.astype(int).tolist(),'weights':w.tolist()},'marginal_probability':p.tolist(),'prefix_prediction':pred.tolist(),'prefix_expected_IoU':detail['expected_proxy_IoU'],
        'global_known_law_diagnostic':{'prediction':list(sets[best]),'expected_IoU':values[best],'deployment_allowed':False},'prefix_gap':gap,'scope':'Additional mathematical counterexample,not a traffic model/outcome or tuning of C79.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite joint scenario summary')
    root=Path('reports/research');study_path=root/'C79_joint_scenario_onset.json';audit_path=root/'C80b_joint_scenario_verification.json';stress_path=root/'C78_nonempty_joint_mask_stress.json';prior_path=root/'C74_probabilistic_onset_sets.json';study=json.loads(study_path.read_text());audit=json.loads(audit_path.read_text());stress=json.loads(stress_path.read_text());prior=json.loads(prior_path.read_text())
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(study_path) or len(audit['cases'])!=4:raise ValueError('New audit source differs')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();paired_cases=[]
    for case,old in zip(study['cases'],prior['cases']):
        copy={**case,'assessment':[]}
        for run in case['assessment']:
            reference=next(r for r in old['assessment'] if r['seed']==run['seed']);copy['assessment'].append({**run,'models':{**run['models'],'exact_independent_full':reference['models']['iou_cal']}})
        paired_cases.append(copy)
    pairs=[('incumbent',r) for r in ['selected']+ROLES]+[('independent_full','copula_full'),('independent_empirical','copula_empirical'),('independent_full','independent_empirical'),('copula_full','copula_empirical'),('threshold_raw','copula_empirical'),('iou_cal_nonempty','copula_empirical'),('exact_independent_full','independent_full'),('incumbent','iou_cal_nonempty')]
    comparisons=[synchronized_comparison(paired_cases,left,right) for left,right in pairs];cases=[]
    for case in study['cases']:
        stability={role:{'n_onset_windows':0,'member_set_disagreements':0,'member_hamming_cells':0,'maximum_member_size_difference':0} for role in ROLES};contrasts={k:{'n_onset_windows':0,'set_disagreements':0,'hamming_cells':0} for k in ['copula_full_vs_independent_full','copula_empirical_vs_independent_empirical']};seen=set()
        for item in case['source_records']:
            r=item['source_record']
            if r['phase']!='assessment':continue
            pred=next(p for p in case['prediction_records'] if p['phase']=='assessment' and p['seed']==r['seed'])
            if sha256_file(r['path'])!=r['sha256'] or sha256_file(pred['path'])!=pred['sha256']:raise ValueError('Summary source cache changed')
            with np.load(r['path']) as a,np.load(pred['path']) as p:
                width=6*len(a['links'])
                for i,origin in enumerate(a['origins']):
                    if a['ongoing'][i] or str(origin) in seen:continue
                    seen.add(str(origin));ids=slice(i*width,(i+1)*width)
                    for role,d in stability.items():
                        left=p[role+'_member0'][ids];right=p[role+'_member1'][ids];h=int(np.count_nonzero(left!=right));d['n_onset_windows']+=1;d['member_set_disagreements']+=int(h>0);d['member_hamming_cells']+=h;d['maximum_member_size_difference']=max(d['maximum_member_size_difference'],int(abs(left.sum()-right.sum())))
                    for key,left,right in [('copula_full_vs_independent_full','copula_full','independent_full'),('copula_empirical_vs_independent_empirical','copula_empirical','independent_empirical')]:
                        h=int(np.count_nonzero(p[left][ids]!=p[right][ids]));d=contrasts[key];d['n_onset_windows']+=1;d['set_disagreements']+=int(h>0);d['hamming_cells']+=h
        with np.load(case['fitted_bank']['path']) as b:
            bank_stats={'n_nodes':len(b['links']),'n_nodes_with_partial_fit_rate':int(((b['fitted_rates']>0)&(b['fitted_rates']<1)).sum()),'n_nodes_never_positive':int((b['fit_positive_counts']==0).sum()),
                'n_fit_onset_windows':case['n_fit_onset_windows'],'fraction_uniform_entries_changed_by_dependence':float((b['copula_uniform']!=b['independent_uniform']).mean()),'last_eligible_fraction':float(b['last_eligible'].mean()),'any_sixstep_evidence_fraction':float(b['evidence'].mean())}
        cases.append({'name':case['name'],'bank_statistics':bank_stats,'MonteCarlo_member_stability':stability,'empirical_dependence_decision_differences':contrasts})
    gates={}
    for role in ['selected']+ROLES:
        rows=[c['comparisons'][role] for c in study['cases']];n=sum(r['paired_macro_gain']>=.01 and r['ci95'] is not None and r['ci95'][0]>0 for r in rows);bounded=all(r['paired_macro_gain']>=-.01 for r in rows);gates[role]={'cases_gain_01_positive_CI':n,'all_case_losses_bounded_01':bounded,'descriptive_gate_passed':n>=2 and bounded}
    checks=[scalar_nonempty_check(row) for row in stress['cases']];counterexample=prefix_counterexample();result={'source_study_sha256':sha256_file(study_path),'independent_audit_sha256':sha256_file(audit_path),'known_law_stress_sha256':sha256_file(stress_path),'reused_exact_IID_reference_sha256':sha256_file(prior_path),'source_snapshot':snapshot,
        'comparisons':comparisons,'n_joint_contrasts':len(comparisons),'cases':cases,'evidence_gates':gates,'known_nonempty_stress_independent_checks':checks,'restricted_prefix_counterexample':counterexample,'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,
        'scope':'Posthoc13uncorrected contrasts,all joint bootstrap replicates independently replayed;MC member probes are conditional simulation variability,not training replicates. Nine known laws scalar-checked;extra prefix counterexample diagnostic only. No policy retuning/promotion.'}
    raw=json.dumps(result,indent=2,allow_nan=False)
    with args.output.open('x',encoding='utf-8') as f:f.write(raw)
    with (folder/'experiment.json').open('x',encoding='utf-8') as f:f.write(raw)
    print('joint_scenario_summary_COMPLETE','contrasts',len(comparisons),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
