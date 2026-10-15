"""Frozen proxy contrasts and independent scalar known-joint stress replay."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.runtime import sha256_file,capture_source_snapshot
from summarize_rolling_onset_decoders import synchronized_comparison


def scalar_stress(row):
    n=row['n_nodes'];states=list(itertools.product([0,1],repeat=n));p=row['truth_marginals'];truth=[]
    if row['kind']=='all_or_nothing':truth=[((0,)*n,.7),((1,)*n,.3)]
    elif row['kind']=='cluster_switch_nonempty':truth=[((1,1,0,0,0),.6),((0,0,1,1,1),.4)]
    else:
        for y in states:truth.append((y,float(np.prod([p[j] if y[j] else 1-p[j] for j in range(n)]))))
    values=[]
    for pred in states:
        expectation=0.
        for y,py in truth:
            for mask in states:
                pm=py
                for j in range(n):
                    rate=row['positive_coverage'][j] if y[j] else row['negative_coverage'][j];pm*=rate if mask[j] else 1-rate
                if not pm:continue
                ids={j for j in range(n) if mask[j]}
                left={j for j in ids if y[j]};right={j for j in ids if pred[j]};union=left|right
                score=len(left&right)/len(union) if union else float(bool(ids));expectation+=pm*score
        values.append(expectation)
    maximum=max(values);differences=[]
    for role in ['independent_decoder','nonempty_control','threshold_05']:
        i=states.index(tuple(int(v) for v in row[role]['prediction']));differences.append(abs(values[i]-row[role]['true_known_joint_expected_masked_IoU']))
    optimum=row['known_joint_distribution_optimum_diagnostic']
    differences.append(abs(maximum-optimum['expected_masked_IoU']))
    if optimum['deployment_allowed'] or max(differences)>1e-12:raise ValueError('Independent known-joint enumeration differs')
    return {'case':row['name'],'n_candidate_sets':len(states),'maximum_score_difference':max(differences),'regret':maximum-row['independent_decoder']['true_known_joint_expected_masked_IoU']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite probabilistic summary')
    root=Path('reports/research');study_path=root/'C74_probabilistic_onset_sets.json';audit_path=root/'C75_probabilistic_onset_verification.json';stress_path=root/'C76_expected_IoU_known_joint_stress.json';study=json.loads(study_path.read_text());audit=json.loads(audit_path.read_text());stress=json.loads(stress_path.read_text())
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(study_path) or len(audit['cases'])!=4:raise ValueError('Independent source audit differs')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();comparisons=[]
    pairs=[('incumbent',r) for r in ['selected','threshold_raw','threshold_cal','iou_raw','iou_cal','iou_cal_nonempty']]+[('threshold_raw','iou_raw'),('iou_raw','iou_cal'),('threshold_cal','iou_cal'),('iou_cal','iou_cal_nonempty'),('threshold_raw','iou_cal_nonempty')]
    for left,right in pairs:comparisons.append(synchronized_comparison(study['cases'],left,right))
    evidence={}
    for role in ['selected','threshold_cal','iou_raw','iou_cal','iou_cal_nonempty']:
        rows=[c['comparisons'][role] for c in study['cases']];n=sum(r['paired_macro_gain']>=.01 and r['ci95'] is not None and r['ci95'][0]>0 for r in rows);bounded=all(r['paired_macro_gain']>=-.01 for r in rows)
        evidence[role]={'cases_gain_01_positive_CI':n,'all_case_losses_bounded_01':bounded,'descriptive_gate_passed':n>=2 and bounded}
    cases=[]
    for case in study['cases']:
        hist={r:{} for r in ['incumbent','threshold_raw','threshold_cal','iou_raw','iou_cal','iou_cal_nonempty','selected']};seen=set()
        for item in case['source_records']:
            r=item['source_record']
            if r['phase']!='assessment':continue
            pred=next(p for p in case['prediction_records'] if p['phase']=='assessment' and p['seed']==r['seed'])
            if sha256_file(r['path'])!=r['sha256'] or sha256_file(pred['path'])!=pred['sha256']:raise ValueError('Summary cache changed')
            with np.load(r['path']) as a,np.load(pred['path']) as p:
                width=6*len(a['links'])
                for i,origin in enumerate(a['origins']):
                    if a['ongoing'][i] or str(origin) in seen:continue
                    seen.add(str(origin));ids=slice(i*width,(i+1)*width)
                    for role in hist:
                        size=str(int(p[role][ids].sum()));hist[role][size]=hist[role].get(size,0)+1
        cases.append({'name':case['name'],'feature_arm':case['feature_arm'],'calibration_states':case['calibration_states'],'decision_selected':case['decision_selected'],'selected':case['selected'],'n_unique_onset_origins':len(seen),'complete_domain_set_size_histograms':hist})
    checks=[scalar_stress(row) for row in stress['cases']]
    result={'source_study_sha256':sha256_file(study_path),'independent_audit_sha256':sha256_file(audit_path),'stress_study_sha256':sha256_file(stress_path),'source_snapshot':snapshot,'comparisons':comparisons,'evidence_gates':evidence,'cases':cases,
        'known_joint_stress_independent_checks':checks,'n_joint_contrasts':len(comparisons),'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,
        'scope':'Posthoc11contrasts,uncorrected multiple comparisons,not new selection. All joint-date bootstrap replicates independently replayed;all7toy distributions replayed with scalar set enumeration. No formal promotion or hidden-label claims.'}
    raw=json.dumps(result,indent=2,allow_nan=False)
    with args.output.open('x',encoding='utf-8') as f:f.write(raw)
    with (folder/'experiment.json').open('x',encoding='utf-8') as f:f.write(raw)
    print('probabilistic_summary_COMPLETE','contrasts',len(comparisons),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
