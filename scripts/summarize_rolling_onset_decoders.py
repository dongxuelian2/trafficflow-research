"""Posthoc matched summary with calendar-synchronized public-proxy intervals."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.runtime import sha256_file,capture_source_snapshot


def window_table(case,baseline,candidate):
    rows=[]
    for run in case['assessment']:
        right={r['window']:r for r in run['models'][candidate]['windows']}
        for r in run['models'][baseline]['windows']:
            other=right[r['window']];origin=r['window'].rsplit(':',1)[0]
            rows.append({'origin':origin,'date':pd.Timestamp(origin.split(':',1)[1]).date().isoformat(),'condition':r['condition'],'a':r['IoU'],'b':other['IoU']})
    frame=pd.DataFrame(rows)
    for _,g in frame.groupby('origin'):
        if g.condition.nunique()!=1 or not np.allclose(g[['a','b']],g[['a','b']].iloc[0],atol=1e-12,rtol=0):raise ValueError('Repeated origin outcomes differ')
    return frame.drop_duplicates('origin')


def synchronized_comparison(cases,baseline,candidate):
    """Share date draws between panels with the SAME assessment calendar."""
    groups={};stats=[];case_values=[]
    for i,case in enumerate(cases):
        days=case['dates']['assessment'];groups.setdefault(tuple(days),[]).append(i);frame=window_table(case,baseline,candidate)
        arrays={k:np.zeros((len(days),2)) for k in ['count','left','right']}
        for row in frame.itertuples():
            d=days.index(row.date);c=['queue_onset','queue_ongoing'].index(row.condition);arrays['count'][d,c]+=1;arrays['left'][d,c]+=row.a;arrays['right'][d,c]+=row.b
        count=arrays['count'].sum(axis=0)
        if not (count>0).all():raise ValueError('Both conditions required per case')
        case_values.append({'case':case['name'],'baseline':float((arrays['left'].sum(axis=0)/count).mean()),'candidate':float((arrays['right'].sum(axis=0)/count).mean()),'unique_origins':len(frame)})
        stats.append(arrays)
    # This diagnostic assumes the two different rolling calendars are disjoint.
    keys=list(groups)
    if any(set(a)&set(b) for i,a in enumerate(keys) for b in keys[i+1:]):raise ValueError('Overlapping calendars need a different joint bootstrap')
    rng=np.random.default_rng(20261015);indices={};case_samples=np.full((4000,len(cases)),np.nan)
    for calendar,case_ids in groups.items():
        n=len(calendar);starts=rng.integers(0,n,(4000,int(np.ceil(n/7))));ix=((starts[:,:,None]+np.arange(7))%n).reshape(4000,-1)[:,:n]
        for i in case_ids:
            indices[i]=ix;count=stats[i]['count'][ix].sum(axis=1);valid=(count>0).all(axis=1);delta=(stats[i]['right'][ix]-stats[i]['left'][ix]).sum(axis=1)
            case_samples[valid,i]=(delta[valid]/count[valid]).mean(axis=1)
    valid=np.isfinite(case_samples).all(axis=1);samples=case_samples[valid].mean(axis=1);reference=[]
    # Separate scalar implementation verifies every retained bootstrap replicate.
    for draw in np.flatnonzero(valid):
        gains=[]
        for i,arrays in enumerate(stats):
            ids=indices[i][draw];count=np.sum(arrays['count'][ids],axis=0)
            gains.append(float(np.mean((np.sum(arrays['right'][ids],axis=0)-np.sum(arrays['left'][ids],axis=0))/count)))
        reference.append(float(np.mean(gains)))
    maximum=float(np.max(np.abs(samples-reference))) if len(samples) else 0.
    if maximum>1e-12:raise ValueError('Independent synchronized bootstrap differs')
    values={k:float(np.mean([r[k] for r in case_values])) for k in ['baseline','candidate']}
    return {'baseline':baseline,'candidate':candidate,'equal_case_condition_macro_IoU':values,'paired_macro_gain':values['candidate']-values['baseline'],
        'ci95':np.quantile(samples,[.025,.975]).tolist() if len(samples) else None,'probability_gain_positive':float((samples>0).mean()) if len(samples) else None,
        'cases':case_values,'n_cases':len(cases),'calendar_groups':[{'first':k[0],'last':k[-1],'n_days':len(k),'case_indices':v} for k,v in groups.items()],
        'resamples':4000,'valid_resamples':int(valid.sum()),'block_days':7,'seed':20261015,'independent_replicate_maximum_difference':maximum,
        'scope':'Posthoc descriptive equal-case/equal-condition proxy. Same calendar draws synchronized across panels;disjoint periods resampled separately. Zero-event days retained. Conditional on these fitted models;not training-seed uncertainty,independent case replications,formal Queue/Total or a promotion rule.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite rolling summary')
    study=json.loads(args.study.read_text());audit=json.loads(args.audit.read_text())
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(args.study) or len(audit['cases'])!=len(study['cases']):raise ValueError('Independent audit source differs')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();comparisons=[]
    for baseline,candidate in [('incumbent','selected'),('incumbent','threshold'),('incumbent','fixed_k'),('fixed_k','threshold')]:
        comparisons.append(synchronized_comparison(study['cases'],baseline,candidate))
    cases=[]
    for case in study['cases']:
        conditions={role:window_table(case,'incumbent',role).groupby('condition')[['a','b']].mean().to_dict() for role in ['fixed_k','threshold','selected']};hist={role:{} for role in ['incumbent','fixed_k','threshold','selected']};seen=set()
        for record,pred_row in zip([r for r in case['cache_records'] if r['phase']=='assessment'],[r for r in case['prediction_records'] if r['phase']=='assessment']):
            if record['seed']!=pred_row['seed'] or sha256_file(record['path'])!=record['sha256'] or sha256_file(pred_row['path'])!=pred_row['sha256']:raise ValueError('Summary prediction/cache binding differs')
            with np.load(record['path']) as a,np.load(pred_row['path']) as p:
                width=6*len(a['links'])
                for i,origin in enumerate(a['origins']):
                    if a['ongoing'][i] or str(origin) in seen:continue
                    seen.add(str(origin));ids=slice(i*width,(i+1)*width)
                    for role in hist:
                        value=str(int(p[role][ids].sum()));hist[role][value]=hist[role].get(value,0)+1
        cases.append({'name':case['name'],'panel':case['panel'],'feature_arm':case['feature_arm'],'decision_selected':case['decision_selected'],'selected':case['selected'],
            'assessment_origin_deduplicated_conditions':conditions,'onset_complete_domain_size_histograms':hist,'n_unique_onset_origins':len(seen)})
    result={'source_study_sha256':sha256_file(args.study),'independent_audit_sha256':sha256_file(args.audit),'source_snapshot':snapshot,'comparisons':comparisons,'cases':cases,
        'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,'scope':'Posthoc summary of frozen registered rolling refits;no new model/parameter/policy selection or formal promotion. Four joint intervals independently checked over all valid replicates.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'experiment.json').write_text(raw,encoding='utf-8');print('rolling_summary_COMPLETE','comparisons',len(comparisons),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
