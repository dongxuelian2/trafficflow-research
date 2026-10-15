"""Post-assessment cardinality headroom diagnostic, never a predictor.

An evaluator-label oracle chooses among already defined k values. It is not
exported, never enters a selection gate, and cannot be used on hidden windows.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.queue_domain import score_full_forecast
from onset_rank_queue_experiment import apply_onset,KGRID


def deduplicated_macro(rows):
    frame=pd.DataFrame(rows)
    frame['origin']=frame.window.map(lambda w:w.rsplit(':',1)[0])
    for _,g in frame.groupby('origin'):
        if g.condition.nunique()!=1 or np.max(g.IoU)-np.min(g.IoU)>1e-12:raise ValueError('Repeated origin diagnostic differs')
    unique=frame.drop_duplicates('origin');conditions=unique.groupby('condition').IoU.mean().to_dict()
    return {'macro':float(np.mean(list(conditions.values()))),'conditions':conditions,'n_unique_origins':len(unique)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite onset cardinality diagnostics')
    path=Path('reports/research/C65_onset_query_rank_public.json');audit_path=Path('reports/research/C66_onset_query_rank_public_verification.json')
    study=json.loads(path.read_text());audit=json.loads(audit_path.read_text())
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(path) or len(audit['cases'])!=4:raise ValueError('Four completed independent case audits required')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();cases=[]
    for case in study['cases']:
        with np.load(case['training_rows']['path']) as a:
            counts=[];at=0
            for size in a['groups']:counts.append(int(a['labels'][at:at+size].sum()));at+=size
        rows={'incumbent':[],'classifier':[],'ranker':[],'classifier_label_oracle':[],'ranker_label_oracle':[]};chosen_sizes={'classifier':[],'ranker':[]}
        for item in case['source_records']:
            record=item['source_record']
            if record['phase']!='assessment':continue
            if sha256_file(record['path'])!=record['sha256']:raise ValueError('Assessment source changed')
            data=dict(np.load(record['path']));prediction_record=next(r for r in case['score_records'] if r['phase']=='assessment' and r['seed']==record['seed'])
            if sha256_file(prediction_record['path'])!=prediction_record['sha256']:raise ValueError('Frozen assessment prediction changed')
            with np.load(prediction_record['path']) as p:
                for role in ['incumbent','classifier','ranker']:
                    rows[role]+=score_full_forecast(data['labels'].ravel(),p[role],data['window'],data['condition'],data['available'].ravel())['windows']
                for kind in ['classifier','ranker']:
                    by_k={}
                    for k in KGRID:
                        pred=apply_onset(data,p[kind+'_score'],k,p['incumbent'])
                        by_k[k]={r['window']:r for r in score_full_forecast(data['labels'].ravel(),pred,data['window'],data['condition'],data['available'].ravel())['windows']}
                    for window in by_k[KGRID[0]]:
                        best=max(KGRID,key=lambda k:(by_k[k][window]['IoU'],-k));row=by_k[best][window]
                        rows[kind+'_label_oracle'].append(row)
                        if row['condition']=='queue_onset':chosen_sizes[kind].append({'window':window,'k':best})
        results={key:deduplicated_macro(values) for key,values in rows.items()}
        expected=case['comparisons']['selected']['origin_deduplicated_condition_macro_IoU']['baseline']
        if abs(results['incumbent']['macro']-expected)>1e-12:raise ValueError('Original proxy reference differs')
        histograms={}
        for kind,values in chosen_sizes.items():
            frame=pd.DataFrame(values);frame['origin']=frame.window.map(lambda w:w.rsplit(':',1)[0])
            if frame.groupby('origin').k.nunique().max()>1:raise ValueError('Repeated origin oracle size differs')
            histograms[kind]={str(int(k)):int(n) for k,n in frame.drop_duplicates('origin').k.value_counts().sort_index().items()}
        cases.append({'source_study':case['source_study'],'observed_fit_onset_positive_counts':{'queries':len(counts),'min':min(counts),'median':float(np.median(counts)),'q25':float(np.quantile(counts,.25)),'q75':float(np.quantile(counts,.75)),'max':max(counts)},
            'assessment_proxies':results,'label_oracle_best_k_unique_origin_counts':histograms,
            'original_decision_selected':case['decision_selected'],'original_selected':case['selected'],'selection_changed':False})
    result={'source_study_sha256':sha256_file(path),'independent_audit_sha256':sha256_file(audit_path),'source_snapshot':snapshot,'cases':cases,
        'uses_public_assessment_labels':True,'deployment_allowed':False,'new_model_or_candidate':False,'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,
        'scope':'Optimistic per-window best k among five fixed sets using public assessment labels and eligibility. Scoring diagnostic only,not a legal future decoder or achievable score. No new fitting,selection,hidden-label use or submission.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'experiment.json').write_text(raw,encoding='utf-8')
    print('onset_rank_diagnostic','cases',len(cases),'selection_changed',False,flush=True)


if __name__=='__main__':main()
