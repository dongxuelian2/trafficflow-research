"""Matched frozen-score decoder contrasts and honest research promotion gate."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.runtime import sha256_file,capture_source_snapshot
from compare_queue_events import comparison
from verify_onset_rank_queue import paired,same_paired


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite onset decoder summary')
    path=Path('reports/research/C68_onset_threshold_cardinality.json');audit_path=Path('reports/research/C69_onset_threshold_cardinality_verification.json');study=json.loads(path.read_text());audit=json.loads(audit_path.read_text())
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(path) or len(audit['cases'])!=4:raise ValueError('Completed independent audit required')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();cases=[];n_checks=0
    for case in study['cases']:
        report={'dates':{'assessment':case['dates']['assessment']},'assessment':case['assessment']};contrasts={}
        for a,b in [('fixed_k','threshold'),('fixed_k','count_prior'),('count_prior','count_adjusted'),('count_observed','count_adjusted')]:
            c=comparison(report,a,b);same_paired(paired(report['assessment'],report['dates']['assessment'],a,b),c);contrasts[b+'_vs_'+a]=c;n_checks+=1
        rows=[]
        for item in case['source_records']:
            record=item['source_record']
            if record['phase']!='assessment':continue
            data=dict(np.load(record['path']));saved=next(r for r in case['prediction_records'] if r['phase']=='assessment' and r['seed']==record['seed'])
            if sha256_file(saved['path'])!=saved['sha256'] or sha256_file(record['path'])!=record['sha256']:raise ValueError('Verified predictions/source changed')
            with np.load(saved['path']) as p:
                for key in pd.unique(data['window']):
                    ids=np.flatnonzero(data['window']==key)
                    if data['condition'][ids[0]]!='queue_onset':continue
                    row={'origin':str(key).rsplit(':',1)[0]}
                    for role in ['fixed_k','threshold','count_prior','count_observed','count_adjusted']:row[role]=int(p[role][ids].sum())
                    rows.append(row)
        frame=pd.DataFrame(rows)
        for _,g in frame.groupby('origin'):
            if (g.drop(columns='origin').nunique()>1).any():raise ValueError('Repeated origin predicted size differs')
        unique=frame.drop_duplicates('origin');histograms={role:{str(int(k)):int(n) for k,n in unique[role].value_counts().sort_index().items()} for role in unique.columns if role!='origin'}
        cases.append({'source_study':case['source_study'],'decision_selected':case['decision_selected'],'selected':case['selected'],
            'matched_decoder_contrasts':contrasts,'full_domain_predicted_size_histograms_unique_origins':histograms,
            'selection_gain':case['selection_gate']['paired_macro_gain'],'selected_assessment_gain':case['comparisons']['selected']['paired_macro_gain'],
            'selected_assessment_ci95':case['comparisons']['selected']['ci95'],'selection_changed':False})
    def passes(role):
        results=[c['comparisons'][role] for c in study['cases']];positive=[r for r in results if r['paired_macro_gain']>=.01 and r['ci95'] is not None and r['ci95'][0]>0]
        return {'cases_with_gain_at_least_01_and_positive_CI':len(positive),'all_case_losses_bounded_01':all(r['paired_macro_gain']>=-.01 for r in results),
            'evidence_gate_passed':len(positive)>=2 and all(r['paired_macro_gain']>=-.01 for r in results),'scope':'Descriptive historical public proxy gate;not a statistical guarantee or formal Queue result.'}
    result={'source_study_sha256':sha256_file(path),'independent_audit_sha256':sha256_file(audit_path),'source_snapshot':snapshot,'cases':cases,
        'n_matched_contrasts_independently_verified':n_checks,'selected_policy_evidence_gate':passes('selected'),'threshold_secondary_evidence_gate':passes('threshold'),
        'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,'scope':'Post-fit matched decoder contrasts and legal prediction-size histograms,not extra tuning. Existing classifier weights fixed. New selection unchanged;multiple secondary comparisons uncorrected and folds share data.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'experiment.json').write_text(raw,encoding='utf-8')
    print('onset_decoder_summary','contrasts',n_checks,'primary_gate',result['selected_policy_evidence_gate']['evidence_gate_passed'],flush=True)


if __name__=='__main__':main()
