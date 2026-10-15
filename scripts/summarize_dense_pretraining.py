"""Posthoc matched dense warm-up diagnostics; no fitting or model selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.runtime import sha256_file,capture_source_snapshot
from summarize_rolling_onset_decoders import synchronized_comparison
from verify_onset_rank_queue import manual_baseline,manual_metric,same_metric,paired,same_paired
from compare_queue_events import comparison


ARMS=['scratch','event_warm','dense_warm'];SEEDS=[20261013,20261029]


def checked(record):
    if sha256_file(record['path'])!=record['sha256']:raise ValueError('Frozen diagnostic input changed')
    return dict(np.load(record['path']))


def diagnostic_rows(a,p,v,metric):
    rows=[];by_window={r['window']:r for r in metric['windows']}
    for i,origin in enumerate(a['origins']):
        domain=a['available'][i].copy()
        if not a['ongoing'][i]:domain[:5]=False
        condition='queue_ongoing' if a['ongoing'][i] else 'queue_onset';window=str(a['window'][i*6*len(a['links'])]);row=by_window[window]
        mask=a['available'][i];error=(v[i].astype(float)-a['truth_speed'][i].astype(float))*25.
        rows.append({'origin':str(origin),'condition':condition,'IoU':row['IoU'],
            'NN_corrected_score_Brier':float(np.mean((p[i][domain]-a['labels'][i][domain])**2)),
            'NN_speed_forecast_RMSE_kmh':float(np.sqrt(np.mean(error[mask]**2))),
            'classification_teacher_cells':int(domain.sum()),'speed_teacher_cells':int(mask.sum())})
    return rows


def summarize(rows):
    frame=pd.DataFrame(rows)
    for _,g in frame.groupby('origin'):
        if g.condition.nunique()!=1 or not np.allclose(g[['IoU','NN_corrected_score_Brier','NN_speed_forecast_RMSE_kmh']],g[['IoU','NN_corrected_score_Brier','NN_speed_forecast_RMSE_kmh']].iloc[0],atol=1e-12,rtol=0):raise ValueError('Repeated diagnostic origin differs')
    frame=frame.drop_duplicates('origin');keys=['IoU','NN_corrected_score_Brier','NN_speed_forecast_RMSE_kmh'];conditions=frame.groupby('condition')[keys].mean()
    return {'n_unique_origins':len(frame),'condition_origin_counts':frame.condition.value_counts().to_dict(),'condition_means':conditions.to_dict(orient='index'),
        'equal_condition_window_macro':conditions.mean().to_dict()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite dense warm-up summary')
    study_path=Path('reports/research/C87_dense_dynamics_pretraining.json');study=json.loads(study_path.read_text());audit=json.loads(args.audit.read_text())
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(study_path):raise ValueError('Independent model audit bound to different study')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();cases=[];member_cases=[]
    comparisons=[synchronized_comparison(study['cases'],a,b) for a,b in [('incumbent','selected'),('incumbent','scratch'),('incumbent','event_warm'),('incumbent','dense_warm'),('scratch','event_warm'),('scratch','dense_warm'),('event_warm','dense_warm')]]
    for case in study['cases']:
        proof=next(c for c in audit['cases'] if c['name']==case['name']);rows={arm+'_'+str(seed):[] for arm in ARMS for seed in SEEDS};rows.update({arm+'_mean':[] for arm in ARMS});runs=[];source_records=[]
        for r in case['event_records']:
            if r['phase']!='assessment':continue
            a=checked(r);pr=next(q for q in case['prediction_records'] if q['phase']=='assessment' and q['seed']==r['seed']);pred=checked(pr);mr=next(q for q in proof['prediction_audits'] if q['phase']=='assessment' and q['seed']==r['seed']);members=checked(mr);metrics={}
            for arm in ARMS:
                for label in [str(seed) for seed in SEEDS]+['mean']:
                    p=members[arm+'_'+label+'_probability'] if label!='mean' else np.mean([members[arm+'_'+str(seed)+'_probability'] for seed in SEEDS],axis=0)
                    v=members[arm+'_'+label+'_speed'] if label!='mean' else np.mean([members[arm+'_'+str(seed)+'_speed'] for seed in SEEDS],axis=0)
                    binary=manual_baseline(a,.75*pred['tree_probability']+.25*p.ravel(),case['choices'][arm]['setting']);metric=manual_metric(a,binary);key=arm+'_'+label;metrics[key]=metric;rows[key].extend(diagnostic_rows(a,p,v,metric))
                    if label=='mean':same_metric(metric,next(q for q in case['assessment'] if q['seed']==r['seed'])['models'][arm])
            runs.append({'seed':r['seed'],'models':metrics});source_records.append({'event':r,'prediction':pr,'independent_members':mr})
        pseudo={'name':case['name'],'dates':case['dates'],'assessment':runs};member_cases.append(pseudo);matched=[]
        for baseline,candidate in [('scratch_mean','event_warm_mean'),('scratch_mean','dense_warm_mean'),('event_warm_mean','dense_warm_mean')]:
            independent=paired(runs,case['dates']['assessment'],baseline,candidate);original=comparison(pseudo,baseline,candidate);same_paired(independent,original);matched.append({'baseline':baseline,'candidate':candidate,**independent})
        exposure=[]
        for seed in SEEDS:
            training={t['arm']:t for t in case['training'] if t['seed']==seed};ev=training['event_warm']['warmup_exposure'];de=training['dense_warm']['warmup_exposure']
            exposure.append({'seed':seed,'warmup_event':ev,'warmup_dense':de,'dense_over_event_unique_speed_cells':de['unique_regression_teacher_cells']/ev['unique_regression_teacher_cells'],
                'same_finetune_exposure_across_arms':all(t['finetune_exposure']==training['scratch']['finetune_exposure'] for t in training.values()),
                'training_seconds':{k:t['seconds'] for k,t in training.items()},'selected_steps':{k:t['selected']['step'] for k,t in training.items()}})
        cases.append({'name':case['name'],'panel':case['panel'],'decision_selected':case['decision_selected'],'selected':case['selected'],'selection_gate':case['selection_gate'],
            'assessment_diagnostics':{k:summarize(v) for k,v in rows.items()},'matched_warmup_diagnostics':matched,'actual_exposures':exposure,'source_records':source_records})
    seed_comparisons=[synchronized_comparison(member_cases,a+'_'+str(seed),b+'_'+str(seed)) for seed in SEEDS for a,b in [('scratch','dense_warm'),('event_warm','dense_warm')]]
    result={'source_study_sha256':sha256_file(study_path),'independent_audit_sha256':sha256_file(args.audit),'source_snapshot':snapshot,'joint_comparisons':comparisons,'cases':cases,
        'single_member_comparisons_frozen_mean_thresholds':seed_comparisons,'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'Posthoc previously-viewed public noisy observation proxy only. Shared calendar block draws synchronized across panels,origin dedup,conditions/cases equal weight;conditional on fitted models,uncorrected intervals. Members use frozen arm-mean decision thresholds without retuning;never primary choices. NN Brier is analytic weighted-BCE corrected score,not a guaranteed calibrated posterior. NN speed forecast errors are future speed teacher diagnostics,not official State,latent traffic or Physics. No fitting,new selection,promotion or formal submission.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'experiment.json').write_text(raw,encoding='utf-8');print('dense_pretrain_summary_COMPLETE','comparisons',len(comparisons),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
