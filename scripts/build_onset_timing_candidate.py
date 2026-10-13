"""Queue-only control using a timing property explicitly published by the host."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release,csv,QUEUE_KEY
from trafficflow.submission import merge_files,validate_task
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite a controlled candidate')
    audit=json.loads(Path('reports/research/C23_released_onset_timing_audit.json').read_text())
    release=Release('data/kaggle_public')
    source=Path('artifacts/submission_best')
    if sha256_file(source/'queue.csv')!='420080d4582ff74f96a33bc5ca726aedb541ca3f576ced416be307d6c22d3fa4':
        raise ValueError('Original Queue component changed')
    hypothesis={'id':'K02_Queue_only_public_onset_timing','baseline_ref':57138874,'baseline_public_total':.82940,
        'baseline_sha256':'43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1',
        'changed_component':'Queue only: existing positive predictions at onset T+5..T+25 become zero',
        'unchanged':'All State/OD/Physics inputs, ongoing forecasts and onset T+30 forecasts',
        'public_authorization_source':'https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350',
        'host_metadata_evidence':'reports/research/onset_topic_author_verification.json',
        'public_train_audit':'reports/research/C23_released_onset_timing_audit.json',
        'hypothesis':'The published empty first five onset steps allow removing false positives without changing the predicted set at T+30.',
        'expected_formal_total_delta':None,'formal_components_unknown':True,
        'inference_inputs':'Public window origins/conditions and our own frozen predictions only',
        'hidden_labels_or_future_measurements_read':False,'models_retrained':False,
        'validation_private_labels':'Unobserved. Public timing rule used as published, not inferred from hidden values.',
        'earlier_random_onset_validation':'C01-C22 are general-event stress tests; onset distribution not aligned with published sampling.'}
    register_protocol(Path('reports/research/K02_Queue_only_public_timing_preregistered.json'),hypothesis)
    args.output.mkdir(parents=True)
    snapshot=capture_source_snapshot(args.output/'source_snapshot.zip')
    start=time.perf_counter()
    queue=csv(source/'queue.csv')
    metadata=[]
    for panel in release.panels():
        if panel.startswith('D12_I405'):continue
        for split in ('validation','private'):
            index=csv(release.root/'task2'/panel/split/'window_index.csv')
            metadata.append(index[['window_id','forecast_origin','condition']])
    metadata=pd.concat(metadata,ignore_index=True)
    if metadata.window_id.duplicated().any():raise ValueError('Window IDs must be globally unique')
    joined=queue.merge(metadata,on='window_id',how='left',validate='many_to_one',sort=False)
    if joined.condition.isna().any():raise ValueError('Unknown Queue window')
    lead=(pd.to_datetime(joined.timestamp,utc=True)-pd.to_datetime(joined.forecast_origin,utc=True)).dt.total_seconds()/60
    if not lead.isin([5,10,15,20,25,30]).all():raise ValueError('Unexpected horizon metadata')
    old=pd.to_numeric(joined.queue_pred).to_numpy()
    remove=(joined.condition=='queue_onset')&(lead<30)&(old==1)
    joined.loc[remove,'queue_pred']='0'
    modified=joined[QUEUE_KEY+['queue_pred']]
    validate_task(modified,'queue')
    queue_path=args.output/'queue.csv'
    modified.to_csv(queue_path,index=False)
    if int(remove.sum())!=audit['total_early_positive_predictions_in_own_submission']:
        raise ValueError('Candidate differs from independently audited change set')
    files={'state':source/'state.csv','queue':queue_path,'odme':source/'odme.csv'}
    report=merge_files(files,release.root/'submission_key.csv',args.output/'submission.csv')
    report.update({'hypothesis':hypothesis,'source_snapshot':snapshot,'early_ones_changed_to_zero':int(remove.sum()),
        'component_sha256':{k:sha256_file(p) for k,p in files.items()},
        'submission_sha256':sha256_file(args.output/'submission.csv'),
        'seconds':time.perf_counter()-start,'formal_public_total':None,'uploaded':False})
    if report['component_sha256']['state']!='49c48651db680a46a05f952ae4a89001374ea5912d73627d9a2ad8117e53a294':
        raise ValueError('Original State component changed')
    (args.output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report['submission_sha256'],report['early_ones_changed_to_zero'],report['seconds'],flush=True)


if __name__=='__main__':main()
