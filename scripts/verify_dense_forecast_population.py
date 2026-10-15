"""Independent shifted-window count audit for fit-only forecast inventory."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol
from dense_forecast_population import FIELDS


def values(path,times,links):
    f=pd.read_parquet(path);f['timestamp']=f.timestamp.astype(str);f['link_id']=f.link_id.astype(str);v=pd.to_numeric(f.speed_kmh,errors='coerce');q=pd.to_numeric(f.flow_vph,errors='coerce');coverage=pd.to_numeric(f.pct_observed,errors='coerce');good=(coverage>=75)&np.isfinite(v)&np.isfinite(q);f['_eligible_speed']=v.where(good)
    speed=f.groupby(['timestamp','link_id'])['_eligible_speed'].mean().unstack('link_id').reindex(index=times,columns=links).to_numpy(float)
    return speed,np.isfinite(speed)


def shifted_counts(v,m,tmask,cut,selected_times,stride):
    q=(v<=cut)&m;at=np.arange(12,len(v)-6,stride);history=at[:,None]+np.arange(-12,0);good=m[history].mean(axis=(1,2))>=.7;accepted=at[good];histq=q[history[good]];future=accepted[:,None]+np.arange(1,7);fm=m[future];fq=q[future];ongoing=(histq.sum(axis=1)>=2).any(axis=1);positive=fq.any(axis=(1,2));covered=np.zeros(len(v),bool);covered[np.unique(future)]=True;unique=covered[:,None]&m;selected=selected_times[:,None]&m
    drops=np.zeros(m.shape,bool);drops[1:]=m[1:]&m[:-1]&(np.diff(v,axis=0)<=-20);free=m&~q;isolated=q&np.roll(free,1,axis=0)&np.roll(free,-1,axis=0);isolated[[0,-1]]=False
    counts={'n_proposed_origins':len(at),'n_history_valid_origins':len(accepted),'n_eligible_target_occurrences':int(fm.sum()),'n_unique_eligible_target_cells':int(unique.sum()),'n_positive_target_occurrences':int(fq.sum()),'n_unique_positive_target_cells':int((unique&q).sum()),
        'n_history_no_observed_queue':int((~ongoing).sum()),'n_future_observed_queue_windows':int(positive.sum()),'n_history_free_with_future_queue':int((~ongoing&positive).sum()),'n_last_step_only_onset_proxy':int((~ongoing&~fq[:,:5].any(axis=(1,2))&fq[:,5].any(axis=1)).sum()),
        'n_future_observed_queue_empty_windows':int((~positive).sum()),'n_future_no_evidence_windows':int((~fm.any(axis=(1,2))).sum()),'n_current_masked_eligible_cells':int(tmask[accepted].sum()),'n_current_input_cells':len(accepted)*v.shape[1],
        'n_unique_adjacent_speed_drops20':int((unique&drops).sum()),'n_unique_isolated_observed_queue_cells':int((unique&isolated).sum()),'n_selected_target_cells':int(selected.sum()),'n_selected_targets_in_dense_union':int((selected&unique).sum())}
    if counts.keys()!=dict.fromkeys(FIELDS).keys():raise ValueError('Inventory statistic schema changed')
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite dense-population audit')
    source_path=Path('reports/research/C85_dense_forecast_fit_population.json');study=json.loads(source_path.read_text());registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_dense_forecast_fit_inventory_audit','source_study_sha256':sha256_file(source_path),
        'scope':'Rebuild raw/masked arrays with direct public eligibility predicate/groupby;shifted vectorized history/future windows and union-of-times counts independently verify every daily/aggregate statistic. No model fit or hidden/assessment data.', 'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public');rows=integer_checks=0;cases=[]
    for path,digest in study['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Inventory public source changed')
    for case in study['cases']:
        panel=case['panel'];fit_record=case['selected_fit_record']
        if sha256_file(fit_record['path'])!=fit_record['sha256']:raise ValueError('Selected training reference changed')
        fit=dict(np.load(fit_record['path']));links=fit['links'].tolist();cut=release.network(panel).v_cut.reindex(links).to_numpy(float);raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};masked_paths={partition_date(release,p):p for p in release.files(panel,'train')};origins=pd.to_datetime(fit['origins'],utc=True);sums={str(s):{k:0 for k in FIELDS} for s in [5,30]}
        for date in case['fit_dates']:
            times=[t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in pd.date_range(pd.Timestamp(date,tz='UTC'),periods=288,freq='5min')];v,m=values(raw_paths[date],times,links);_,current=values(masked_paths[date],times,links);selected=np.zeros(288,bool)
            for origin in origins[origins.strftime('%Y-%m-%d')==date]:
                at=int((origin-pd.Timestamp(date,tz='UTC'))/pd.Timedelta(minutes=5));selected[at+1:at+7]=True
            for stride in [1,6]:
                measured=shifted_counts(v,m,current,cut,selected,stride);row=next(r for r in case['daily'] if r['date']==date and r['stride_minutes']==5*stride)
                if any(measured[k]!=row[k] for k in FIELDS):raise ValueError('Independent daily statistic differs: '+case['name']+' '+date)
                for k in FIELDS:sums[str(5*stride)][k]+=measured[k]
                rows+=1;integer_checks+=len(FIELDS)
        for stride,counts in sums.items():
            expected=case['summaries'][stride]
            if any(counts[k]!=expected[k] for k in FIELDS):raise ValueError('Independent population aggregation differs')
            ratios={'unique_teacher_expansion_vs_selected':counts['n_unique_eligible_target_cells']/max(1,counts['n_selected_target_cells']),'target_occurrences_per_unique_cell':counts['n_eligible_target_occurrences']/max(1,counts['n_unique_eligible_target_cells']),'current_masked_coverage':counts['n_current_masked_eligible_cells']/max(1,counts['n_current_input_cells'])}
            if any(abs(value-expected[key])>1e-12 for key,value in ratios.items()):raise ValueError('Population ratio differs')
        if sums['5']['n_selected_target_cells']!=int(fit['available'].sum()) or sums['5']['n_selected_targets_in_dense_union']!=int(fit['available'].sum()):raise ValueError('Dense targets fail to include selected training targets')
        cases.append({'name':case['name'],'all_daily_stride_integer_counts_aggregate_ratios_and_selected_union_verified':True,'n_fit_days':len(case['fit_dates'])});print('dense_population_verified_case',case['name'],flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'n_daily_stride_rows_independently_checked':rows,'n_daily_integer_checks':integer_checks,'n_unique_source_files_rehashed':len(study['source_files_sha256']),'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,
        'scope':'Full FIT-only raw/masked inventory replay. Verified data counts,not additional independent labels or future model gain;no selection/assessment/private data or model fitting.'};raw=json.dumps(result,indent=2,allow_nan=False)
    with args.output.open('x',encoding='utf-8') as f:f.write(raw)
    with (folder/'verification.json').open('x',encoding='utf-8') as f:f.write(raw)
    print('dense_population_verification_COMPLETE','rows',rows,'integer_checks',integer_checks,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
