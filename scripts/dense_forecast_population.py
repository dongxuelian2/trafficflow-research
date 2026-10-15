"""Fit-only all-time causal forecast supervision inventory; no model fitting."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release,eligible
from trafficflow.experiments import partition_date
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol


FIELDS=['n_proposed_origins','n_history_valid_origins','n_eligible_target_occurrences','n_unique_eligible_target_cells','n_positive_target_occurrences','n_unique_positive_target_cells',
    'n_history_no_observed_queue','n_future_observed_queue_windows','n_history_free_with_future_queue','n_last_step_only_onset_proxy','n_future_observed_queue_empty_windows','n_future_no_evidence_windows',
    'n_current_masked_eligible_cells','n_current_input_cells','n_unique_adjacent_speed_drops20','n_unique_isolated_observed_queue_cells','n_selected_target_cells','n_selected_targets_in_dense_union']


def grid(frame,times,links):
    f=frame.copy();f.loc[~eligible(f),['speed_kmh','flow_vph']]=np.nan
    def values(column):return f.pivot_table(index='timestamp',columns='link_id',values=column).reindex(index=times,columns=links).to_numpy(float)
    speed=values('speed_kmh');return speed,np.isfinite(speed)


def count_day(speed,quality,current_quality,cutoff,selected_mask,stride):
    ntime,nodes=speed.shape;queue=(speed<=cutoff)&quality;visited=np.zeros(quality.shape,bool);counts={k:0 for k in FIELDS};isolated=np.zeros(quality.shape,bool)
    isolated[1:-1]=queue[1:-1]&~queue[:-2]&~queue[2:]&quality[:-2]&quality[2:]
    drops=np.zeros(quality.shape,bool);drops[1:]=quality[1:]&quality[:-1]&((speed[1:]-speed[:-1])<=-20)
    for at in range(12,ntime-6,stride):
        counts['n_proposed_origins']+=1
        if quality[at-12:at].mean()<.7:continue
        counts['n_history_valid_origins']+=1;future=quality[at+1:at+7];truth=queue[at+1:at+7];visited[at+1:at+7]|=future
        ongoing=bool((queue[at-12:at].sum(axis=0)>=2).any());positive=bool(truth.any())
        counts['n_eligible_target_occurrences']+=int(future.sum());counts['n_positive_target_occurrences']+=int(truth.sum());counts['n_history_no_observed_queue']+=int(not ongoing);counts['n_future_observed_queue_windows']+=int(positive)
        counts['n_history_free_with_future_queue']+=int(not ongoing and positive);counts['n_last_step_only_onset_proxy']+=int(not ongoing and not truth[:-1].any() and truth[-1].any());counts['n_future_observed_queue_empty_windows']+=int(not positive);counts['n_future_no_evidence_windows']+=int(not future.any())
        counts['n_current_masked_eligible_cells']+=int(current_quality[at].sum());counts['n_current_input_cells']+=nodes
    counts['n_unique_eligible_target_cells']=int(visited.sum());counts['n_unique_positive_target_cells']=int((visited&queue).sum());counts['n_unique_adjacent_speed_drops20']=int((visited&drops).sum());counts['n_unique_isolated_observed_queue_cells']=int((visited&isolated).sum());counts['n_selected_target_cells']=int(selected_mask.sum());counts['n_selected_targets_in_dense_union']=int((visited&selected_mask).sum())
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite dense-fit population audit')
    source_path=Path('reports/research/C71_rolling_onset_threshold_refits.json');source=json.loads(source_path.read_text());registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Fit_only_dense_forecast_population_inventory','source_study_sha256':sha256_file(source_path),
        'definition':'All C71 FIT days only;origins12..last-6 within UTC day,5/30minute stride,accept using past12slot eligible coverage>=.7 only. No future-positive/noise/label eligibility gate on origin sampling.',
        'input_contract':'Past12 raw eligible observations plus published maskedT. Inventory records maskedTcoverage;future raw observations are teacher diagnostics only. No model trained or score computed.',
        'targets':'All six5minute future horizons,both-required-channel eligibility. Repeated/unique target cells separately counted. Observed speed drops<=-20km/h and isolated5min queue are teacher diagnostics,not latent accidents.',
        'comparison':'Same-case selected C71 fit supervision union versus all-time dense union. Overlapping folds and temporal windows are not independent labels/events.',
        'scope':'Feasibility evidence for generic traffic-dynamics pretraining after postprocessing screen failure;not official validation/hidden truth. No selection/assessment/private data read.',
        'formal_Queue':None,'formal_Total':None,'platform':'Local only','final_simulator_seeds_generated':False})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public');hashes={};cases=[]
    for case in source['cases']:
        panel=case['panel'];net=release.network(panel);raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};masked_paths={partition_date(release,p):p for p in release.files(panel,'train')};fit_record=next(r for r in case['cache_records'] if r['phase']=='fit')
        if sha256_file(fit_record['path'])!=fit_record['sha256']:raise ValueError('Selected fit cache changed')
        fit=dict(np.load(fit_record['path']));links=fit['links'].tolist();cut=net.v_cut.reindex(links).to_numpy(float);origins=pd.to_datetime(fit['origins'],utc=True);daily=[]
        for date in case['dates']['fit']:
            rp,mp=raw_paths[date],masked_paths[date]
            for path in [rp,mp]:
                if path.as_posix() not in hashes:hashes[path.as_posix()]=sha256_file(path)
            raw=release.read(rp);masked=release.read(mp);times=sorted(raw.timestamp.unique())
            if len(times)!=288 or raw.link_id.drop_duplicates().tolist()!=links:raise ValueError('Daily grid differs from registered full-day law')
            speed,quality=grid(raw,times,links);_,current=grid(masked,times,links);selected=np.zeros(quality.shape,bool);ix=np.flatnonzero(origins.strftime('%Y-%m-%d')==date)
            for i in ix:
                token=origins[i].strftime('%Y-%m-%dT%H:%M:%SZ');at=times.index(token);future=quality[at+1:at+7]
                if not np.array_equal(future,fit['available'][i]):raise ValueError('Selected teacher eligibility changed')
                selected[at+1:at+7]|=future
            for stride in [1,6]:daily.append({'date':date,'stride_minutes':5*stride,**count_day(speed,quality,current,cut,selected,stride)})
        summaries={}
        for stride in [5,30]:
            rows=[r for r in daily if r['stride_minutes']==stride];counts={k:sum(r[k] for r in rows) for k in FIELDS};counts['unique_teacher_expansion_vs_selected']=counts['n_unique_eligible_target_cells']/max(1,counts['n_selected_target_cells']);counts['target_occurrences_per_unique_cell']=counts['n_eligible_target_occurrences']/max(1,counts['n_unique_eligible_target_cells']);counts['current_masked_coverage']=counts['n_current_masked_eligible_cells']/max(1,counts['n_current_input_cells']);summaries[str(stride)]=counts
        if summaries['5']['n_selected_targets_in_dense_union']!=int(fit['available'].sum()) or summaries['5']['n_selected_target_cells']!=int(fit['available'].sum()):raise ValueError('Dense5 union fails to contain same-case selected fit targets')
        record={'name':case['name'],'panel':panel,'fit_dates':case['dates']['fit'],'selected_fit_record':fit_record,'n_selected_fit_windows':len(fit['origins']),'n_selected_fit_supervised_cells':int(fit['available'].sum()),'n_nodes':len(links),'daily':daily,'summaries':summaries};cases.append(record);print('dense_forecast_population_case',case['name'],'unique_target_expansion',summaries['5']['unique_teacher_expansion_vs_selected'],flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'source_files_sha256':hashes,'n_fit_day_occurrences':sum(len(c['fit_dates']) for c in cases),'n_unique_source_files':len(hashes),'seconds':time.perf_counter()-start,'source':evidence(),'new_models':0,'formal_Queue':None,'formal_Total':None,
        'scope':'Fit-only data inventory. More occurrences/unique public noisy targets do not establish information gain or future model performance. No simulated final seeds or private/selection/assessment arrays used.'};raw=json.dumps(result,indent=2,allow_nan=False)
    with args.output.open('x',encoding='utf-8') as f:f.write(raw)
    with (folder/'experiment.json').open('x',encoding='utf-8') as f:f.write(raw)
    print('dense_forecast_population_COMPLETE','fit_day_occurrences',result['n_fit_day_occurrences'],'source_files',len(hashes),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
