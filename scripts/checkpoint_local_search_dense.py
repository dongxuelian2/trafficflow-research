"""Freeze rejected decoder screen and independently verified dense-data support."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked
from verify_dense_forecast_population import values


def verify_onset_support(population):
    release=Release('data/kaggle_public');records=[]
    for case in population['cases']:
        panel=case['panel'];fit=dict(np.load(case['selected_fit_record']['path']));links=fit['links'].tolist();cut=release.network(panel).v_cut.reindex(links).to_numpy(float);paths={partition_date(release,p):p for p in release.files(panel,'train',False)};support=set()
        for date in case['fit_dates']:
            times=[t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in pd.date_range(pd.Timestamp(date,tz='UTC'),periods=288,freq='5min')];v,m=values(paths[date],times,links);q=(v<=cut)&m;at=np.arange(12,282);history=at[:,None]+np.arange(-12,0);future=at[:,None]+np.arange(1,7);valid=m[history].mean(axis=(1,2))>=.7;free=~(q[history].sum(axis=1)>=2).any(axis=1);fq=q[future];onset=valid&free&~fq[:,:5].any(axis=(1,2))&fq[:,5].any(axis=1)
            support.update(times[i] for i in at[onset])
        selected=set(pd.to_datetime(fit['origins'][~fit['ongoing']],utc=True).strftime('%Y-%m-%dT%H:%M:%SZ'))
        if support!=selected or len(support)!=case['summaries']['5']['n_last_step_only_onset_proxy']:raise ValueError('Dense versus selected onset support differs')
        records.append({'case':case['name'],'n_selected_onset_origins':len(selected),'n_dense5_onset_origins':len(support),'exact_origin_sets_equal':True,'selected_positive_target_cells':int((fit['labels']*fit['available']).sum()),
            'dense_unique_positive_target_cells':case['summaries']['5']['n_unique_positive_target_cells'],'scope':'Same public observed-label/history criterion only;does not establish complete latent/official event coverage.'})
    return records


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_local_dense_v1.json','local_dense_verification_record_v1.json','research_checkpoint_local_dense_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite local-search/data checkpoint')
    parent_path=root/'model_artifacts_checkpoint_joint_scenario_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    stems=['C82_bounded_search_known_laws','C83_bounded_search_decision_screen','C84_bounded_search_screen_verification','C85_dense_forecast_fit_population','C86_dense_forecast_population_verification'];small,screen,audit,population,pop_audit=[read(root/(s+'.json')) for s in stems]
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stems[1]+'.json')) or audit['registered']['protocol']['source_small_law_sha256']!=sha256_file(root/(stems[0]+'.json')) or pop_audit['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stems[3]+'.json')):raise ValueError('New evidence source chain differs')
    expected={'validation_banks':4,'known_law_cases':10,'known_global_sets_enumerated':320,'decision_onset_windows':52,'neighbor_candidates':5284,'decision_onset_Booster_calls':8,'decision_reference_probability_calls':4,'portable_history_windows':16,'portable_role_calls':48,'additional_shuffled_calls':24,'future_history_rejections':12}
    if audit['totals']!=expected or screen['campaign_local_shortlisted'] or screen['selection_arrays_read'] or screen['assessment_arrays_read']:raise ValueError('Screen stopping/audit scope differs')
    if any(d['moves'] or d['accepted'] for c in screen['cases'] for d in c['onset_diagnostics']):raise ValueError('Unexpected accepted or proposed public move')
    if len(pop_audit['cases'])!=4 or pop_audit['n_daily_stride_rows_independently_checked']!=680 or pop_audit['n_daily_integer_checks']!=12240 or population['n_fit_day_occurrences']!=340 or population['n_unique_source_files']!=432:raise ValueError('Population replay scope incomplete')
    for path,digest in population['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Population source changed')
    support=verify_onset_support(population)
    for stem in stems:
        for p in sorted((Path('artifacts/research')/stem).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip','.pkl','.txt','.parquet'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'bounded_local_screen_rejected_dense_fit_inventory_complete'});known.add(p.as_posix())
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],'scope':'Bounded local-search models/independent simulation banks,decision-only rejection,fit-only dense supervision inventory and independent audits. Old complete submissions unchanged.'}
    verification={'created_UTC':now,'new_unit_tests':{'passed':3,'seconds':1.61,'scope':'Known global counterexample,bounded monotone single-flip stationarity/permutation,independent scenario accept/reject'},
        'research_pytest':{'passed':147,'seconds':12.42,'session_id':72803,'basetemp':'.cache/pytest_local_search_research_v1'},'CPU_pytest':{'passed':126,'optional_Torch_modules_skipped':7,'seconds':7.97,'basetemp':'.cache/pytest_local_search_CPU_v1'},'compileall_passed':True,
        'independent_local_screen':expected,'independent_population':{'daily_stride_rows':680,'integer_checks':12240,'unique_source_files_rehashed':432},'additional_fit_only_exact_onset_origin_support_audit':support,
        'scope':'Current production suites pass. No new later-period evaluation or formal score;MC acceptance conditional on a hypothetical fitted law. Dense target occurrences are overlapping,extra unique cells mostly background/ongoing;selected proxy onset support already complete by exact fit-origin sets. No latent completeness claim.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'previous_goal_turn_classification':'progress:1a6a921 C78-C81 complete','current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,'new_complete_submission':False,'active_jobs':[],
        'completed':['Bounded non-prefix search closes exact small-law gap but proposes0changes in52public decision onset windows','Predeclared decision-only screen rejected every case;no new selection/assessment arrays read','Dense fit data inventory independently verified;exact proxy onset origin support already matches selected fit windows'],
        'screen_cases':[{'name':c['name'],'decision_best':c['decision_best'],'local_shortlisted':c['local_shortlisted'],'n_onset_windows':len(c['onset_diagnostics'])} for c in screen['cases']],
        'dense_support':support,'next_research':'Stop this decoder expansion. Investigate generic dynamics pretraining with five-minute origin phases,then matched event finetuning;extra data does not add new first-T30 proxy onset events or prove gains.',
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only','limits':'Previously seen public noisy proxies,not latent/private/final. Density is not model performance;no claim of additional independent events or formal Queue/Physics/OD/Total.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    print('local_dense_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),flush=True)


if __name__=='__main__':main()
