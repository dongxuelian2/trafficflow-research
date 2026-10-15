"""Freeze complete rolling study; preserve every old artifact and submission."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_rolling_onset_v1.json','rolling_onset_verification_record_v1.json','research_checkpoint_rolling_onset_v1.json']
    if any((root/name).exists() for name in names):raise FileExistsError('Never overwrite rolling checkpoint')
    parent_path=root/'model_artifacts_checkpoint_onset_set_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    stems=['C71_rolling_onset_threshold_refits','C72b_rolling_onset_threshold_verification','C73_rolling_onset_matched_summary'];study,audit,summary=[read(root/(name+'.json')) for name in stems]
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or summary['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or summary['independent_audit_sha256']!=sha256_file(root/(stems[1]+'.json')):raise ValueError('Report source chain changed')
    totals=audit['totals']
    if len(study['cases'])!=4 or len(audit['cases'])!=4 or len(summary['comparisons'])!=4 or study['new_classifier_models']!=24 or totals['new_training_models_checked']!=24:raise ValueError('Rolling case/model scope incomplete')
    if sum(len(c['cache_records']) for c in study['cases'])!=32 or not all(c['all_supervised_rows_capacity_schemas_grids_seals_gates_verified'] for c in audit['cases']):raise ValueError('Incomplete cache verification')
    if totals['feature_arm_window_rebuilds']!=2*totals['raw_window_occurrences'] or totals['direct_booster_calls']!=88 or totals['baseline_calibration_score_calls']!=1504:raise ValueError('Fresh input/independent calibration scope incomplete')
    expected_portable=4*sum(sum(min(2,n) for n in r['sampler']['selected_counts'].values()) for c in study['cases'] for r in c['cache_records'] if r['phase']!='fit')
    if totals['portable_window_calls']!=expected_portable or totals['additional_shuffled_calls']!=224 or totals['future_history_rejections']!=16:raise ValueError('Portable checks incomplete')
    if any(c['independent_replicate_maximum_difference']>1e-12 for c in summary['comparisons']):raise ValueError('Joint interval replay differs')
    for path,digest in audit['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Audited public data changed')
    for stem in stems+['C72_rolling_onset_threshold_verification']:
        for path in sorted((Path('artifacts/research')/stem).rglob('*')):
            if path.is_file() and path.suffix in {'.json','.npz','.zip','.pkl','.txt','.parquet'} and path.as_posix() not in known:
                artifacts.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),'phase':'rolling_onset_decoder_refits_complete'});known.add(path.as_posix())
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'Four rolling public-observation proxy refits:8strong baseline trees+16onset trees;all new input/fit/calibration/policy evidence. Existing complete submissions unchanged.'}
    verification={'created_UTC':now,'new_scientific_tests':{'passed':3,'seconds':2.54,'basetemp':'.cache/pytest_rolling_queue_v1','scope':'Synchronized opposite panel errors cancel;duplicate event seeds deduplicate;overlapping unmatched calendars rejected;missing/empty evaluator windows calibrated independently'},
        'analytic_smoke':{'baseline_grid_score_checks':188,'threshold_decision_exact':True},'compileall_passed':True,'production_src_changed':False,
        'inherited_kernel_tests':{'record':'reports/research/onset_set_verification_record_v1.json','sha256':sha256_file(root/'onset_set_verification_record_v1.json'),'rerun_in_this_phase':False},
        'new_independent_verifier':totals,'joint_intervals':{'comparisons':4,'resamples_each':4000,'maximum_independent_difference':max(c['independent_replicate_maximum_difference'] for c in summary['comparisons'])},
        'failed_audit_preserved':'reports/research/C72_rolling_onset_threshold_verification.failed.json','failed_audit_repair':'Timestamp integer us/ns assumption replaced by direct Timedelta comparison;original30/360min rules preserved;no model/cache/score changes.',
        'scope':'New experiment/auditor/statistic tests;prior full production suites explicitly inherited. No independent model retraining or candidate-universe RNG replay. Runtime may include concurrent work.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'previous_goal_turn_classification':'progress:e9b268e,completed C68-C70 independent set-decoder study','current_goal_turn_classification':'progress',
        'objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,'new_complete_submission':False,'active_jobs':[],
        'cases':[{'name':c['name'],'feature_arm':c['feature_arm'],'decision_selected':c['decision_selected'],'selected':c['selected'],'assessment_comparisons':c['comparisons']} for c in study['cases']],
        'selected_evidence_gate':study['selected_evidence_gate'],'threshold_secondary_gate':study['threshold_secondary_gate'],
        'completed':['Four preregistered earlier rolling fits,two panels,two seeds per onset arm,strong plain/rich recipe refit before decision','Full selected-window public input replay,both feature arms and independent grids/sets/scores/gates','Posthoc equal-case intervals synchronize simultaneous corridor dates;no policy retuning'],
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only',
        'limits':'Previously available/partly viewed noisy public proxy,not blind latent labels/private/final. New temporal refits do not add later official data. Cross-case and joint intervals descriptive/conditional,not a formal Queue/Total claim.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('rolling_onset_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),flush=True)


if __name__=='__main__':main()
