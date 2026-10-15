"""Freeze probability-calibrated IoU study and preserve all prior evidence."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_probabilistic_onset_v1.json','probabilistic_onset_verification_record_v1.json','research_checkpoint_probabilistic_onset_v1.json']
    if any((root/name).exists() for name in names):raise FileExistsError('Never overwrite probabilistic checkpoint')
    parent_path=root/'model_artifacts_checkpoint_rolling_onset_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    stems=['C74_probabilistic_onset_sets','C75_probabilistic_onset_verification','C76_expected_IoU_known_joint_stress','C77_probabilistic_onset_matched_summary'];study,audit,stress,summary=[read(root/(name+'.json')) for name in stems]
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or summary['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or summary['independent_audit_sha256']!=sha256_file(root/(stems[1]+'.json')) or summary['stress_study_sha256']!=sha256_file(root/(stems[2]+'.json')):raise ValueError('New report source chain changed')
    cfg=study['registered']['protocol'];proof=read(cfg['reused_raw_input_proof'])
    if sha256_file(cfg['reused_raw_input_proof'])!=cfg['reused_raw_input_proof_sha256'] or study['reused_raw_window_audit']!=2196 or study['new_tree_models']!=0:raise ValueError('Reused audit or fitting scope differs')
    for path,digest in {**cfg['unchanged_source_constructors_sha256'],**proof['source_files_sha256'],**audit['source_files_sha256']}.items():
        if sha256_file(path)!=digest:raise ValueError('Public inputs or reused feature constructors changed')
    t=audit['totals']
    if len(study['cases'])!=4 or len(audit['cases'])!=4 or not all(c['both_calibration_fits_and_new_decoders_grids_gates_verified'] for c in audit['cases']):raise ValueError('Incomplete case audit')
    expected={'new_calibration_raw_windows':62,'new_calibration_feature_arm_rebuilds':124,'new_calibration_states_verified':8,'quadrature_curve_checks':582,'direct_booster_calls':100,'portable_window_calls':872,'additional_shuffled_calls':448,'future_history_rejections':32}
    if t!=expected or study['new_calibration_window_occurrences']!=62 or study['n_calibration_states']!=8:raise ValueError('New verification scope differs')
    if audit['maximum_integral_DP_curve_difference']>1e-10 or audit['maximum_calibration_parameter_difference']>1e-5 or audit['maximum_projected_KKT_gradient']>1e-7:raise ValueError('Independent mathematical checks failed')
    if len(summary['comparisons'])!=11 or len(summary['known_joint_stress_independent_checks'])!=7 or not stress['all_3_full_coverage_IID_cases_match_exhaustive_optimum']:raise ValueError('Known-joint or matched summary incomplete')
    if any(r['independent_replicate_maximum_difference']>1e-12 for r in summary['comparisons']) or any(r['maximum_score_difference']>1e-12 for r in summary['known_joint_stress_independent_checks']):raise ValueError('Independent summary replay differs')
    for stem in stems:
        for path in sorted((Path('artifacts/research')/stem).rglob('*')):
            if path.is_file() and path.suffix in {'.json','.npz','.zip','.pkl','.txt','.parquet'} and path.as_posix() not in known:
                artifacts.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),'phase':'probabilistic_onset_calibration_and_exact_IoU_complete'});known.add(path.as_posix())
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],'scope':'Eight separate-phase monotone calibrators,new probability set decoders,exact-known-joint stress and independent audits. No tree refits;complete submissions unchanged.'}
    verification={'created_UTC':now,'new_unit_tests':{'passed':4,'seconds':1.71,'scope':'Exhaustive independent-label truth/prediction sets,empty/ties/complete-domain/permutation,calibration weights and gradient'},
        'research_pytest':{'passed':141,'seconds':12.64,'session_id':74352,'basetemp':'.cache/pytest_probabilistic_onset_research_v1'},'CPU_pytest':{'passed':120,'optional_Torch_modules_skipped':7,'seconds':7.88,'basetemp':'.cache/pytest_probabilistic_onset_CPU_v1'},
        'compileall_passed':True,'new_independent_verifier':t,'old_raw_input_proof_reused':2196,'maximum_integral_curve_difference':audit['maximum_integral_DP_curve_difference'],'maximum_calibration_parameter_difference':audit['maximum_calibration_parameter_difference'],'maximum_projected_KKT_gradient':audit['maximum_projected_KKT_gradient'],
        'joint_intervals':{'comparisons':11,'resamples_each':4000,'maximum_independent_difference':max(r['independent_replicate_maximum_difference'] for r in summary['comparisons'])},'known_joint_stress_cases_independently_checked':7,
        'scope':'Fresh current production suites;old2196raw proof explicitly reused. Eight2parameter calibrators,not new classifiers. Timings may include concurrent work. Mathematical optimum valid only under specified full-domain independent distribution.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'previous_goal_turn_classification':'progress:c480022 completed C71-C73 rolling refits/audits','current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved',
        'formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,'new_complete_submission':False,'active_jobs':[],
        'cases':[{'name':c['name'],'calibration_onset_windows':c['calibration_states'][c['feature_arm']]['n_windows'],'decision_selected':c['decision_selected'],'selected':c['selected'],'assessment_comparisons':c['comparisons']} for c in study['cases']],
        'evidence_gates':summary['evidence_gates'],'joint_secondary_evidence':[r for r in summary['comparisons'] if (r['baseline'],r['candidate']) in [('incumbent','iou_cal_nonempty'),('iou_cal','iou_cal_nonempty'),('threshold_raw','iou_cal_nonempty')]],
        'completed':['Exact Bernoulli expected-IoU kernel verified against exhaustive sets and independent quadrature','Eight monotone calibrators from separate15day prior phase,all frozen rolling trees unchanged','Primary policies all retain incumbent;nonempty joint signal is secondary and uncorrected','Known correlated/missing toy laws show regret despite correct marginals;not formal traffic scores'],
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only','limits':'Previously viewed public observation proxy;sampling forces future observed positives and can inflate nonempty benefit. Joint11contrasts uncorrected,not policy selection. Real traffic dependence/future eligibility unknown. No formal Queue/Physics/OD/Total claim.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    print('probabilistic_onset_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),flush=True)


if __name__=='__main__':main()
