"""Freeze shifted-mask expert evidence without claiming formal improvement."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_gap_expert_v1.json',
        'gap_expert_test_record_v1.json','research_checkpoint_gap_expert_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite expert evidence')
    parent_path=root/'model_artifacts_checkpoint_augmentation_phase1_v1.json';parent=read(parent_path)
    artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts}
    preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    folders=['A92_equal_blend_augmentation_diagnostic','A92b_equal_blend_direct_diagnostic',
        'A93_I210_W_gap_expert_frozen_policy','A94_I210_W_gap_expert_shifted_calendar','A95_I210_W_gap_expert_shift_independent_verification']
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.pt','.pkl','.npz','.zip'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'gap_expert_shifted_mask_evidence'})
                known.add(p.as_posix())
    equal=read(root/'A92b_equal_blend_direct_diagnostic.json');policy_path=root/'A93_I210_W_gap_expert_frozen_policy.json';policy=read(policy_path)
    study_path=root/'A94_I210_W_gap_expert_shifted_calendar.json';study=read(study_path)
    check=read(root/'A95_I210_W_gap_expert_shift_independent_verification.json')
    if equal['n_direct_fixed_alpha_predictor_calls']!=728:raise ValueError('Incomplete equal-alpha reinference')
    for path,digest in equal['sources_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Equal-alpha evaluator source changed')
    if sha256_file(policy['export']['path'])!=policy['export']['sha256'] or policy['new_evaluation_labels_read'] or policy['new_model_parameters_fitted']:
        raise ValueError('Gate not fixed before outcomes')
    if study['registered']['protocol']['source_policy_sha256']!=sha256_file(policy_path) or study['threshold_or_models_changed']:
        raise ValueError('Evaluated gate differs')
    if check['registered']['protocol']['source_study_sha256']!=sha256_file(study_path):raise ValueError('Wrong verification source')
    expected={'n_ordinary_saved_metrics_verified':60,'n_case_geometries_regenerated':336,'n_day_segments_regenerated':378,
        'n_segment_scores_verified':1512,'n_pooled_case_scores_verified':1344,'n_independent_gate_arrays_exact':393,'n_direct_serialized_predictor_calls':44}
    for key,value in expected.items():
        if check[key]!=value:raise ValueError('Verification incomplete: '+key)
    if check['maximum_score_error']>1e-10 or check['maximum_prediction_error_total_units']>1e-8:raise ValueError('Verification outside tolerance')
    if sum(r['n_gate_targets'] for r in study['ordinary_files'])!=0 or study['ordinary_comparisons']['gated']['pooled_State_gain']!=0:
        raise ValueError('Ordinary no-activation finding changed')
    if sha256_file('vendor/official/docs/MASK_SPEC.md')!='8425656f94f9d82c75ed8fc0bad35a79499996ab211fc1e27aeab2981e0ee749':raise ValueError('Official mask spec changed')
    for record in study['ordinary_files']+[r for c in study['case_records'] for r in c['segment_files']]:
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Saved predictions changed')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'All previous artifacts plus equal-alpha inversion failure/direct rerun,frozen input gate,all new shifted-case predictions and independent verification'}
    tests={'created_UTC':now,'research_pytest':{'passed':114,'seconds':11.79,'session_id':17631,'basetemp':'E:/ieee/.cache/pytest_gap_shift_research_v1'},
        'CPU_pytest':{'passed':98,'optional_Torch_modules_skipped':5,'seconds':9.15,'session_id':41489,'basetemp':'E:/ieee/.cache/pytest_gap_shift_cpu_v1'},
        'compileall_passed':True,'equal_alpha_direct_predictor_calls':728,'independent_verification_counts':expected,
        'maximum_independent_score_error':check['maximum_score_error'],'maximum_direct_prediction_error_total_units':check['maximum_prediction_error_total_units'],
        'scope':'All gate/geometry contracts plus current-machine replay;not independent new traffic events or formal leaderboard accuracy'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:matched augmentation models,independent new-calendar evidence and local commit6c76883',
        'current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,
        'formal_model_promoted':False,'new_formal_score':None,'all_old_complete_submissions_unchanged':True,
        'equal_alpha_ordinary_gain':equal['ordinary_augmented_minus_control'],
        'ordinary_gate_target_count':0,'ordinary_gate_gain':0.,
        'whole_corridor4h_synthetic_gate_gain':study['stress_summary']['whole_corridor_4h']['gated_minus_control'],
        'decision':'Do not promote180min gate:zero ordinary activation/no Task1 gain and material all-sensor blackout regression',
        'active_jobs_at_checkpoint':[],'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'platform_policy':'Local modeling only per user;no publication,push or upload',
        'next_priorities':['Return to official20/30/50pct point masks and Queue90min additional blanks on actual Task1 targets',
            'Use fit-only public teacher supervision,new training masks and chronological strong controls;do not extend same severe-mask threshold grid',
            'Keep shifted complete-blackout and day-boundary scenarios as regression tests',
            'State and incomplete physical diagnostics remain separate;formal State/Physics/Total unknown'],
        'limits':'One corridor,public research calendars,conditional unadjusted intervals. Cross-midnight events share adjacent days. '
        'Gate reads full released offline State observations and must not be used as causal Queue input.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('gap_expert_checkpoint_complete','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),flush=True)


if __name__=='__main__':main()
