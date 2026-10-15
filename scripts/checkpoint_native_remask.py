"""Immutable native-remask experiment checkpoint,all previous bests retained."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_native_remask_v1.json',
        'native_remask_test_record_v1.json','research_checkpoint_native_remask_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite native evidence')
    parent_path=root/'model_artifacts_checkpoint_gap_expert_v1.json';parent=read(parent_path)
    artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts}
    preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    folders=['A96_I210_W_native_remask_fit_views','A97_I210_W_native_remask_ablation','A98_I210_W_native_remask_forward_calendar',
        'A99_I210_W_native_remask_cache_independent_replay','A100_I210_W_native_remask_training_independent_replay',
        'A101_I210_W_native_remask_forward_independent_verification']
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.pt','.pkl','.npz','.zip'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'native_point_and_Queue_shaped_State_data_augmentation'})
                known.add(p.as_posix())
    cache=read(root/'A96_I210_W_native_remask_fit_views.json');training_path=root/'A97_I210_W_native_remask_ablation.json';training=read(training_path)
    forward_path=root/'A98_I210_W_native_remask_forward_calendar.json';forward=read(forward_path)
    ccheck=read(root/'A99_I210_W_native_remask_cache_independent_replay.json');tcheck=read(root/'A100_I210_W_native_remask_training_independent_replay.json')
    fcheck=read(root/'A101_I210_W_native_remask_forward_independent_verification.json')
    for name,check in [('A96_I210_W_native_remask_fit_views',ccheck),('A97_I210_W_native_remask_ablation',tcheck),('A98_I210_W_native_remask_forward_calendar',fcheck)]:
        if check['registered']['protocol']['source_study_sha256']!=sha256_file(root/(name+'.json')):raise ValueError('Verifier source changed')
    if ccheck['n_views_independently_rebuilt']!=736 or ccheck['n_actual_target_poison_checks']!=16 or ccheck['maximum_teacher_grid_difference_normalized_units']>1e-6:
        raise ValueError('Cache/teacher/input verification incomplete')
    if len(tcheck['exposure_checks'])!=4 or tcheck['n_role_day_predictor_calls']!=336 or tcheck['maximum_lane_unit_prediction_difference']>1e-8:
        raise ValueError('Training exposure/deployment verification incomplete')
    if fcheck['n_raw_teacher_days_rebuilt']!=14 or fcheck['n_saved_role_day_metrics_recalculated']!=168 or fcheck['n_direct_predictor_calls']!=24:
        raise ValueError('Forward teacher/metric verification incomplete')
    if fcheck['maximum_score_error']>1e-10 or fcheck['maximum_prediction_difference_lane_units']>1e-8:raise ValueError('Forward replay failed')
    source=Path('artifacts/research')/training_path.stem;seal=source/'selection_sealed.json'
    if sha256_file(seal)!=training['selection_seal_sha256'] or read(seal)!=training['selection']:raise ValueError('Model selection changed')
    if forward['selection_changed'] or forward['registered']['protocol']['source_study_sha256']!=sha256_file(training_path):raise ValueError('Forward choices changed')
    if forward['registered']['protocol']['dates']!=training['registered']['protocol']['future_dates']:raise ValueError('Forward calendar changed')
    for row in cache['rows']+forward['prediction_files']:
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Saved input/prediction changed')
    for path,digest in forward['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Public source changed')
    controls=read(root/'A87_I210_W_outage_augmentation.json')
    for seed in training['registered']['protocol']['seeds']:
        a,b=[training['training'][f'seed{seed}_{v}'] for v in ['point','queue']]
        if a['initial_sha256']!=b['initial_sha256'] or a['initial_sha256']!=controls['training'][f'seed{seed}_control']['initial_sha256']:
            raise ValueError('Not initialized like frozen control')
        if a['patch_sequence_sha256']!=b['patch_sequence_sha256']:raise ValueError('Patch centers differ')
    boundary_path=Path('artifacts/research/A96_I210_W_native_remask_fit_views/public_origin_boundary_audit.json');boundary=read(boundary_path)
    if boundary['n_dates']!=92 or not boundary['all_existing_cache_origins_unchanged']:raise ValueError('Boundary correction changes current training cache')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'Previous artifacts plus all public-fit remasks/new label masks,trained State variants,forward predictions and independent checks'}
    tests={'created_UTC':now,'research_pytest':{'passed':116,'seconds':9.63,'session_id':42407,'basetemp':'E:/ieee/.cache/pytest_native_research_v1'},
        'CPU_pytest':{'passed':100,'optional_Torch_modules_skipped':5,'seconds':7.04,'basetemp':'E:/ieee/.cache/pytest_native_cpu_v1'},
        'additional_neighboring_day_guard_kernel_tests':{'passed':2,'seconds':1.00,'basetemp':'E:/ieee/.cache/pytest_native_midnight_kernels_v1'},
        'neighboring_day_origin_boundary_audit_sha256':sha256_file(boundary_path),
        'compileall_passed':True,'fit_input_loss_views_exact':736,'actual_target_poison_checks':16,
        'maximum_independent_raw_teacher_difference_normalized_units':ccheck['maximum_teacher_grid_difference_normalized_units'],
        'all4actual_label_exposure_arrays_reconstructed':True,'original_cal_ass_role_day_predict_calls':336,
        'maximum_original_prediction_difference_lane_units':tcheck['maximum_lane_unit_prediction_difference'],
        'forward_raw_teacher_days':14,'forward_saved_role_day_metrics':168,'forward_direct_predict_calls':24,
        'maximum_forward_score_error':fcheck['maximum_score_error'],'maximum_forward_prediction_difference_lane_units':fcheck['maximum_prediction_difference_lane_units']}
    primary=training['selection']['primary']
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:equal-alpha diagnosis,new input gate,shifted-law failure audit and local commit0d6e3ec',
        'current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,
        'new_formal_model_promoted':False,'new_formal_score':None,'all_old_complete_submissions_unchanged':True,
        'cal_primary':primary,'old_public_assessment_primary_State':training['assessment'][primary]['S_state'],
        'old_public_assessment_prior_GRU_State':training['assessment']['prior_GRU']['S_state'],
        'registered_forward_primary_State':forward['metrics']['primary_export']['S_state'],
        'registered_forward_prior_GRU_State':forward['metrics']['prior_GRU']['S_state'],
        'additional_label_population_intentional':True,'frozen_control_reuse_explicit':True,
        'active_jobs_at_checkpoint':[],'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'platform_policy':'Local modeling only per user;no publication,push or upload',
        'limits':'One direction and2optimizerseeds share public calendars. New data strategies differ in distinct label positions and label exposure counts. '
        'Mock Queue origins are not actual selector/events. Forward remains public research,not final/private. No new complete production submission or formal Physics/Total score.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('native_checkpoint_complete','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),flush=True)


if __name__=='__main__':main()
