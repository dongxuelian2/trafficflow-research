"""Freeze complete matched augmentation study,failures/negatives included."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import numpy as np
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_augmentation_phase1_v1.json',
        'augmentation_phase1_test_record_v1.json','research_checkpoint_augmentation_phase1_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite augmentation evidence')
    parent_path=root/'model_artifacts_checkpoint_recurrent_phase2_v1.json';parent=read(parent_path)
    artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts}
    preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    folders=['A86_I210_W_fit_outage_views','A87_I210_W_outage_augmentation','A88_I210_W_augmentation_future_outages',
        'A89_I210_W_fit_augmentation_independent_replay','A90_I210_W_augmentation_training_independent_replay',
        'A91_I210_W_augmentation_future_independent_verification']
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.pt','.pkl','.npz','.zip'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'matched_input_only_State_outage_augmentation'})
                known.add(p.as_posix())
    cache=read(root/'A86_I210_W_fit_outage_views.json');training=read(root/'A87_I210_W_outage_augmentation.json')
    future=read(root/'A88_I210_W_augmentation_future_outages.json');cache_check=read(root/'A89_I210_W_fit_augmentation_independent_replay.json')
    train_check=read(root/'A90_I210_W_augmentation_training_independent_replay.json');future_check=read(root/'A91_I210_W_augmentation_future_independent_verification.json')
    for source,verification,key in [('A86_I210_W_fit_outage_views',cache_check,'source_study_sha256'),
        ('A87_I210_W_outage_augmentation',train_check,'source_study_sha256'),('A88_I210_W_augmentation_future_outages',future_check,'source_study_sha256')]:
        if verification['registered']['protocol'][key]!=sha256_file(root/(source+'.json')):raise ValueError('Verification source changed')
    if cache_check['n_views_rebuilt']!=368 or cache_check['n_actual_public_measurement_poison_checks']!=8 or not cache_check['all_inputs_and_regions_exact']:
        raise ValueError('Fit-only input checks incomplete')
    if len(train_check['exposure_checks'])!=2 or train_check['n_role_day_predictions_replayed']!=280 or train_check['maximum_absolute_lane_unit_error']>1e-8:
        raise ValueError('Training replay incomplete')
    if any(not r['original_control_checkpoint_tensor_exact'] or not r['both_actual_exposure_arrays_exact'] for r in train_check['exposure_checks']):
        raise ValueError('Controls/exposure not matched')
    if future_check['n_ordinary_saved_role_day_metrics_verified']!=126 or future_check['n_stress_masks_and_teachers_regenerated']!=168 or future_check['n_stress_saved_scores_independently_recalculated']!=1512:
        raise ValueError('Future saved metric checks incomplete')
    if future_check['n_direct_role_predictor_calls']!=54 or future_check['maximum_score_error']>1e-10 or future_check['maximum_prediction_error_total_units']>1e-8:
        raise ValueError('Future sampled prediction verification failed')
    source=Path('artifacts/research/A87_I210_W_outage_augmentation');seal=source/'selection_sealed.json'
    if sha256_file(seal)!=training['selection_seal_sha256'] or read(seal)!=training['selection']:raise ValueError('Original choice changed')
    if future['selection_changed'] or future['registered']['protocol']['source_study_sha256']!=sha256_file(root/'A87_I210_W_outage_augmentation.json'):
        raise ValueError('Future choices/source changed')
    if future['registered']['protocol']['dates']!=training['registered']['protocol']['future_evaluation_dates']:raise ValueError('Future dates changed')
    for row in cache['rows']:
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Input views changed')
    for row in future['case_records']+future['ordinary_files']:
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Future saved predictions changed')
    for seed in training['registered']['protocol']['seeds']:
        a,b=[training['training'][f'seed{seed}_{arm}'] for arm in ['control','augmented']]
        if a['initial_sha256']!=b['initial_sha256'] or a['patch_sequence_sha256']!=b['patch_sequence_sha256'] or a['total_supervision_exposures']!=b['total_supervision_exposures']:
            raise ValueError('Training not matched')
        with np.load(source/(f'seed{seed}_control_exposure.npz')) as x,np.load(source/(f'seed{seed}_augmented_exposure.npz')) as y:
            if any(not np.array_equal(x[k],y[k]) for k in x.files):raise ValueError('Actual exposures differ')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'All old artifacts plus fit-only augmented inputs,matched weights,all saved predictions,controls and independent verifications'}
    tests={'created_UTC':now,'research_pytest':{'passed':103,'seconds':9.08,'session_id':92262,'basetemp':'E:/ieee/.cache/pytest_augmentation_research_v1'},
        'CPU_pytest':{'passed':87,'optional_Torch_modules_skipped':5,'seconds':6.69,'basetemp':'E:/ieee/.cache/pytest_augmentation_cpu_v1'},
        'compileall_passed':True,'fit_inputs_exactly_rebuilt':368,'actual_public_measurement_poison_checks':8,
        'original_control_checkpoints_tensor_exact':2,'training_cal_assessment_independent_role_day_calls':280,
        'maximum_training_prediction_error_lane_units':train_check['maximum_absolute_lane_unit_error'],
        'future_ordinary_metrics':126,'future_stress_masks_teachers':168,'future_stress_saved_scores':1512,
        'direct_future_role_calls':54,'maximum_future_score_error':future_check['maximum_score_error'],
        'maximum_future_prediction_error_total_units':future_check['maximum_prediction_error_total_units']}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:completed twofold recurrent training,forward validation,severe-outage evidence and committed9d732ea',
        'current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,
        'formal_model_promoted':False,'new_formal_score':None,'all_old_complete_submissions_unchanged':True,
        'cal_primary':training['selection']['primary'],
        'original_public_assessment_augmented_minus_control':training['comparisons']['augmented_seed_mean_vs_control_seed_mean']['pooled_State_gain'],
        'future_ordinary_augmented_minus_control':future['ordinary_comparisons']['augmented_seed_mean_vs_control_seed_mean']['pooled_State_gain'],
        'future_synthetic_augmented_minus_control':{k:v['augmented_seed_mean']-v['control_seed_mean'] for k,v in future['stress_summary'].items()},
        'active_jobs_at_checkpoint':[],'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'platform_policy':'Local modeling only per user;no publication,push or upload',
        'limits':'One direction,viewed original development dates and fixed new public research calendar. '
        'Same labels/architecture and optimizer budget do not ensure same selected epoch/alpha. '
        'Synthetic target populations differ from Task1 targets;neither score is a formal Kaggle result.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('augmentation_checkpoint_complete','old_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),flush=True)


if __name__=='__main__':main()
