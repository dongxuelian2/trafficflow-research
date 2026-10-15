"""Freeze completed scalar-flow transfer evidence and recheck earlier artifacts."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_flow_transfer_v1.json','flow_transfer_verification_record_v1.json','research_checkpoint_flow_transfer_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite checkpoint')
    parent_path=root/'model_artifacts_checkpoint_cross_remask_phase2_v1.json';parent=read(parent_path)
    artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    stems=['A116_I5_N_point_flow_transfer','A117_I5_N_flow_transfer_forward_calendar','A118_I5_N_flow_table_independent_verification',
        'A119_I5_N_flow_transfer_forward_independent_verification','A120_I5_N_flow_transfer_deployment_verification']
    reports={name:read(root/(name+'.json')) for name in stems};study,forward,table,fcheck,deployment=[reports[n] for n in stems]
    diagnostic_name='A121_I5_N_flow_cal_covariance_diagnostic';diagnostic=read(root/(diagnostic_name+'.json'))
    if diagnostic['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or diagnostic['n_cal_days']!=14 or diagnostic['new_model_exported'] or diagnostic['new_future_results_read']:raise ValueError('Cal-only diagnostic scope differs')
    for check,source in [(fcheck,stems[1]),(deployment,stems[0])]:
        if check['registered']['protocol']['source_study_sha256']!=sha256_file(root/(source+'.json')):raise ValueError('Verifier source changed')
    if table['registered']['protocol']['source_registration_sha256']!=sha256_file(root/(stems[0]+'.registered.json')) or table['registered']['protocol']['dataset_manifest_sha256']!=sha256_file(Path('artifacts/research')/stems[0]/'dataset.json'):raise ValueError('Early input proof not bound to fit')
    if table['n_training_groups_rebuilt']!=765 or table['n_training_rows_verified']!=study['dataset']['rows'] or table['n_actual_60feature_poison_checks']!=8 or table['maximum_weight_error']>1e-12 or table['maximum_raw_flow_teacher_difference_normalized_units']>1e-6:raise ValueError('Full table audit incomplete')
    if len(study['training'])!=2 or not all(r['new_channels_fit']==1 and r['speed_tree_reused'] for r in study['training'].values()):raise ValueError('Flow-only fit count differs')
    reused=deployment['registered']['protocol']['reused_table_proof']
    if reused['sha256']!=sha256_file(root/(stems[2]+'.json')) or deployment['n_training_groups_rebuilt']!=0 or deployment['n_training_groups_previously_verified']!=765:raise ValueError('Reused table proof miscounted')
    if deployment['n_direct_predictor_calls']!=252 or deployment['n_saved_role_day_metrics_recalculated']!=238 or deployment['maximum_prediction_error_lane_units']>1e-8 or not deployment['cal_gate_and_reused_speed_trees_verified']:raise ValueError('Deployment audit incomplete')
    if forward['n_direct_predictor_calls']!=70 or forward['selection_changed'] or forward['maximum_deployed_transfer_error_lane_units']>1e-8:raise ValueError('Forward selection/deployment changed')
    if fcheck['n_raw_teacher_days_rebuilt']!=14 or fcheck['n_saved_role_day_metrics_recalculated']!=126 or fcheck['n_direct_predictor_calls']!=18 or fcheck['maximum_prediction_difference_lane_units']>1e-8:raise ValueError('Forward independent audit incomplete')
    for name in stems+[diagnostic_name]:
        folder=Path('artifacts/research')/name
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.zip','.npz','.npy','.pkl'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'completed_flow_only_channel_transfer'});known.add(p.as_posix())
    for source in [study,forward]:
        for path,digest in source['source_files_sha256'].items():
            if sha256_file(path)!=digest:raise ValueError('Public eval source changed')
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,
        'formal_best_submission_sha256':parent['formal_best_submission_sha256'],'scope':'Scalar flow fits,completed mapped table,all model/score audits;previous artifacts and complete submissions unchanged'}
    verification={'created_UTC':now,'compileall_passed':True,'production_src_changed':False,
        'prior_unit_suite_reference':'cross_remask_phase2_test_record_v1.json:118research,102CPU/5Torch skips;not rerun/count as new experiment',
        'table_groups_rebuilt_once':765,'actual_full60feature_poison_checks_once':8,'training_rows':study['dataset']['rows'],
        'cal_ass_direct_export_calls':252,'cal_ass_metrics_recalculated':238,'forward_direct_calls':70,'forward_saved_metrics_verified':126,
        'forward_audit_additional_direct_calls':18,'table_audit_reused_in_deployment_report':True,
        'cal_only_covariance_diagnostic':{'days':14,'new_export':False,'no_new_forward_arrays_read':True,'source_JSON_contains_previously_viewed_assessment_summaries':True},
        'maximum_cal_ass_prediction_error_lane_units':deployment['maximum_prediction_error_lane_units'],'maximum_forward_score_error':fcheck['maximum_score_error']}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:completed winter remask and channel ensemble,commit0f6f115','current_goal_turn_classification':'progress',
        'objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,
        'completed':['Two400tree flow-only fits using immutable original speed tree','I210fixed channel policy transferred without winter weight search',
            'All765tables/8poisons independently verified while fitting','Frozen new calendar/all deployment and scoring audits complete'],
        'primary':study['selection']['primary'],'old_assessment_fixed_transfer':study['comparisons']['transfer_fixed'],
        'new_calendar_fixed_transfer':forward['comparisons']['transfer_fixed'],'active_jobs':[],
        'cal_covariance_diagnostic':{k:diagnostic[k] for k in ['optimal_bounded_cal_flow_tree_alpha','in_sample_cal_State_gain','passes_predefined_cal_gate']},
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'all_previous_complete_submissions_preserved':True,
        'platform_policy':'Local models and Git only','limits':'Public observations and research dates;not private/final. MSE/row weighting/exposures differ from NN. No formal Physics/Total,whole-network fit or new complete submission.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('flow_checkpoint','old_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'primary',study['selection']['primary'],flush=True)


if __name__=='__main__':main()
