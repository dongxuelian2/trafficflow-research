"""Freeze the completed third-family paired State experiment and audits."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from trafficflow.research_eval import paired_block_interval
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_relative_position_third_v1.json','relative_position_third_verification_record_v1.json','research_checkpoint_relative_position_third_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite third-family checkpoint')
    parent_path=root/'model_artifacts_checkpoint_relative_position_winter_v1.json';parent=read(parent_path)
    artifacts=[checked(row) for row in parent['artifacts']];known={row['path'] for row in artifacts};preserved=[checked(row) for row in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency lock changed')
    stems=['A138_I405_S_matched_relative_position','A139_I405_S_matched_relative_forward','A140b_I405_S_matched_relative_verification','A141_I405_S_matched_relative_forward_verification']
    study,forward,audit,faudit=[read(root/(s+'.json')) for s in stems];nmodels=len(study['exports']);nroles=len(study['assessment'])
    if nmodels!=11 or nroles!=10 or len(study['training'])!=4 or len(study['registered']['protocol']['seeds'])!=2:raise ValueError('Study scope differs')
    for check,stem in [(audit,stems[0]),(faudit,stems[1])]:
        if check['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stem+'.json')):raise ValueError('Audit source differs')
    if not audit['public_geometry_and_graph_exact'] or audit['n_raw_fit_days_rebuilt']!=92 or len(audit['exposure_checks'])!=4 or not all(r['shared_initialization_and_full_cell_exposure_exact'] for r in audit['exposure_checks']):raise ValueError('Matched training audit incomplete')
    if audit['n_direct_predictor_calls']!=308 or audit['n_additional_pure_cal_predictor_calls']!=56 or audit['n_saved_role_day_metrics_recalculated']!=280 or audit['n_selected_checkpoint_alpha_metrics_recalculated']!=20 or audit['maximum_prediction_error_lane_units']>1e-8:raise ValueError('Selected deployment/grid audit incomplete')
    if forward['reference_role']!='strong_reference' or forward['selection_changed'] or forward['n_direct_predictor_calls']!=154:raise ValueError('Frozen forward differs')
    if faudit['n_raw_teacher_days_rebuilt']!=14 or faudit['n_saved_role_day_metrics_recalculated']!=154 or faudit['n_direct_predictor_calls']!=22 or faudit['maximum_prediction_difference_lane_units']>1e-8:raise ValueError('Forward audit incomplete')
    diagnosis_path=root/'A140_I405_S_alpha_precision_diagnosis.json';diagnosis=read(diagnosis_path)
    if diagnosis['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or diagnosis['selected_prediction_maximum_lane_unit_errors']['native_float32_mix']!=0 or any(r['native_float32_mix']!=0 for r in diagnosis['grid_score_differences']):raise ValueError('Precision diagnosis incomplete')
    if sha256_file('artifacts/research/A140_I405_S_matched_relative_verification/source_snapshot.zip')!=diagnosis['failed_attempt_source_snapshot_sha256']:raise ValueError('Failed attempt source archive changed')
    for stem in stems+['A140_I405_S_matched_relative_verification']:
        for p in sorted((Path('artifacts/research')/stem).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip','.pt','.pkl'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'completed_third_family_matched_relative'});known.add(p.as_posix())
    for result in [study,forward,audit]:
        for path,digest in result['source_files_sha256'].items():
            if sha256_file(path)!=digest:raise ValueError('Public source file changed')
    paired=[]
    for phase,metrics in [('old_assessment',study['assessment']),('future_calendar',forward['metrics'])]:
        for control,geo in [('nn_primary','geo_seed_mean')]+[(f'control_seed{s}',f'geo_seed{s}') for s in study['registered']['protocol']['seeds']]:
            a,b=metrics[control],metrics[geo];paired.append({'phase':phase,'control':control,'candidate':geo,'pooled_State_gain':b['S_state']-a['S_state'],
                'day_mean_diagnostic':paired_block_interval([d['daily_State'] for d in a['days']],[d['daily_State'] for d in b['days']])})
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),
        'n_artifacts':len(artifacts),'artifacts':artifacts,'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'Four new paired fits on original target pool,all exports/predictions/raw fit and deployment audits;all previous complete submissions unchanged'}
    verification={'created_UTC':now,'production_src_unchanged':True,'unit_test_evidence':'Prior I210 phase125 research/104CPU plus7optional Torch-module skips;not rerun or counted as new tests',
        'compileall_passed':True,'preflight':{'first_attempt':'Wrong .venv-repro interpreter lacked Torch;no output artifacts or training affected',
            'corrected_research_interpreter':'2030-07-16 raw inputs/teacher/mask exact,1899target cells'},
        'failed_full_grid_audit':{'diagnosis_report_sha256':sha256_file(diagnosis_path),'failed_exit_code':1,'original_float64_mix_max_score_error':max(abs(r['float64_mix']) for r in diagnosis['grid_score_differences']),
            'native_float32_mix_max_score_error':0.,'resolution':'Correct verifier neural arithmetic back to training float32 semantics;no tolerance,model,selection or prediction changed. First source archive/registration preserved;full replay uses new A140b output.'},
        'n_raw_fit_days_rebuilt':92,'n_exposure_archives_verified':4,'n_direct_cal_ass_calls':308,'n_additional_pure_neural_cal_calls':56,
        'n_cal_ass_saved_metrics':280,'n_selected_checkpoint_alpha_grid_metrics':20,'n_direct_forward_calls':154,'n_additional_forward_audit_calls':22,
        'maximum_cal_ass_prediction_error_lane_units':audit['maximum_prediction_error_lane_units'],'maximum_score_error':audit['maximum_score_error'],
        'paired_geometry_vs_plain_GRU':paired,'scope':'Original fit targets here,previous two directions used augmented remask targets;no pooled three-direction causal comparison. Public research dates,two seeds,not final/private.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:winter relative-position experiment/audits completed and cal rejected promotion,commitff276ac','current_goal_turn_classification':'progress',
        'objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,
        'completed':['Third corridor family,original released target pool,four paired plain/relative GRU fits',
            'Frozen full-label LGB and selected A47 TCN as references,cal choice sealed before assessment','Independent all-fit inputs/teachers/plans/cell exposure and selected alpha-grid/deployment checks','Frozen14day forward and independent raw audits'],
        'primary':study['selection']['primary'],'reference_choice':study['selection']['reference_choice'],'old_assessment_comparisons':study['comparisons'],'future_comparisons':forward['comparisons'],
        'active_jobs':[],'formal_submission_changed':False,'platform_policy':'Local models and Git only','final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'limits':'One additional direction/autumn fold;two seeds share labels. Original target pool unlike prior native-remask policies. No hidden Physics/Total,whole-network fit or new complete submission.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('third_family_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'primary',study['selection']['primary'],flush=True)


if __name__=='__main__':main()
