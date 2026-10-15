"""Completed tree controls plus immutable winter inputs and live-job handoff."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked
from native_remask_reference import reference_controls
from native_remask_roles import role_paths


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-session',type=int,required=True)
    parser.add_argument('--last-live-poll-utc',required=True);args=parser.parse_args()
    root=Path('reports/research');names=['model_artifacts_checkpoint_cross_remask_phase1_v1.json',
        'cross_remask_phase1_test_record_v1.json','research_checkpoint_cross_remask_phase1_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite cross-remask checkpoint')
    parent_path=root/'model_artifacts_checkpoint_native_remask_v1.json';parent=read(parent_path)
    artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts}
    preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    folders=['A102_I5_N_native_remask_fit_views','A103_I210_W_native_remask_LGB_control','A105_I5_N_native_remask_cache_independent_replay',
        'A109_I210_W_remask_tree_forward_calendar','A110_I210_W_remask_tree_independent_verification','A111_I210_W_remask_tree_forward_independent_verification']
    def append(path,phase):
        if path.as_posix() not in known:
            artifacts.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),'phase':phase});known.add(path.as_posix())
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.pkl','.npz','.zip','.parquet'}:append(p,'completed_tree_and_winter_remask_input_evidence')
    winter=Path('artifacts/research/A104_I5_N_native_remask_ablation')
    if (root/'A104_I5_N_native_remask_ablation.json').exists():raise ValueError('Winter already complete;use a completed checkpoint instead')
    append(winter/'source_snapshot.zip','immutable_start_archive_of_live_winter_job')
    progress=read(winter/'progress.json') if (winter/'progress.json').exists() else {'training':{},'choices':{}}
    frozen_roles={}
    for role,row in progress['training'].items():
        for suffix,expected in [('.pt',None),('_exposure.npz',row['exposure_sha256'])]:
            p=winter/(role+suffix)
            if expected is not None and sha256_file(p)!=expected:raise ValueError('Completed role exposure changed')
            append(p,'completed_fit_cal_role_in_ongoing_study_no_assessment')
        p=winter/('State_'+role+'.pkl')
        if sha256_file(p)!=row['model_sha256']:raise ValueError('Completed role export changed')
        append(p,'completed_fit_cal_role_in_ongoing_study_no_assessment')
        frozen_roles[role]={'training':row,'choice':progress['choices'][role],'assessment':None}
    views=read(root/'A102_I5_N_native_remask_fit_views.json');vcheck=read(root/'A105_I5_N_native_remask_cache_independent_replay.json')
    tree_path=root/'A103_I210_W_native_remask_LGB_control.json';tree=read(tree_path)
    tcheck=read(root/'A110_I210_W_remask_tree_independent_verification.json');forward_path=root/'A109_I210_W_remask_tree_forward_calendar.json';forward=read(forward_path)
    fcheck=read(root/'A111_I210_W_remask_tree_forward_independent_verification.json')
    for name,check in [('A102_I5_N_native_remask_fit_views',vcheck),('A103_I210_W_native_remask_LGB_control',tcheck),('A109_I210_W_remask_tree_forward_calendar',fcheck)]:
        if check['registered']['protocol']['source_study_sha256']!=sha256_file(root/(name+'.json')):raise ValueError('Verification source changed')
    if vcheck['n_views_independently_rebuilt']!=1224 or vcheck['n_actual_target_poison_checks']!=16 or vcheck['maximum_teacher_grid_difference_normalized_units']>1e-6:
        raise ValueError('Winter input/teacher verification incomplete')
    if tcheck['n_training_groups_rebuilt']!=920 or tcheck['n_direct_tree_role_day_calls']!=168 or tcheck['n_direct_selected_policy_calls']!=4 or tcheck['n_saved_role_day_metrics_verified']!=392:
        raise ValueError('Tree verification incomplete')
    if tcheck['maximum_prediction_error_lane_units']>1e-8 or tcheck['maximum_weight_error']>1e-12:raise ValueError('Tree replay/weights failed')
    if fcheck['n_saved_role_day_metrics_recalculated']!=210 or fcheck['n_direct_predictor_calls']!=18 or fcheck['maximum_prediction_difference_lane_units']>1e-8:
        raise ValueError('Tree forward verification incomplete')
    if forward['choices_changed'] or tree['selection']['primary']!='nn_primary':raise ValueError('Tree policy finding changed')
    for p,digest in forward['source_files_sha256'].items():
        if sha256_file(p)!=digest:raise ValueError('Forward source changed')
    matches={}
    for name in ['A72b_recurrent_I210_W_autumn_full_day','A73_recurrent_I5_N_winter']:
        replay,_,rows,initials=reference_controls(root/(name+'.json'))
        matches[name]={'replay':replay.as_posix(),'constructor_sources_and_shared_initial_hashes_exact':True,'full_initial_sha256':initials,'control_roles':list(rows)}
    compatible=len(role_paths(root/'A97_I210_W_native_remask_ablation.json'))
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),
        'n_artifacts':len(artifacts),'artifacts':artifacts,'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'live_worker_policy':'Mutable progress/current best files excluded;only already-completed per-role files and start archive frozen',
        'scope':'Completed I210tree controls/forward checks and all winter inputs;ongoing winter assessment explicitly unknown'}
    tests={'created_UTC':now,'research_pytest':{'passed':117,'seconds':13.92,'session_id':46376,'basetemp':'E:/ieee/.cache/pytest_cross_remask_research_v1'},
        'CPU_pytest':{'passed':101,'optional_Torch_modules_skipped':5,'seconds':11.42,'session_id':76807,'basetemp':'E:/ieee/.cache/pytest_cross_remask_cpu_v1'},
        'compileall_passed':True,'old_A97_role_loader_compatible_count':compatible,'original_reference_initial_checks':matches,
        'winter_input_views_rebuilt':1224,'actual_winter_target_poison_checks':16,'tree_training_groups_rebuilt':920,
        'tree_direct_role_day_calls':168,'direct_selected_calls':4,'tree_saved_cal_ass_metrics':392,'tree_forward_saved_metrics':210,'tree_forward_direct_calls':18,
        'maximum_tree_replay_error_lane_units':tcheck['maximum_prediction_error_lane_units'],'maximum_forward_score_error':fcheck['maximum_score_error']}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:native-remask I210 training,new-calendar gains and commit71bf316',
        'current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,
        'new_formal_score':None,'formal_model_promoted':False,'all_previous_complete_submissions_preserved':True,
        'completed':['Four same-view weighted tree fits and cal-only policy selection','New14day tree/NN forward evaluation with no reselection',
            'All tree tables/weights and saved-policy metrics independently verified','Winter1224fit-only views and16poison checks complete'],
        'winter_completed_fit_cal_roles':frozen_roles,'winter_primary':None,'winter_assessment':None,'winter_forward':None,
        'active_jobs':[{'session_id':args.training_session,'task':'A104_I5_N_native_remask_ablation','last_tool_poll_confirmed_live_UTC':args.last_live_poll_utc,
            'policy':'Poll this same handle;never restart due an observation timeout. Need all4models and selection seal before assessment.'}],
        'resume_steps':['Poll live unified-exec session first;do not duplicate A104 training',
            'After terminal0 and final A104 JSON,run evaluate_native_remask_forward.py as A106_I5_N_native_remask_forward_calendar',
            'Run verify_native_remask_training.py as A107_I5_N_native_remask_training_independent_replay',
            'Run verify_native_remask_forward.py as A108_I5_N_native_remask_forward_independent_verification',
            'Then make actual two-corridor comparison,do not infer winter accuracy from partialcal'],
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models only;no publication,push or upload',
        'limits':'Same view pools do not isolate architecture from feature encoding/loss/reweighting. Winter remains incomplete and running. No new full submission or formal Physics/Total.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('cross_phase1_checkpoint','old_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'live_session',args.training_session,'completed_fit_cal_roles',list(frozen_roles),flush=True)


if __name__=='__main__':main()
