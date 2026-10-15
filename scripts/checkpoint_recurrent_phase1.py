"""Freeze completed first State fold;record the verified live winter handle."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import numpy as np
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import checked,read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--active-session',type=int,required=True)
    parser.add_argument('--verified-utc',required=True);args=parser.parse_args()
    reports=Path('reports/research');names=['model_artifacts_checkpoint_recurrent_phase1_v1.json',
        'recurrent_phase1_test_record_v1.json','research_checkpoint_recurrent_phase1_v1.json']
    if any((reports/n).exists() for n in names):raise FileExistsError('Never overwrite State phase checkpoints')
    parent_path=reports/'model_artifacts_checkpoint_demand_residual_v1.json';parent=read(parent_path)
    artifacts=[checked(r) for r in parent['artifacts']]
    folders=['A72_recurrent_I210_W_autumn','A72b_recurrent_I210_W_autumn_full_day',
        'A74_recurrent_I210_W_independent_replay','A74b_recurrent_I210_W_independent_replay']
    for name in folders:
        folder=Path('artifacts/research')/name
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.pt','.pkl','.npz','.parquet','.csv','.zip'}:
                artifacts.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),
                    'phase':'recurrent_State_first_fold_and_preserved_trials'})
    # Only the immutable start archive is frozen from the live winter job.
    live=Path('artifacts/research/A73_recurrent_I5_N_winter');snapshot=live/'source_snapshot.zip'
    artifacts.append({'path':snapshot.as_posix(),'bytes':snapshot.stat().st_size,'sha256':sha256_file(snapshot),
        'phase':'live_winter_immutable_start_archive_only'})
    if len({r['path'] for r in artifacts})!=len(artifacts):raise ValueError('Duplicate artifact paths')
    preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Locks changed')
    study=read(reports/(folders[1]+'.json'));source=Path('artifacts/research')/folders[1]
    replay=read(reports/(folders[3]+'.json'));stop=read(reports/'A72_restricted_supervision_stop.json')
    if stop['selection_seal_exists'] or stop['assessment_labels_read']:raise ValueError('Restricted pilot label/seal claim differs')
    if read(source/'selection_sealed.json')!=study['selection'] or sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256']:
        raise ValueError('Original State choices changed')
    if (source/'experiment.json').read_bytes()!=(reports/(folders[1]+'.json')).read_bytes():raise ValueError('Report mirror changed')
    if replay['registered']['protocol']['source_study_sha256']!=sha256_file(reports/(folders[1]+'.json')):
        raise ValueError('Replay source differs')
    if replay['n_role_day_replays']!=280 or not all(replay[k] for k in ['all_legal_inputs_exact','all_graph_arrays_exact','all_predictions_within_absolute_tolerance']):
        raise ValueError('Independent deployment incomplete')
    for name,digest in study['model_sha256'].items():
        if sha256_file(source/name)!=digest:raise ValueError('Trained checkpoint changed')
    for name,digest in study['cache_sha256'].items():
        if sha256_file(name)!=digest:raise ValueError('State cache changed')
    for name,digest in study['registered']['protocol']['cache_aux_sha256'].items():
        if sha256_file(name)!=digest:raise ValueError('Auxiliary cache changed')
    for seed in study['registered']['protocol']['seeds']:
        left,right=f'seed{seed}_tcn',f'seed{seed}_tcn_gru';a,b=study['training'][left],study['training'][right]
        if a['shared_initial_sha256']!=b['shared_initial_sha256'] or a['patch_sequence_sha256']!=b['patch_sequence_sha256']:
            raise ValueError('Initialization or batches not matched')
        with np.load(source/(left+'_exposure.npz')) as x,np.load(source/(right+'_exposure.npz')) as y:
            if x.files!=y.files or any(not np.array_equal(x[k],y[k]) for k in x.files):raise ValueError('Exposure arrays differ')
        if any(study['training'][n]['unique_target_grid_cells_supervised']!=517171 for n in [left,right]):
            raise ValueError('Full first-fold target coverage not established')
    for row in replay['exports'].values():
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Frozen exported role changed')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'Completed first fold,stopped pilot,failed/fixed exporter and immutable winter source only;live checkpoints intentionally excluded'}
    tests={'created_UTC':now,'research_pytest':{'passed':97,'seconds':9.96,'session_id':2835,
        'basetemp':'E:/ieee/.cache/pytest_recurrent_boundary_research'},
        'CPU_pytest':{'passed':81,'skipped_optional_torch_modules':5,'seconds':7.46,
            'basetemp':'E:/ieee/.cache/pytest_recurrent_boundary_cpu'},
        'additional_batch_independence_kernel_tests':{'passed':4,'seconds':2.93},'compileall_passed':True,
        'completed_matched_neural_models':4,'unique_fit_grid_targets_each':517171,'actual_exposure_arrays_matched':True,
        'independent_role_day_replays':280,'maximum_prediction_difference_lane_units':replay['maximum_absolute_prediction_difference_lane_units'],
        'formal_actions':False,'scope':'Current-machine inference and data/label separation;not independent retraining or formal score evidence'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'objective_status':'active;formal0.95000+unachieved','current_goal_turn_classification':'progress',
        'previous_goal_turn_classification':'progress:eb52efa,OD residual/floor controls and six-seed stress evidence',
        'formal_best_Total':.82940,'new_formal_score':None,'formal_best_preserved':True,'new_formal_model_promoted':False,
        'completed':['Independent bidirectional GRU branch and finite-context deployable State model',
            'Identical TCN shared initialization,batches and actual full-target supervision per optimizer seed',
            'Restricted center-only trial stopped before assessment;all partial weights/failure evidence preserved',
            'I210_W autumn two-seed matched study complete;cal primary GRU mean,pooled State gain.001914',
            '10exported policy roles,280independent predictor/day replays;strong original incumbent retained'],
        'first_fold_only':{'selected_primary':study['selection']['primary'],'incumbent_State':study['assessment']['incumbent']['S_state'],
            'primary_State':study['assessment'][study['selection']['primary']]['S_state'],
            'gap_to_project_minimum_pooled_State_gain':.002-study['comparisons']['tcn_gru_seed_mean']['gain_vs_incumbent'],
            'missing180min_diagnostic_targets':study['assessment']['incumbent']['diagnostic_strata']['missing180min']['n']},
        'active_jobs_at_checkpoint':[{'session_id':args.active_session,'task':'A73_I5_N_winter_four_model_training',
            'output':live.as_posix(),'last_tool_verified_running_UTC':args.verified_utc,
            'live_progress_files_not_frozen':True,'start_archive_sha256':sha256_file(snapshot)}],
        'pending':['Poll the same live session;observation timeout is not terminal',
            'Complete winter four-model study,then independent A75 deployment',
            'A76twofold comparison and A77c preregistered February frozen-policy extension',
            'Full phase artifact/checkpoint and final twofold results;no platform tasks'],
        'platform_policy':'User directs local models;no publication,push or upload',
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'limitations':'First fold only,14previously viewed public-train assessment days. Additional GRU capacity and cal selection uncertainty remain. '
        'Positive day-mean CI is not a formal or nonlinear pooled-score CI. Winter/February outcomes remain unknown.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (reports/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('previous_unchanged',len(parent['artifacts']),'total_artifacts',len(artifacts),'winter_session',args.active_session,flush=True)


if __name__=='__main__':main()
