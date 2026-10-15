"""Freeze two-fold recurrent State and both positive/negative stress evidence."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import numpy as np
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    reports=Path('reports/research');names=['model_artifacts_checkpoint_recurrent_phase2_v1.json',
        'recurrent_phase2_test_record_v1.json','research_checkpoint_recurrent_phase2_v1.json']
    if any((reports/n).exists() for n in names):raise FileExistsError('Never overwrite recurrent phase evidence')
    parent_path=reports/'model_artifacts_checkpoint_recurrent_phase1_v1.json';parent=read(parent_path)
    artifacts=[checked(r) for r in parent['artifacts']];existing={r['path'] for r in artifacts}
    folders=['A73_recurrent_I5_N_winter','A75_recurrent_I5_N_independent_replay','A77c_recurrent_forward_calendar_v3',
        'A81_I210_W_public_outage_stress','A81b_I210_W_public_outage_stress','A82_I210_W_outage_independent_replay',
        'A84_recurrent_forward_independent_verification']
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.pt','.pkl','.npz','.parquet','.zip','.csv'} and p.as_posix() not in existing:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),
                    'phase':'recurrent_State_winter_forward_and_public_outage_stress'});existing.add(p.as_posix())
    preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Locks changed')
    replays=[];studies=[];matched=0
    for name,replay_name in [('A72b_recurrent_I210_W_autumn_full_day','A74b_recurrent_I210_W_independent_replay'),
        ('A73_recurrent_I5_N_winter','A75_recurrent_I5_N_independent_replay')]:
        study=read(reports/(name+'.json'));folder=Path('artifacts/research')/name;replay=read(reports/(replay_name+'.json'))
        if sha256_file(folder/'selection_sealed.json')!=study['selection_seal_sha256'] or read(folder/'selection_sealed.json')!=study['selection']:
            raise ValueError('Original cal policy changed')
        if (folder/'experiment.json').read_bytes()!=(reports/(name+'.json')).read_bytes():raise ValueError('Study mirror differs')
        if replay['registered']['protocol']['source_study_sha256']!=sha256_file(reports/(name+'.json')):
            raise ValueError('Wrong independent replay')
        if replay['n_role_day_replays']!=280 or not replay['all_legal_inputs_exact'] or not replay['all_graph_arrays_exact']:
            raise ValueError('Incomplete role/input replay')
        for row in replay['exports'].values():
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Exported policy changed')
        for key,digest in study['cache_sha256'].items():
            if sha256_file(key)!=digest:raise ValueError('Cached input/evaluator changed')
        for key,digest in study['registered']['protocol']['cache_aux_sha256'].items():
            if sha256_file(key)!=digest:raise ValueError('Graph/profile/prepared cache changed')
        for seed in study['registered']['protocol']['seeds']:
            a,b=f'seed{seed}_tcn',f'seed{seed}_tcn_gru';left,right=study['training'][a],study['training'][b]
            if left['shared_initial_sha256']!=right['shared_initial_sha256'] or left['patch_sequence_sha256']!=right['patch_sequence_sha256']:
                raise ValueError('Architectures not matched')
            with np.load(folder/(a+'_exposure.npz')) as x,np.load(folder/(b+'_exposure.npz')) as y:
                if x.files!=y.files or any(not np.array_equal(x[k],y[k]) for k in x.files):raise ValueError('Exposure differs')
            if any(study['training'][n]['unique_target_grid_cells_supervised']!=study['training'][n]['total_available_target_grid_cells'] for n in [a,b]):
                raise ValueError('Not all fit targets actually supervised')
            matched+=1
        studies.append(study);replays.append(replay)
    forward=read(reports/'A77c_recurrent_forward_calendar_v3.json')
    if forward['selection_changed'] or len(forward['studies'])!=2:raise ValueError('Forward choices changed/incomplete')
    for row,study in zip(forward['studies'],studies):
        name=row['source_study']
        if row['source_study_sha256']!=sha256_file(reports/(name+'.json')) or not row['all_original_choices_frozen']:
            raise ValueError('Forward source/policy changed')
        if len(row['dates'])!=28 or set(row['dates'])&{d for group in study['prepared']['dates'].values() for d in group}:
            raise ValueError('Forward calendar is not independently held from these weights')
    forward_check=read(reports/'A84_recurrent_forward_independent_verification.json')
    if forward_check['registered']['protocol']['source_forward_sha256']!=sha256_file(reports/'A77c_recurrent_forward_calendar_v3.json'):
        raise ValueError('Forward verification source differs')
    if forward_check['n_panel_days']!=56 or forward_check['n_saved_role_day_metrics_verified']!=560 or len(forward_check['direct_fixed_date_policy_replays'])!=8:
        raise ValueError('Incomplete forward metric/sample verification')
    if forward_check['maximum_daily_score_error']>1e-10 or forward_check['maximum_prediction_difference_lane_units']>1e-8:
        raise ValueError('Forward independent replay failed')
    stress=read(reports/'A81b_I210_W_public_outage_stress.json');stress_replay=read(reports/'A82_I210_W_outage_independent_replay.json')
    if stress['selection_changed'] or stress['n_nested_stress_cases']!=168 or stress['n_center_date_seed_draws']!=42:
        raise ValueError('Stress cases/choices changed')
    if stress_replay['registered']['protocol']['source_study_sha256']!=sha256_file(reports/'A81b_I210_W_public_outage_stress.json'):
        raise ValueError('Wrong stress replay source')
    if stress_replay['n_masks_and_teachers_regenerated']!=168 or stress_replay['n_saved_scores_independently_recalculated']!=672:
        raise ValueError('Stress input/score verification incomplete')
    for row in stress['case_files']:
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Stress prediction file changed')
    calendar=read(reports/'A83_I210_W_28day_frozen_calendar.json')
    if not np.all(np.diff(np.array(calendar['dates'],dtype='datetime64[D]'))==np.timedelta64(1,'D')):
        raise ValueError('Secondary28day calendar not contiguous')
    figure=reports/'A85_recurrent_evidence.png';figure_record=read(figure.with_suffix('.json'))
    if sha256_file(figure)!=figure_record['figure_sha256']:raise ValueError('Figure bytes changed')
    for path,digest in figure_record['source_reports_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Figure source changed')
    artifacts.append({'path':figure.as_posix(),'bytes':figure.stat().st_size,'sha256':sha256_file(figure),'phase':'recurrent_evidence_figure'})
    now=datetime.now(timezone.utc).isoformat();gains=[s['comparisons']['tcn_gru_seed_mean']['gain_vs_incumbent'] for s in studies]
    policy=read(reports/'protocol_v2_registered.json')['protocol']['promotion']
    original_gate=bool(np.mean(gains)>=policy['minimum_mean_state_gain'] and min(gains)>=-policy['maximum_panel_state_loss'] and
        all(s['comparisons']['tcn_gru_seed_mean']['paired_days_vs_incumbent']['ci95'][0]>0 for s in studies) and matched>=2)
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'All eight matched models,raw saved predictions/exports,forward calendar and outage failure/verification;prior winter start archive deduplicated'}
    tests={'created_UTC':now,'research_pytest':{'passed':101,'seconds':9.97,'session_id':94793,'basetemp':'E:/ieee/.cache/pytest_public_outage_research'},
        'CPU_pytest':{'passed':85,'optional_Torch_modules_skipped':5,'seconds':7.38,'basetemp':'E:/ieee/.cache/pytest_public_outage_cpu'},
        'compileall_passed':True,'paired_seed_studies_with_exact_initialization_batches_and_exposures':matched,
        'independent_original_role_day_replays':sum(r['n_role_day_replays'] for r in replays),
        'maximum_original_prediction_difference_lane_units':max(r['maximum_absolute_prediction_difference_lane_units'] for r in replays),
        'stress_masks_and_teachers_regenerated':168,'stress_scores_independently_recalculated':672,
        'direct_stress_policy_calls':stress_replay['n_direct_stress_policy_calls'],'maximum_stress_score_error':stress_replay['maximum_score_error'],
        'forward_teacher_panel_days_verified':56,'saved_forward_role_day_metrics_verified':560,
        'direct_forward_policy_calls':8,'maximum_forward_prediction_difference_lane_units':forward_check['maximum_prediction_difference_lane_units'],
        'maximum_forward_daily_score_error':forward_check['maximum_daily_score_error'],'evidence_plot_visually_inspected':True,
        'scope':'Current-machine policy inference and complete stress implementation checks,not independent model retraining/formal accuracy'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'objective_status':'active;formal0.95000+unachieved','previous_goal_turn_classification':'verified wait:interrupted continuation polled live66989 and read state',
        'current_goal_turn_classification':'progress','formal_best_Total':.82940,'new_formal_score':None,'formal_best_preserved':True,
        'new_formal_model_promoted':False,'original_twofold_local_State_evidence_gate_passed':original_gate,
        'original_twofold_mean_State_gain':float(np.mean(gains)),
        'forward_twofold_mean_State_gain':float(np.mean([r['comparisons']['primary_export']['gain_vs_incumbent'] for r in forward['studies']])),
        'six_hour_synthetic_outage_regressions_observed':True,
        'completed':['Winter two-seed TCN/GRU study finished without restarting original session',
            'All eight matched models fully cover available fit targets;four paired actual exposure arrays verified',
            'Both original cal-selected primary policies independently deployed across all cal/assessment dates',
            'Registered February fixed-policy calendar completed,without reselecting models/alphas',
            'New3h/6h metadata-only sensor-outage stress revealed6h regression;dtype failure retained and corrected in new folder',
            '168stress masks/teachers and672scores regenerated;secondary28day same-policy aggregation recorded'],
        'active_jobs_at_checkpoint':[],'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'platform_policy':'Local modeling only per user;no publication,push or upload',
        'next_priorities':['Use forward-calendar and severe-outage evidence to decide production expansion,not just original fold gains',
            'Investigate leakage-safe mask augmentation or input-derived uncertainty gating on independent development masks',
            'Recompute all base/interpolation features after augmentation;never merely hide cached measurement channels',
            'Keep formal State/Physics effects unknown until authorized controlled official evaluation'],
        'limitations':'Two trained directions and optimizer seeds share public calendars;multiple comparisons not adjusted. '
        'GRU adds capacity and fails some severe masks. Positive local scores are not Kaggle/private performance.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (reports/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'original_gate',original_gate,flush=True)


if __name__=='__main__':main()
