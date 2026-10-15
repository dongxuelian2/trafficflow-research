"""Archive completed winter transfer and the independent channel ensemble study."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_cross_remask_phase2_v1.json',
        'cross_remask_phase2_test_record_v1.json','research_checkpoint_cross_remask_phase2_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite phase checkpoint')
    parent_path=root/'model_artifacts_checkpoint_cross_remask_phase1_v1.json';parent=read(parent_path)
    artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts}
    preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    stems=['A104_I5_N_native_remask_ablation','A106_I5_N_native_remask_forward_calendar',
        'A107_I5_N_native_remask_training_independent_replay','A108_I5_N_native_remask_forward_independent_verification',
        'A112_I210_W_channel_state_ensemble','A113_I210_W_channel_state_forward_calendar',
        'A114_I210_W_channel_state_forward_independent_verification','A115_I210_W_channel_selection_independent_verification',
        'A115b_I210_W_channel_selection_independent_verification']
    for name in stems:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.pkl','.npz','.zip','.parquet','.pt'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'completed_winter_transfer_and_channel_policy'});known.add(p.as_posix())
    studies={stem:read(root/(stem+'.json')) for stem in stems if not stem.startswith('A115_')}
    def proof(check_name,source_name):
        report=studies[check_name]
        if report['registered']['protocol']['source_study_sha256']!=sha256_file(root/(source_name+'.json')):raise ValueError('Verification source changed')
        return report
    winter=studies[stems[0]];forward=studies[stems[1]];channel=studies[stems[4]];channel_forward=studies[stems[5]]
    wc=proof(stems[2],stems[0]);wf=proof(stems[3],stems[1]);cf=proof(stems[6],stems[5]);cc=proof(stems[8],stems[4])
    if len(winter['training'])!=4 or len(wc['exposure_checks'])!=4 or not all(r['full_actual_exposure_arrays_exact'] for r in wc['exposure_checks']):raise ValueError('Winter fits/exposure verification incomplete')
    if wc['n_role_day_predictor_calls']!=336 or wc['maximum_lane_unit_prediction_difference']>1e-8:raise ValueError('Winter deployment incomplete')
    for data,nmetrics,ncalls in [(wf,168,24),(cf,56,8)]:
        if data['n_raw_teacher_days_rebuilt']!=14 or data['n_saved_role_day_metrics_recalculated']!=nmetrics or data['n_direct_predictor_calls']!=ncalls or data['maximum_prediction_difference_lane_units']>1e-8:raise ValueError('Forward audit incomplete')
    if cc['n_raw_teacher_days_rebuilt']!=28 or cc['n_saved_role_day_metrics_recalculated']!=112 or cc['n_cal_grid_candidates_rebuilt']!=50 or not cc['cal_choice_and_serialized_weights_exact']:raise ValueError('Channel choice audit incomplete')
    if channel['n_direct_selected_predictor_calls']!=28 or channel['maximum_lane_unit_prediction_error']>1e-8 or channel_forward['n_direct_role_day_predictor_calls']!=56 or channel_forward['maximum_serialized_vs_component_error_lane_units']>1e-8:raise ValueError('Channel direct deployment incomplete')
    for data in [winter,channel]:
        stem=stems[0] if data is winter else stems[4]
        if sha256_file(Path('artifacts/research')/stem/'selection_sealed.json')!=data['selection_seal_sha256']:raise ValueError('Selection seal changed')
    if forward['selection_changed'] or channel_forward['selection_changed']:raise ValueError('Frozen forward choice changed')
    for data in [forward,channel_forward]:
        for path,digest in data['source_files_sha256'].items():
            if sha256_file(path)!=digest:raise ValueError('Public forward source changed')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),
        'n_artifacts':len(artifacts),'artifacts':artifacts,'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'Completed four-model winter remask transfer,all deployment/exposure checks,and fixed channel study. Prior complete submissions rehashed unchanged.'}
    tests={'created_UTC':now,'research_pytest':{'passed':118,'seconds':9.95,'session_id':99388,'basetemp':'E:/ieee/.cache/pytest_channel_winter_research_v2'},
        'CPU_pytest':{'passed':102,'optional_Torch_modules_skipped':5,'seconds':7.79,'basetemp':'E:/ieee/.cache/pytest_channel_winter_cpu_v2'},
        'initial_test_runs':{'research_passed':105,'CPU_passed':89,'CPU_skipped':5,'setup_errors_each':13,'cause':'Default sandbox temp directory unwritable;explicit fresh workspace basetemp fixed without assertion/source changes'},
        'compileall_passed':True,'winter_exposure_arrays':4,'winter_direct_predictor_calls':336,
        'winter_forward_saved_metrics':168,'winter_forward_direct_predictor_calls':24,
        'channel_direct_calls_original':28,'channel_direct_calls_forward':56,'channel_audit_metrics_original':112,
        'channel_cal_candidates_independently_rebuilt':50,'channel_forward_metrics':56,'channel_forward_audit_direct_calls':8,
        'channel_verifier_initial_failure':'A115 nested recurrent pickle needs Torch;original registration/source/failure retained;A115b research environment passed'}
    prior_path=root/'A97_I210_W_native_remask_ablation.json';prior=read(prior_path)
    summary=[]
    for stem,data in [('A97_I210_W_native_remask_ablation',prior),(stems[0],winter)]:
        selected=data['selection']['primary'];metrics=data['assessment'];summary.append({'study':stem,'primary':selected,
            'reference_State':metrics['prior_GRU']['S_state'],'primary_State':metrics[selected]['S_state'],
            'primary_pooled_State_gain':metrics[selected]['S_state']-metrics['prior_GRU']['S_state'],
            'source_sha256':sha256_file(root/(stem+'.json'))})
    selected=winter['selection']['primary'];fm=forward['metrics']
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:completed remask tree controls and immutable winter inputs,commit3bc0974',
        'current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,
        'new_formal_score':None,'formal_model_promoted':False,'all_previous_complete_submissions_preserved':True,
        'completed':['Winter all4fit/cal models and sealed primary','Winter fixed calendar evaluation and all independent exposure/deployment audits',
            'Cal-only fixed per-channel tree/NN ensemble and new14day evaluation','Channel raw-teacher/metrics/all50choices and serialized deployment independently checked'],
        'remask_two_corridor_old_assessment_summary':summary,
        'winter_forward':{'primary':selected,'reference_State':fm['prior_GRU']['S_state'],'primary_State':fm['primary_export']['S_state'],
            'pooled_State_gain':fm['primary_export']['S_state']-fm['prior_GRU']['S_state']},
        'channel_old_assessment':channel['assessment_comparison'],'channel_forward':channel_forward['comparisons']['primary_export'],
        'active_jobs':[],'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'platform_policy':'Local models and Git only;no publication,push or upload',
        'limits':'Public dates/reused control models,not private/final or independent control retraining. Remask adds label population. Channel policy lacks cross-direction evidence. Formal Physics/Total unknown. No new complete submission.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('phase2_checkpoint','old_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'winter_primary',selected,flush=True)


if __name__=='__main__':main()
