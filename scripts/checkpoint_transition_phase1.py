"""Freeze fit diagnostics and objective code while retaining a live training handle."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import zipfile
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-session',type=int,required=True);parser.add_argument('--last-live-poll-utc',required=True);args=parser.parse_args()
    root=Path('reports/research');names=['model_artifacts_checkpoint_transition_phase1_v1.json','transition_phase1_test_record_v1.json','research_checkpoint_transition_phase1_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite transition checkpoint')
    parent_path=root/'model_artifacts_checkpoint_flow_transfer_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']]
    known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Locks changed')
    audit_path=root/'A122_fit_transition_population_audit.json';audit=read(audit_path);proof=read(root/'A123_fit_transition_independent_verification.json')
    if proof['registered']['protocol']['source_study_sha256']!=sha256_file(audit_path) or proof['n_raw_fit_days']!=245 or proof['n_diagnostic_arrays_verified']!=3185 or proof['n_day_population_or_exposure_records_verified']!=3430:raise ValueError('Fit diagnostic verification incomplete')
    def append(path,phase):
        if path.as_posix() not in known:
            artifacts.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),'phase':phase});known.add(path.as_posix())
    for stem in ['A122_fit_transition_population_audit','A123_fit_transition_independent_verification']:
        for p in sorted((Path('artifacts/research')/stem).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip'}:append(p,'completed_public_fit_transition_diagnostics')
    training=Path('artifacts/research/A124_I5_N_speed_jump_pair_loss');archive=training/'source_snapshot.zip'
    if (root/'A124_I5_N_speed_jump_pair_loss.json').exists():raise ValueError('Model study already complete;make completed checkpoint instead')
    with zipfile.ZipFile(archive) as z:
        for p in ['src/trafficflow/speed_jump_loss.py','scripts/train_speed_jump_ablation.py']:
            if z.read(p)!=Path(p).read_bytes():raise ValueError('Running objective/trainer source differs from its start archive')
    append(archive,'immutable_start_archive_of_live_two_seed_training')
    for path,digest in audit['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Raw fit/cache/exposure/network changed')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),
        'n_artifacts':len(artifacts),'artifacts':artifacts,'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'live_job_policy':'Only immutable start archive added. All A124 checkpoints,progress,exports and counters excluded until whole study completes.'}
    tests={'created_UTC':now,'research_pytest':{'passed':122,'seconds':10.03,'session_id':19666,'basetemp':'E:/ieee/.cache/pytest_speed_jump_research_v1'},
        'CPU_pytest':{'passed':104,'optional_Torch_modules_skipped':6,'seconds':7.99,'basetemp':'E:/ieee/.cache/pytest_speed_jump_cpu_v1'},
        'new_kernel_tests':{'passed':4,'seconds':3.09,'scope':'Persistent versus isolated/invalid label patterns;manual jump endpoint gradients and exact control objective when unpaired'},
        'compileall_passed':True,'same_object_frozen_model_save_probe_bytes_identical':True,
        'fit_raw_days_rebuilt':245,'all_diagnostic_arrays_exact':3185,'population_exposure_records_verified':3430,
        'maximum_absolute_count_or_SSE_difference':proof['maximum_absolute_statistic_difference'],'model_validation':None}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:completed negative winter flow transfer,commit325e4f1','current_goal_turn_classification':'progress',
        'objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,
        'completed':['Fit-only transient population/base-error/exposure audit on245days','Independent all-array and3430record verification',
            'One fixed .05 speed-pair objective with gradient/missing-mask tests','Two-seed matched control study preregistered and running'],
        'new_model_primary':None,'new_model_assessment':None,'new_model_forward':None,
        'active_jobs':[{'session_id':args.training_session,'task':'A124_I5_N_speed_jump_pair_loss','last_tool_poll_confirmed_live_UTC':args.last_live_poll_utc,
            'policy':'Poll this exact handle first. Do not restart due observation timeout. Both seeds/selection seal required before assessment.'}],
        'resume_steps':['Poll A124 unified-exec handle and inspect actual terminal state,do not duplicate fitting',
            'After terminal0 and final A124 JSON,evaluate_exported_state_forward.py -> A125_I5_N_speed_jump_forward_calendar.json',
            'verify_speed_jump_training.py -> A126_I5_N_speed_jump_training_verification.json',
            'verify_native_remask_forward.py on A125 -> A127_I5_N_speed_jump_forward_independent_verification.json',
            'Use actual whole-study results for comparison;do not infer accuracy from fit audit or partialcal'],
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only',
        'all_previous_complete_submissions_preserved':True,'limits':'Raw observed fit diagnostics do not identify latent noise/events. Input-base SSE is not neural error/official State. Ongoing model results unknown.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('transition_phase1_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'live_session',args.training_session,flush=True)


if __name__=='__main__':main()
