"""Freeze matched dense warm-up models, input/model audits and negative controls."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import zipfile
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_dense_pretrain_v1.json','dense_pretrain_verification_record_v1.json','research_checkpoint_dense_pretrain_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite dense warm-up checkpoint')
    parent_path=root/'model_artifacts_checkpoint_local_dense_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    stems=['C87_dense_dynamics_pretraining','C88_dense_pretraining_data_verification','C89_dense_pretraining_model_verification','C89b_dense_pretraining_model_verification','C90_dense_pretraining_matched_summary'];study,data,audit,summary=[read(root/(s+'.json')) for s in [stems[0],stems[1],stems[3],stems[4]]]
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or audit['registered']['protocol']['source_data_audit_sha256']!=sha256_file(root/(stems[1]+'.json')) or summary['independent_audit_sha256']!=sha256_file(root/(stems[3]+'.json')):raise ValueError('Model/data/summary chain differs')
    expected={'network_fits_checked':12,'checkpoint_neural_window_forwards':792,'evaluation_neural_window_forwards':1896,'new_eval_raw_window_occurrences':316,'decision_grid_score_checks':1128,'portable_history_windows':56,'portable_role_calls':336,'additional_shuffled_calls':168,'future_history_rejections':12}
    if audit['totals']!=expected or audit['maximum_probability_difference']!=0 or audit['maximum_normalized_speed_difference']!=0:raise ValueError('Model audit incomplete')
    if data['totals']!={'fit_day_packs':124,'dense_origins':33480,'event_cache_windows':742,'actual_future_rawT_poison_checks':372,'event_cache_records':4}:raise ValueError('Dense input audit incomplete')
    source_files=dict(data['source_files_sha256'])
    for c in study['cases']:source_files.update(c['source_files_sha256'])
    for p,d in source_files.items():
        if sha256_file(p)!=d:raise ValueError('Public input changed')
    if study['new_network_fits']!=12 or any(c['selected']!='incumbent' for c in study['cases']) or study['formal_model_promoted'] or study['new_complete_submission']:raise ValueError('Unexpected model promotion')
    # Initial incomplete audit is retained. Its terminal stderr was lost in a
    # truncated tool result; do not manufacture a traceback or a measured score.
    failed_folder=Path('artifacts/research')/stems[2];old_snapshot=failed_folder/'source_snapshot.zip'
    with zipfile.ZipFile(old_snapshot) as archive:old_source=archive.read('scripts/verify_dense_pretrain_models.py').decode('utf-8')
    if 'released_origin_rows(release' not in old_source or 'from trafficflow.causal_queue import released_origin_rows' in old_source:raise ValueError('Incomplete audit source diagnosis differs')
    incomplete={'status':'incomplete;not accepted as verification','registered_sha256':sha256_file(root/(stems[2]+'.registered.json')),'source_snapshot_sha256':sha256_file(old_snapshot),
        'observed_partial_artifact':'artifacts/research/C89_dense_pretraining_model_verification/I5S_September_decision_4133_members.npz',
        'source_diagnosis':'Saved verifier calls released_origin_rows at portable-history replay but lacks its import. Fixed verifier import only and reran as C89b;models,training,batches,thresholds,seals and tolerances unchanged.',
        'terminal_status':'Original terminal traceback/exit status unavailable after tool output truncation;no traceback fabricated. No running Python process or completed verification report existed on recovery.',
        'superseded_by':{'path':(root/(stems[3]+'.json')).as_posix(),'sha256':sha256_file(root/(stems[3]+'.json'))},'formal_Queue':None,'formal_Total':None}
    for p in [root/(stems[2]+'.incomplete.json'),failed_folder/'incomplete.json']:
        if p.exists():
            if read(p)!=incomplete:raise ValueError('Incomplete audit note changed')
        else:
            with p.open('x',encoding='utf-8') as f:json.dump(incomplete,f,indent=2,allow_nan=False)
    for stem in stems:
        for p in sorted((Path('artifacts/research')/stem).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip','.pkl','.pt','.txt','.parquet'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'dense_warmup_matched_models_no_promotion'});known.add(p.as_posix())
    if len(known)!=len(artifacts):raise ValueError('Duplicate model artifact')
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'All12newnetwork fits,initials/batches,48weight+optimizer+checkpointforecast snapshots,12portable role exports,causal dense and event data,independent audits and posthoc matched summaries. First incomplete verifier preserved. No new complete submission or formal promotion.'}
    verification={'created_UTC':now,'new_kernel_tests':{'passed':4,'seconds':4.50,'scope':'Input units/maskedT,actual future+rawT poison,all270history-onlyorigins/bounds,analytic loss/gradient masks'},
        'research_pytest':{'passed':151,'seconds':13.39,'session_id':53113,'basetemp':'.cache/pytest_dense_queue_research_v2'},'CPU_pytest':{'passed':129,'optional_Torch_modules_skipped':8,'seconds':10.31,'session_id':25851,'basetemp':'.cache/pytest_dense_queue_CPU_v2'},'compileall_passed':True,
        'previous_test_launches':'First full-suite launch results unavailable after tool output truncation;no running Python process on recovery. New v2 suites terminal0 verified;no additional training launched.',
        'independent_dense_input_checks':data['totals'],'independent_model_checks':expected,'source_files_rehashed':len(source_files),'independent_probability_maximum_difference':0.,'independent_normalized_speed_maximum_difference':0.,
        'joint_summary_comparisons':7,'single_member_diagnostic_comparisons':4,'scope':'Fullnew neuraldata audit;all checkpoints/inference/thresholdgrids/gates and actual exposure replayed. Existing strongtree featureproof reused explicitly. No independent re-optimization of12networks,private score or blind final test.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'previous_goal_turn_classification':'progress:ab60160 C82-C86 complete','current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,'new_complete_submission':False,'active_jobs':[],
        'completed':['12matched scratch/eventwarm/densewarm graphnetwork fits,common initialization/fine batches and equal warmcompute control','All dense data and neural checkpoints/inference/gates independently audited;original incomplete verifier archived','Dense vs weaker eventwarm posthoc gain+.009148,but strongincumbent gap-.023199 and both primary choices unchanged'],
        'cases':[{'name':c['name'],'decision_selected':c['decision_selected'],'selected':c['selected'],'selection_gain':c['selection_gate']['paired_macro_gain']} for c in study['cases']],
        'next_research':'Use matched dense inputs to test joint continuous speed/flow dynamics and missing-aware latent-state propagation with explicit flow head;keep identical data/compute controls. Do not expand thresholds on these repeatedly viewed dates or treat lower Brier as Queue/State/Physics gain.',
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only','limits':'Previously viewed public noisy proxy dates,not latent/official Queue/private/final. Shared calendar uncertainty and optimizer seeds are separate. Extra unique teacher cells overlap in time;no additional rare event completeness. Formal0.95000+unmet.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    print('dense_pretrain_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'sources',len(source_files),flush=True)


if __name__=='__main__':main()
