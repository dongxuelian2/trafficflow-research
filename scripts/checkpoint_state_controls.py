"""Freeze rejected State specialization controls and numerical-profile repair."""
import json
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_state_controls_v1.json','state_controls_verification_record_v1.json','research_checkpoint_state_controls_v1.json']
    if any((root/name).exists() for name in names):raise FileExistsError('Never overwrite State controls checkpoint')
    parent_path=root/'model_artifacts_checkpoint_joint_dynamics_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    study_path=root/'C95_state_specialization_controls.json';audit_path=root/'C96b_state_controls_verification.json';summary_path=root/'C97_state_controls_factor_summary.json';study=read(study_path);audit=read(audit_path);summary=read(summary_path);diagnostic_path=root/'C96_state_precision_diagnostic.json';diagnostic=read(diagnostic_path)
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(study_path) or summary['source_study_sha256']!=sha256_file(study_path) or summary['source_audit_sha256']!=sha256_file(audit_path) or diagnostic['source_study_sha256']!=sha256_file(study_path):raise ValueError('State evidence source chain differs')
    expected={'schedules':4,'matched_teacher_rows':343254,'matched_tree_field_models':24,'new_networks':4,'new_NN_snapshots':16,'old_joint_snapshot_CP_records':16,'full_CP_contexts':8280,'event_CP_contexts':33,'CP_member_forwards':133008,'CP_full_FP32_member_forwards':133008,'phase_contexts':16560,'phase_days':60,'phase_member_role_contexts':281520,'phase_unique_full_FP32_NN_contexts':107640,'gradient_pairs':32,'portable_history_windows':12,'portable_role_calls':108,'portable_shuffle_calls':36,'portable_timezone_calls':36,'portable_future_rejections':18,'independent_NN_refits':1,'independent_tree_field_refits':2}
    if audit['totals']!=expected or audit['maximum_differences']['prediction']!=0. or audit['independent_NN_refit']['maximum_weight_difference']!=0. or not audit['full_FP32_checkpoint_choices_unchanged']:raise ValueError('Accepted State replay/refit incomplete')
    if audit['portable_full_FP32_maximum_difference']>1e-5 or audit['maximum_differences']['feature_semantics']>2e-6:raise ValueError('Original numerical contracts exceeded')
    if study['decision_screen']['shortlisted'] or study['selected']!='full_tree' or study['selection_gate'] is not None or study['later_phases_read'] or study['formal_model_promoted'] or study['new_complete_submission']:raise ValueError('Unexpected State promotion/later reads')
    for case in study['cases']:
        if set(case['phase_records'])!={'decision'} or any(r['phase'] not in ['checkpoint','decision'] for r in case['phase_reads']):raise ValueError('Rejected State study read newlaterperiods')
        for path,d in case['source_files_sha256'].items():
            if sha256_file(path)!=d:raise ValueError('Public State source changed')
    failure_path=root/'C96_state_controls_verification.failed.json';failure=read(failure_path)
    if 'Independent State prediction differs' not in failure['traceback']:raise ValueError('Preserved precision failure differs')
    old_source=next(r for r in study['source_snapshot']['captured_files'] if r['path']=='src/trafficflow/state_control_model.py');new_source=next(r for r in audit['source_snapshot']['captured_files'] if r['path']=='src/trafficflow/state_control_model.py')
    if sha256_file(new_source['path'])!=new_source['sha256']:raise ValueError('Corrected FP32 source changed')
    repair={'old_model_source':old_source,'corrected_model_source':new_source,'original_failed_audit':{'path':failure_path.as_posix(),'sha256':sha256_file(failure_path)},'diagnostic_report':{'path':diagnostic_path.as_posix(),'sha256':sha256_file(diagnostic_path)},
        'issue':'C95GPUcuDNNdefaultTF32 givesbatch-size-dependentFP32rounding with identicalinputs. FirstC96strict1e-5portablecomparisonfailed;notaccepted. Repeat48firstrow equalsactual48firstrow;notcross-exampleleakage.',
        'correction':'StateControlModel inference locally sets cuDNNallow_tf32=False and restores backend;no trainedweights,originalCPchoices,trees,sourcearrays ordecisionthreshold changed. CoreTF32researcharraysandcorrectedfullFP32runtime auditedseparately with original1e-5checks,not relaxed tolerance.',
        'research_profile_maximum_replay_error':audit['maximum_differences']['prediction'],'portable_full_FP32_maximum_error':audit['portable_full_FP32_maximum_difference'],'all_decision_context_profile_maximum_change_by_field':audit['TF32_to_full_FP32_normalized_maximum_change_by_field'],'measured_State_point_profile_change_bound':audit['TF32_to_FP32_State_point_change_bound'],'all_CP_choices_and_global_gate_unchanged':True,
        'full_FP32_day_arrays':'SeparateC96bartifacts;originalC95arraysnotmodified. Bound applies tothesecheckedpublicdecisioncontexts/models,not hidden data or a new formal score. Snapshot captures originalTF32source for exactlegacyreplay;currentportablecodeusescorrectedprofile.'}
    repair_folder=Path('artifacts/research/C96_precision_repair');repair_folder.mkdir(exist_ok=True)
    for target in [root/'C96_state_precision_repair.json',repair_folder/'record.json']:
        if target.exists():
            if read(target)!=repair:raise ValueError('Precision repair record changed')
        else:
            with target.open('x',encoding='utf-8') as f:json.dump(repair,f,indent=2,allow_nan=False)
    folders=sorted(p for p in Path('artifacts/research').iterdir() if p.is_dir() and p.name.startswith(('C95','C96','C97')))
    for folder in folders:
        for source,target in [(root/(folder.name+'.run.log'),folder/'terminal.run.log'),(root/(folder.name+'.failed.json'),folder/'failure.json')]:
            if source.exists():
                if not target.exists():shutil.copyfile(source,target)
                if sha256_file(source)!=sha256_file(target):raise ValueError('Preserved failure/terminal changed')
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.npz','.zip','.pkl','.pt','.txt','.parquet','.log'} and path.as_posix() not in known:
                artifacts.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),'phase':'State_specialization_factor_screen_rejected_FP32_repaired'});known.add(path.as_posix())
    if len(known)!=len(artifacts):raise ValueError('Duplicate State artifact')
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'4newStateonlyNN/24matchedtreefieldfits plusreselected4existingjointfits,CP/schedules/exposure/decisionarrays,exports,failedprecisionauditanddiagnosis,newfullFP32dayarrays andacceptedindependentNN/tree refits. Old5completeCSVs/modelsunchanged;no new complete submission or formal score.'}
    verification={'created_UTC':now,'initial_factor_kernel_tests':{'passed':5,'seconds':3.61},'post_FP32_kernel_tests':{'passed':6,'seconds':3.51},'research_pytest':{'passed':164,'seconds':16.45,'session_id':30928,'basetemp':'.cache/pytest_state_controls_research_v2'},'CPU_pytest':{'passed':136,'optional_Torch_modules_skipped':10,'seconds':13.02,'session_id':37552,'basetemp':'.cache/pytest_state_controls_CPU_v2'},'compileall_passed':True,
        'accepted_independent_audit':expected,'maximum_differences':audit['maximum_differences'],'portable_FP32_error':audit['portable_full_FP32_maximum_difference'],'source_files_rehashed':audit['source_files_rehashed'],'independent_scalar_posthoc_comparisons':20,
        'scope':'OnepredeclaredNNand2fieldtrees independentlyreoptimized,logicalNNweightdifference0/treeboostertextexact. Remaining3NN/22fieldtreesnotreoptimized. 343254teacherrowscountseed-specificduplicates,281520role-contextsincludealiasesandtrees,notindependentphysicalGPUforwards. Failedfirstauditretained;no formal/latent/blindscore.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'previous_goal_turn_classification':'progress:28aa20c C91-C94 complete','current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,'new_complete_submission':False,'active_jobs':[],
        'completed':['Matchedactualcurrentlabelsets andnominalupdateweights;4State-onlyNN and24causaltreecontrols','Selectiontarget/CPcoverage/reweighting/L1vsL2 differences measured ondecision;allcandidatefamilies rejectedbeforeadditionalheldperiods','OneNN/twotrees independentlyreoptimized;TF32singlebatchmismatch fixed in portable FP32runtime with separateimmutablelegacyprofile replay'],
        'decision_screen':study['decision_screen'],'new_selection_metrics':None,'new_assessment_metrics':None,'new_late_transfer_metrics':None,'next_research':'ProspectivelycompareState-onlyL1/window versusL2/window versusL2importance-correcteduniquelabelrisk,withsameinit/schedules/exactlabelunion. Correctioncancelscontextmass/inverse-window-countwhilepreservingtotalnominalweight;retainL2weighted/unweightedtreecontrols. TestNNdirectly;treeobjectiveeffectdoesnotproveNNgain.',
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models/Git only','limits':'Decisionperiodisalreadyviewedpublicdevelopment,posthocuncorrectedintervalsnotfinaltest. No newlater raw/arrays for rejectedcandidates;existingoldreportscontainpreviouslyseenresults. StrongcausalStatecontrolisnot formalbestwholeTask1model. Formal0.95000+unmet.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    print('State_controls_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'sources',audit['source_files_rehashed'],flush=True)


if __name__=='__main__':main()
