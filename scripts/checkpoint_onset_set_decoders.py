"""Freeze matched frozen-score decoder study and explicit reused input audit."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_onset_set_v1.json','onset_set_verification_record_v1.json','research_checkpoint_onset_set_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite onset-set checkpoint')
    parent_path=root/'model_artifacts_checkpoint_onset_rank_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    stems=['C68_onset_threshold_cardinality','C69_onset_threshold_cardinality_verification','C70_onset_decoder_matched_summary'];study,audit,summary=[read(root/(s+'.json')) for s in stems];cfg=study['registered']['protocol']
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or summary['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or summary['independent_audit_sha256']!=sha256_file(root/(stems[1]+'.json')):raise ValueError('New source evidence changed')
    old_proof=read(cfg['reused_input_proof'])
    if sha256_file(cfg['reused_input_proof'])!=cfg['reused_input_proof_sha256'] or study['reused_raw_input_window_verification']!=2345 or study['new_raw_input_rebuild_in_this_experiment']!=0:raise ValueError('Reused audit scope differs')
    for path,digest in {**cfg['unchanged_source_constructors_sha256'],**old_proof['source_files_sha256']}.items():
        if sha256_file(path)!=digest:raise ValueError('Reused public raw/masked inputs or feature constructor changed')
    if len(study['cases'])!=4 or len(audit['cases'])!=4 or not all(r['new_summaries_fit_targets_ridge_decoders_gates_exact_within_tolerance'] for r in audit['cases']):raise ValueError('Case verification incomplete')
    if audit['n_new_onset_summary_window_occurrences']!=665 or audit['maximum_summary_difference']>1e-10 or audit['maximum_count_fit_parameter_difference']>1e-9:raise ValueError('Independent fit summary/Ridge replay incomplete')
    if audit['n_direct_frozen_classifier_booster_calls']!=56 or audit['n_new_fresh_public_history_windows']!=112 or audit['n_portable_role_window_calls']!=896 or audit['n_additional_shuffled_target_calls']!=448 or audit['n_future_history_poison_rejections']!=16:raise ValueError('Portable decoder verification incomplete')
    if summary['n_matched_contrasts_independently_verified']!=16:raise ValueError('Matched comparisons incomplete')
    for stem in stems:
        for p in sorted((Path('artifacts/research')/stem).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip','.pkl','.txt','.parquet'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'completed_frozen_onset_decoder_comparison'});known.add(p.as_posix())
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),
        'n_artifacts':len(artifacts),'artifacts':artifacts,'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,
        'formal_best_submission_sha256':parent['formal_best_submission_sha256'],'scope':'Frozen classifier threshold/count decoder study,8Ridge+4scalarcontrols,portable policies and audits. Original classifiers and complete submissions unchanged.'}
    verification={'created_UTC':now,'new_unit_tests':{'passed':3,'seconds':1.33,'scope':'Analytic window summaries/permutations,fit-only count/coverage correction/bounds,threshold/force/full-domain separation'},
        'research_pytest':{'passed':134,'seconds':12.31,'session_id':84531,'basetemp':'.cache/pytest_onset_set_research_v1'},
        'CPU_pytest':{'passed':113,'optional_Torch_modules_skipped':7,'seconds':7.37,'basetemp':'.cache/pytest_onset_set_CPU_v1'},'compileall_passed':True,
        'old_raw_window_audit_reused':2345,'new_derived_summary_windows_verified':665,'new_public_history_windows_for_portable_checks':112,
        'count_parameter_maximum_difference':audit['maximum_count_fit_parameter_difference'],'frozen_classifier_booster_calls':56,'portable_window_calls':896,'additional_shuffled_calls':448,'future_history_rejections':16,
        'new_matched_day_block_contrasts_verified':16,'scope':'No new classifier training or repeat of all2345raw inputs. Twelve count states:8Ridge/4intercept. Some tests/verification concurrent;not isolated speed benchmark.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:public onset ranking study rejected and independently verified,commit585d6b8','current_goal_turn_classification':'progress',
        'objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,
        'completed':['Frozen onset classifier:variable threshold versus fixedk and causal count controls','All new summary/target/Ridge/decoder/gate recipes independently verified,old raw proof reused explicitly','One nonzero research selection observed;assessment and cross-case evidence do not support formal promotion'],
        'cases':[{'source_study':c['source_study'],'decision_selected':c['decision_selected'],'selected':c['selected'],'assessment_comparisons':c['comparisons']} for c in study['cases']],
        'selected_evidence_gate':summary['selected_policy_evidence_gate'],'threshold_secondary_gate':summary['threshold_secondary_evidence_gate'],'active_jobs':[],'new_complete_submission':False,
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only',
        'limits':'Previously viewed noisy public proxy calendars,not latent truth/final/private. MCAR count correction unverified. Threshold gains over fixedk do not prove gains over the strong existing pipeline. One research selected policy has uncertain assessment gain;no formal Queue/Total.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('onset_set_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),flush=True)


if __name__=='__main__':main()
