"""Freeze completed jump-loss outcomes and static graph-scope diagnostics."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from trafficflow.research_eval import paired_block_interval
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_transition_phase2_v1.json','transition_phase2_verification_record_v1.json','research_checkpoint_transition_phase2_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite transition outcome checkpoint')
    parent_path=root/'model_artifacts_checkpoint_transition_phase1_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']]
    known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Locks changed')
    stems=['A124_I5_N_speed_jump_pair_loss','A125_I5_N_speed_jump_forward_calendar','A126_I5_N_speed_jump_training_verification',
        'A127_I5_N_speed_jump_forward_independent_verification','A128_public_graph_coverage_audit','A129_public_graph_independent_verification']
    study,forward,check,fcheck,graph,gcheck=[read(root/(s+'.json')) for s in stems]
    for verifier,source in [(check,stems[0]),(fcheck,stems[1]),(gcheck,stems[4])]:
        if verifier['registered']['protocol']['source_study_sha256']!=sha256_file(root/(source+'.json')):raise ValueError('Verifier source changed')
    if len(check['exposure_checks'])!=2 or not all(r['initial_patch_and_full_cell_exposure_equal_control'] and r['new_pair_exposure_exact'] for r in check['exposure_checks']):raise ValueError('Matched supervision not verified')
    if check['n_direct_predictor_calls']!=196 or check['n_saved_role_day_metrics_recalculated']!=168 or check['maximum_prediction_error_lane_units']>1e-8:raise ValueError('Model deployment verification incomplete')
    if forward['n_direct_predictor_calls']!=98 or forward['selection_changed'] or fcheck['n_raw_teacher_days_rebuilt']!=14 or fcheck['n_saved_role_day_metrics_recalculated']!=98 or fcheck['n_direct_predictor_calls']!=14 or fcheck['maximum_prediction_difference_lane_units']>1e-8:raise ValueError('Forward verification incomplete')
    if gcheck['n_panels_verified']!=10:raise ValueError('Static graph verification incomplete')
    note=read(root/'A128_attention_scope_note.json')
    if sha256_file(root/'A128_attention_scope_note.json')!=sha256_file(Path('artifacts/research')/stems[4]/'attention_scope_note.json'):raise ValueError('Attention scope artifact differs')
    if note['source_graph_audit_sha256']!=sha256_file(root/(stems[4]+'.json')) or note['finite_attention_mask_cells']!=note['attention_mask_cells'] or note['gradient_to_other_isolated_node_L1']<=0:raise ValueError('Global attention scope note incompatible')
    for stem in stems:
        for p in sorted((Path('artifacts/research')/stem).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip','.pt','.pkl'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'completed_jump_loss_and_graph_scope'});known.add(p.as_posix())
    for report in [study,forward,graph]:
        for path,digest in report['source_files_sha256'].items():
            if sha256_file(path)!=digest:raise ValueError('Public sources changed')
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,
        'formal_best_submission_sha256':parent['formal_best_submission_sha256'],'scope':'Completed two jump-loss fits,all exposures/deployment/metrics and static graph audits;previous artifacts unchanged'}
    paired=[]
    for phase,metrics in [('old_assessment',study['assessment']),('new_calendar',forward['metrics'])]:
        for seed in study['registered']['protocol']['seeds']:
            a=metrics[f'control_seed{seed}'];b=metrics[f'jump_seed{seed}'];paired.append({'phase':phase,'seed':seed,
                'pooled_State_gain_vs_same_seed_control':b['S_state']-a['S_state'],
                'day_mean_diagnostic':paired_block_interval([r['daily_State'] for r in a['days']],[r['daily_State'] for r in b['days']]),
                'scope':'Existing frozen perseed policy contrast;checkpoint/alpha may differ,seeds share dates. No additional model/event.'})
    verification={'created_UTC':now,'compileall_passed':True,'unit_suite_reference':'transition_phase1_test_record_v1.json:122research,104CPU/6Torch skips,unchanged production src since phase1',
        'n_independent_cell_and_pair_counter_archives':4,'n_direct_cal_ass_predictor_calls':196,'n_cal_ass_metrics_recalculated':168,
        'n_direct_forward_predictor_calls':98,'n_forward_metrics_recalculated':98,'n_additional_forward_audit_predictor_calls':14,
        'maximum_cal_ass_prediction_error_lane_units':check['maximum_prediction_error_lane_units'],'maximum_forward_score_error':fcheck['maximum_score_error'],
        'graph_panels_independently_verified':10,'attention_scope_probe_replayed_exactly':True,'attention_probe_gradient_L1':note['gradient_to_other_isolated_node_L1']}
    verification['paired_seed_policy_diagnostics']=paired
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:fit population audit,new loss kernel and live fitting,commit3807356','current_goal_turn_classification':'progress',
        'objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,
        'completed':['Both speed-jump models and ordinarycal gate','All original cell/pair supervision independently reconstructed',
            'Frozen new14day calendar and all exports independently checked','Published graph isolation counted and independently checked;global attention scope corrected'],
        'primary':study['selection']['primary'],'old_assessment_jump_comparison':study['comparisons']['jump_seed_mean'],
        'new_calendar_jump_comparison':forward['comparisons']['jump_seed_mean'],'active_jobs':[],
        'next_research':'Consider explicit static relative-distance attention prior only as separate modeling hypothesis;existing attention is already global. Do not relabel graph isolation a reader failure.',
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only',
        'all_previous_complete_submissions_preserved':True,'limits':'One fixed loss coefficient/update budget failed;does not refute all temporal losses. Cal checkpoint/alpha differ. Public observations,not latent events/private/final. No new complete submission or formal Physics/Total.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('transition_phase2_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'primary',study['selection']['primary'],flush=True)


if __name__=='__main__':main()
