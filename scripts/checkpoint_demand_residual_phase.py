"""Archive fresh demand-residual controls,full replays and subspace diagnostics."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import checked,read


def main():
    reports=Path('reports/research');names=['model_artifacts_checkpoint_demand_residual_v1.json',
        'demand_residual_test_record_v1.json','research_checkpoint_demand_residual_v1.json']
    if any((reports/n).exists() for n in names):raise FileExistsError('Never overwrite demand-residual evidence')
    parent_path=reports/'model_artifacts_checkpoint_log_demand_v1.json';parent=read(parent_path)
    artifacts=[checked(row) for row in parent['artifacts']]
    folders=['D21_demand_entropy_path_residual','D22_demand_residual_independent_replay','D23_demand_residual_error_subspaces',
        'D24_demand_residual_floor_followup','D25_demand_residual_floor_independent_replay','D26_demand_residual_floor_error_subspaces']
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.pkl','.pt','.npz','.parquet','.zip','.csv'}:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),
                    'phase':'shared_demand_entropy_residual_and_shift_controls'})
    if len({r['path'] for r in artifacts})!=len(artifacts):raise ValueError('Duplicate artifacts')
    preserved=[checked(row) for row in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency lock changed')
    study=read(reports/(folders[0]+'.json'));replay=read(reports/(folders[1]+'.json'));geometry=read(reports/(folders[2]+'.json'))
    source=Path('artifacts/research')/folders[0]
    if len(study['development'])!=1890 or study['n_unique_simulated_cases']!=720:raise ValueError('Incomplete simulation study')
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256']:raise ValueError('Selection seal changed')
    if read(source/'selection_sealed.json')!=study['selection']:raise ValueError('Choice changed')
    if (source/'experiment.json').read_bytes()!=(reports/(folders[0]+'.json')).read_bytes():raise ValueError('Source mirror changed')
    if replay['registered']['protocol']['source_study_sha256']!=sha256_file(reports/(folders[0]+'.json')):
        raise ValueError('Replay study changed')
    if replay['n_cases_regenerated']!=720 or not all(replay[k] for k in ['all_data_exact','all_direct_API_predictions_exact']):
        raise ValueError('Incomplete data/API verification')
    if len(geometry['ranks'])!=10 or geometry['max_relative_count_residual_of_null_component']>1e-9:
        raise ValueError('Subspace diagnostic failed')
    if geometry['registered']['protocol']['source_study_sha256']!=sha256_file(reports/(folders[0]+'.json')):
        raise ValueError('Geometry study changed')
    successful=sum(r['success'] for r in study['development']+study['assessment'])
    if replay['n_successful_score_records_recalculated']!=successful:raise ValueError('Incomplete independent scores')
    followup=read(reports/(folders[3]+'.json'));replay2=read(reports/(folders[4]+'.json'));geometry2=read(reports/(folders[5]+'.json'))
    followup_source=Path('artifacts/research')/folders[3]
    if len(followup['development'])!=1890 or followup['n_unique_simulated_cases']!=720 or followup['n_reused_development_predictions']!=1350:
        raise ValueError('Incomplete followup or incorrectly counted reused controls')
    if followup['registered']['protocol']['source_development_sha256']!=sha256_file(source/'development_progress.json'):
        raise ValueError('Followup source development changed')
    if followup['registered']['protocol']['source_selection_seal_sha256']!=sha256_file(source/'selection_sealed.json'):
        raise ValueError('Followup original choice source changed')
    if sha256_file(followup_source/'selection_sealed.json')!=followup['selection_seal_sha256'] or read(followup_source/'selection_sealed.json')!=followup['selection']:
        raise ValueError('Followup seal changed')
    if (followup_source/'experiment.json').read_bytes()!=(reports/(folders[3]+'.json')).read_bytes():raise ValueError('Followup mirror changed')
    if replay2['n_cases_regenerated']!=720 or not replay2['all_data_exact'] or not replay2['all_direct_API_predictions_exact']:
        raise ValueError('Incomplete followup direct replay')
    successful2=sum(r['success'] for r in followup['development']+followup['assessment'])
    if replay2['n_successful_score_records_recalculated']!=successful2:raise ValueError('Incomplete followup independent scores')
    for check in [replay2,geometry2]:
        if check['registered']['protocol']['source_study_sha256']!=sha256_file(reports/(folders[3]+'.json')):
            raise ValueError('Followup diagnostic source changed')
    if len(geometry2['ranks'])!=10 or geometry2['max_relative_count_residual_of_null_component']>1e-9:
        raise ValueError('Followup subspace diagnostic failed')
    if study['selection']['development_gate_passed'] or followup['selection']['development_gate_passed']:
        raise ValueError('Unexpected model promotion claim')
    components=read(reports/'D27_demand_residual_component_audit.json')
    if [r['study_sha256'] for r in components['studies']]!=[sha256_file(reports/(folders[i]+'.json')) for i in [0,3]]:
        raise ValueError('Incomplete/stale component diagnostic')
    if components['pooled_identical_hypothesis']['n_fresh_research_seeds']!=6:raise ValueError('Pooled seeds incomplete')
    unique_cases={(r['panel'],r['seed'],r['regime']) for s in [study,followup] for r in s['case_files']}
    if len(unique_cases)!=1170:raise ValueError('Duplicated development cases miscounted as fresh')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'purpose':'Two720-case known-truth studies,full direct API/metric replay,public SVD bases and all original candidates',
        'limitations':'Fresh synthetic stress tests only;no actual hidden OD/Physics/Total or posterior guarantee.'}
    tests={'created_UTC':now,'research_pytest':{'passed':93,'seconds':12.07,'session_id':49571,
        'basetemp':'E:/ieee/.cache/pytest_demand_residual_floor_research'},
        'CPU_pytest':{'passed':81,'skipped_optional_torch_modules':4,'seconds':8.96,'session_id':59998,
        'basetemp':'E:/ieee/.cache/pytest_demand_residual_floor_cpu'},'compileall_passed':True,
        'earlier_pre_floor_tests':{'research_passed':92,'CPU_passed':80,'CPU_Torch_skips':4},
        'n_case_inputs_independently_regenerated_exact':replay['n_cases_regenerated']+replay2['n_cases_regenerated'],
        'n_unique_case_inputs_excluding_reused_development':len(unique_cases),
        'n_reused_D24_development_predictions':1350,
        'n_successful_scores_independently_recalculated':successful+successful2,
        'maximum_score_error':max(replay['max_independent_score_error'],replay2['max_independent_score_error']),
        'n_direct_uncached_API_calls_exact':replay['n_direct_API_method_case_replays']+replay2['n_direct_API_method_case_replays'],
        'rowspace_decompositions_checked':geometry['n_unique_case_prediction_records']+geometry2['n_unique_case_prediction_records'],
        'new_formal_actions':False,'scope':'Zero-alpha exact strong reference,primal numeric oracle,invalid input,generator determinism and complete study replay'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'objective_status':'active;formal 0.95000+ unachieved','current_goal_turn_classification':'progress',
        'previous_goal_turn_classification':'progress:1e43150,positive shared OD demand and complete K07 control',
        'formal_best_Total':.82940,'formal_best_preserved':True,'new_formal_score':None,
        'platform_policy':'User directs local modeling;no publication,push or upload',
        'completed':['Independent shared demand center plus entropy path correction,with exact strong-reference endpoint',
            'Four development-only settings,270problems/1890method records and sealed choice',
            'Two three-seed research batches;540same-law and360new-law assessment cases',
            '1440case NPZ records,1170unique input cases;all inputs and successful scores independently regenerated/recalculated',
            'Preregistered development-only center-floor followup rejected;1350original controls explicitly reused',
            'Direct uncached API deployment replay and conditional leave-one-family-out selection sensitivity',
            'Ten public counted operators,orthogonal path-error rowspace/nullspace diagnostics repeated on two batches',
            'Secondary pooled six-seed identical-hypothesis/component diagnostic;both failed primary gates retained'],
        'terminal_jobs':[{'session_id':sid,'task':task,'exit_code':0} for sid,task in [
            (21882,'D21 shared demand residual study'),(89999,'D24 preregistered center-floor followup'),
            (49243,'D22 full data/metric/direct API replay'),(95971,'D23 SVD diagnostics'),
            (88523,'D25 followup full replay'),(41614,'D26 followup SVD diagnostics'),
            (49571,'93 research tests'),(59998,'81 CPU tests')]],
        'active_jobs_at_checkpoint':[],'new_formal_model_promoted':False,
        'development_model_gate_passed':False,'floor_followup_development_gate_passed':False,
        'simulation_comparisons':{folders[0]:study['comparisons'],folders[3]:followup['comparisons']},
        'optimizer_failure_records_preserved':study['optimizer_case_failure_records']+followup['optimizer_case_failure_records'],
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'next_research_priorities':['Let matched strong-reference and fresh-shift evidence decide further OD architecture investment',
            'Keep prior-driven nullspace error distinct from detector fit and from official OD accuracy',
            'Focus next substantial State research on latent dynamics/abrupt congestion and recovery,with strong matched LGB controls'],
        'limitations':'Prior model family was informed by earlier research;new seeds/laws remain own synthetic assumptions. '
        'Counts used in both stages;no calibrated posterior or formal gain. Model choice does not use fresh assessment.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (reports/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('previous_unchanged',len(parent['artifacts']),'total_artifacts',len(artifacts),'direct_API_exact',tests['n_direct_uncached_API_calls_exact'],flush=True)


if __name__=='__main__':main()
