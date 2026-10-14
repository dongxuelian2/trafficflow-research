"""Archive frozen ensembles, causal API replays and known-latent diagnostics."""
import ast
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.ctm_scenarios import CASES
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import checked,read


def main():
    reports=Path('reports/research');names=['model_artifacts_checkpoint_queue_ensemble_v1.json',
        'queue_ensemble_test_record_v1.json','research_checkpoint_queue_ensemble_v1.json']
    if any((reports/n).exists() for n in names):raise FileExistsError('Never overwrite checkpoint evidence')
    parent_path=reports/'model_artifacts_checkpoint_queue_overlap_v1.json';parent=read(parent_path)
    artifacts=[checked(r) for r in parent['artifacts']];previous_paths={r['path'] for r in artifacts}
    folders=['C49_two_seed_Queue_ensemble_uncertainty','C49_two_seed_Queue_ensemble_uncertainty_v2',
             'C50_probability_API_legacy_replay','C51_CTM_Queue_initial_import_failure',
             'C51_CTM_Queue_latent_observation_diagnostics','C53_two_seed_Queue_ensemble_deployment',
             'C54_Queue_geometry_integrity_rebuild']
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.pkl','.pt','.npz','.zip','.csv'}:
                artifacts.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),
                                  'phase':'queue_ensemble_and_known_latent_diagnostics'})
    geometry=read(reports/'C54_Queue_geometry_integrity_rebuild.json')
    if len(geometry['rows'])!=6:raise ValueError('Incomplete public graph reconstruction')
    for row in geometry['rows']:
        if row['path'] in previous_paths or not row['all_arrays_exact_from_public_network']:
            raise ValueError('Legacy missing-geometry claim inconsistent')
        checked(row);artifacts.append({k:row[k] for k in ['path','bytes','sha256']}|
            {'phase':'legacy_geometry_first_byte_registration_after_public_array_rebuild'})
    if len({r['path'] for r in artifacts})!=len(artifacts):raise ValueError('Duplicate artifacts')
    preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {name:sha256_file(name) for name in parent['dependency_locks']}!=parent['dependency_locks']:
        raise ValueError('Dependency locks changed')
    producer=Path('artifacts/research/C49_two_seed_Queue_ensemble_uncertainty_v2/experiment.json')
    run=read(producer);replay=read(reports/'C53_two_seed_Queue_ensemble_deployment.json')
    if replay['source_experiment_sha256']!=sha256_file(producer):raise ValueError('Ensemble replay source changed')
    if len(run['studies'])!=2 or len(replay['studies'])!=2:raise ValueError('Incomplete seasonal studies')
    exported=0
    for study in run['studies']:
        if len(study['calibration_candidates'])!=21 or study['cal_sensitivity']['selected']!='tree':
            raise ValueError('Candidate/policy record inconsistent')
        token=study['source_study'].split('_',1)[0]
        if sha256_file(producer.parent/(token+'_selection_sealed.json'))!=study['selection_seal_sha256']:
            raise ValueError('Calibration seal changed')
        for row in study['exported_models'].values():
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Exported role changed')
            exported+=1
    ensemble_calls=sum(sum(r['model_window_calls'].values()) for r in replay['studies'])
    if ensemble_calls!=550 or exported!=10 or not all(all(r[k] for k in [
        'all_ensemble_scores_exact','all_full_template_decisions_exact','all_fresh_legal_inputs_exact']) for r in replay['studies']):
        raise ValueError('Incomplete ensemble deployment')
    legacy=read(reports/'C50_probability_API_legacy_replay.json')
    legacy_calls=sum(sum(r['model_window_calls'].values()) for r in legacy['studies'])
    controls=[r['original_seed_training_control'] for r in legacy['studies'] if r['original_seed_training_control']]
    if legacy_calls!=660 or len(controls)!=2 or not all(r['all_tensors_exact'] for r in controls):
        raise ValueError('Legacy API control failed')
    if not all(all(r[k] for k in ['all_raw_neural_probabilities_exact','all_portable_full_template_decisions_exact',
        'fresh_legal_history_inputs_exact']) for r in legacy['studies']):raise ValueError('Legacy output changed')
    sim_path=reports/'C51_CTM_Queue_latent_observation_diagnostics.json';sim=read(sim_path)
    summary=read(reports/'C52_CTM_Queue_observation_summary.json')
    if summary['source_report_sha256']!=sha256_file(sim_path) or not summary['all_18_latent_quality_counterfactuals_exact']:
        raise ValueError('Known-latent summary contract failed')
    if len(sim['rows'])!=144 or sim['registered']['protocol']['final_seeds_generated']:
        raise ValueError('Simulation scope differs')
    # Static AST extraction avoids importing the old GPU program in a CPU process.
    old_source=subprocess.check_output(['git','show','b5e5704:scripts/ctm_adaptive_stress.py'],text=True)
    assignment=next(n for n in ast.parse(old_source).body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets))
    old_cases={ast.literal_eval(key):{kw.arg:ast.literal_eval(kw.value) for kw in value.keywords}
               for key,value in zip(assignment.value.keys,assignment.value.values)}
    if CASES!=old_cases:raise ValueError('Shared CTM case configuration changed')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_legacy_geometry_files_first_byte_registered':6,
        'n_artifacts':len(artifacts),'artifacts':artifacts,'preserved_complete_submissions':preserved,
        'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,
        'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'purpose':'Source/model/cache/unchanged full candidates and legacy graph repair; hashes only',
        'limitations':'Graph current bytes newly registered,not retrospective integrity. No model promotion/new training/formal score.'}
    tests={'created_UTC':now,'research_pytest':{'passed':78,'seconds':14.35,'session_id':51011,
            'basetemp':'E:/ieee/.cache/pytest_queue_ensemble_phase_research'},
        'CPU_pytest':{'passed':66,'skipped_optional_torch_modules':4,'seconds':8.49,
            'basetemp':'E:/ieee/.cache/pytest_queue_ensemble_phase_cpu'},'compileall_passed':True,
        'legacy_API_model_window_calls_exact':legacy_calls,'ensemble_score_and_binary_role_window_calls_exact':ensemble_calls,
        'n_exported_roles':exported,'original_BCE_controls_all_tensors_exact':2,
        'public_geometry_array_reconstruction_files':6,'CTM_shared_cases_exact_static_old_source_check':True,
        'replay_report_sha256':{n:sha256_file(reports/n) for n in ['C50_probability_API_legacy_replay.json',
            'C53_two_seed_Queue_ensemble_deployment.json','C54_Queue_geometry_integrity_rebuild.json']},
        'CTM_counterfactual_scenes':72,'independent_simulator_seeds':3,'overlapping_origins_per_noise':503,
        'formal_best_and_K06_rehashed_unchanged':True,'formal_Kaggle_actions_this_phase':False,
        'scope':'Observed tests,API exact replay,public graph rebuild and own-latent diagnostics; not formal-score evidence.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'objective_status':'active; formal 0.95000+ unachieved',
        'previous_goal_turn_classification':'progress: b5e5704,matched overlap losses and TRAIN-window bias audit',
        'current_goal_turn_classification':'progress','formal_best_Total':.82940,
        'platform_policy':'User directs local model research; no publication,push or upload',
        'completed':['Frozen two-seed ensembles/21cal candidates per season; winter cal tiny gain but significant assessment cost',
            'Paired-cal sensitivity heuristic falls to weaker tree; no promotion or independent-confidence claim',
            'C50 legacy660 exact API calls;C53 new550 exact score/binary role calls',
            'Own72 CTM noise counterfactuals,latent/quality exact; only3 independent seeds,overlapping events',
            'Six omitted old graph NPZs publicly rebuilt and first byte-registered; current complete candidates preserved',
            '78 research/66 CPU tests,4 Torch skips; two failed attempts preserved'],
        'terminal_jobs':[{'session_id':sid,'task':task,'exit_code':code} for sid,task,code in [
            (20156,'C49 missing legacy geometry manifest entry',1),(26273,'C49 v2',0),
            (91191,'C50 legacy API replay',0),(36230,'C51 CTM diagnostics',0),
            (51011,'78 research tests',0),(77262,'C53 ensemble deployment',0)]],
        'other_terminal_failures':[{'task':'C51 CPU import before registration','exit_code':1,
            'record':'reports/research/C51_CTM_Queue_initial_import_failure.json'}],
        'active_jobs_at_checkpoint':[],'new_model_promoted':False,'new_training':False,
        'new_formal_score':None,'formal_best_preserved':True,
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'next_research_priorities':['New-direction checkpoint/decision/selection periods without reusing selected weights as independent cal',
            'Keep strong incumbent when uncertain;do not automatically fall to weaker tree',
            'Joint latent/observation noise and future-availability model using lawful history,not future mask',
            'State dynamics and OD count/prior mismatch retain all negative evidence'],
        'limitations':'One direction,old TRAIN assessment dates,shared optimizer-seed events. Simulated labels are not organizer labels. Full formal target remains unachieved.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (reports/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('previous_unchanged',len(parent['artifacts']),'total_artifacts',len(artifacts),'legacy_calls',legacy_calls,'new_role_calls',ensemble_calls,flush=True)


if __name__=='__main__':main()
