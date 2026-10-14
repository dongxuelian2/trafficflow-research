"""Preserve completed assimilation studies, failures and next loss label cache."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from compare_state_assimilation import PAIRS


def main():
    reports=Path('reports/research')
    output_names=['model_artifacts_checkpoint_assimilation_v1.json',
                  'assimilation_test_record_v1.json','research_checkpoint_assimilation_v1.json']
    if any((reports/n).exists() for n in output_names):
        raise FileExistsError('Never overwrite phase checkpoints')
    parent_path=reports/'model_artifacts_checkpoint_quality_density_v1.json'
    parent=json.loads(parent_path.read_text())
    artifacts=[]
    for row in parent['artifacts']:
        path=Path(row['path'])
        if path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:
            raise ValueError('Previous artifact changed: '+str(path))
        artifacts.append(dict(row))
    studies=[name for pair in PAIRS for name in pair]
    verification={}
    solver_calls=0
    for name in studies:
        producer_path=Path('artifacts/research')/name/'experiment.json'
        run=json.loads(producer_path.read_text())
        verification_path=reports/(name+'.deployment.json')
        replay=json.loads(verification_path.read_text())
        if replay['source_experiment_sha256']!=sha256_file(producer_path):
            raise ValueError('Independent replay source changed')
        for role,row in replay['roles'].items():
            if row['n_calibration_days']!=14 or not row['visible_inputs_independently_rebuilt'] or row['max_target_prediction_abs_drift']>1e-7:
                raise ValueError('Independent full-cal replay incomplete')
            if sha256_file(Path('artifacts/research')/name/('State_'+role+'.pkl'))!=row['model_sha256']:
                raise ValueError('Saved model changed')
        solver_calls+=2*len(run['solver_records'])
        verification[verification_path.name]=sha256_file(verification_path)
    if solver_calls!=3836:
        raise ValueError('Expected completed solver evidence changed')
    folders=['A58_residual_assimilation_I210_W_autumn',*studies,
             *[Path(n).stem for n in verification], 'A65_full_public_density_labels_I210_W_autumn',
             'A65_full_loss_test_boundary_failure']
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.pkl','.pt','.npz','.zip'}:
                artifacts.append({'path':path.as_posix(),'bytes':path.stat().st_size,
                                  'sha256':sha256_file(path),'phase':'assimilation'})
    if len({r['path'] for r in artifacts})!=len(artifacts):
        raise ValueError('Duplicate artifact records')
    preserved=[]
    for row in parent['preserved_complete_submissions']:
        path=Path(row['path'])
        if sha256_file(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('Complete frozen submission changed')
        preserved.append(dict(row))
    if {n:sha256_file(n) for n in parent['dependency_locks']}!=parent['dependency_locks']:
        raise ValueError('Dependency locks changed')
    comparison=json.loads((reports/'A64_assimilation_matched_comparison.json').read_text())
    if any(abs(r['pooled_State_gain'])>=.002 for r in comparison['rows']):
        raise ValueError('Manual promotion policy audit needed')
    labels=json.loads((reports/'A65_full_public_density_labels_I210_W_autumn.json').read_text())
    if any(r['non_target_eligible_input_blanks']!=0 for r in labels['coverage_summary'].values()):
        raise ValueError('Expected coverage-information audit changed')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),
        'artifacts':artifacts,'n_artifacts':len(artifacts),
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'dependency_locks':parent['dependency_locks'],'preserved_complete_submissions':preserved,
        'purpose':'Exact model/cache/config/source integrity; hashes only, no data redistribution',
        'limitations':'New models not promoted. A65 is a label cache, no trained-model gain. Final simulator seeds unopened.'}
    tests={'created_UTC':now,
        'research_pytest':{'passed':59,'seconds':15.10,'session_id':86805,
                          'basetemp':'E:/ieee/.cache/pytest_assimilation_full_loss_repaired_research'},
        'intermediate_research_pytest':{'passed':58,'seconds':15.60,'session_id':80121},
        'retained_test_edit_failure_sha256':sha256_file(reports/'assimilation_full_loss_test_repair_v1.json'),
        'CPU_pytest':{'passed':51,'skipped_optional_torch_modules':3,'seconds':9.52,'session_id':14092,
                     'basetemp':'E:/ieee/.cache/pytest_assimilation_phase_cpu'},
        'compileall':'src/scripts/tests passed after all source additions',
        'independent_deployment_reports':verification,'experiment_channel_solves':solver_calls,
        'independent_sparse_direct_scope':'Real 24-slot subset per study; full-grid CG certificates retained separately',
        'formal_baseline_rehashed':True,'formal_Kaggle_actions_this_phase':False,
        'scope':'Observed tests, complete cal saved-model replays and solver references; not hidden score evidence'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'objective_status':'active; formal 0.95000+ unachieved',
        'previous_goal_turn_classification':'progress: A51-A57 completed, independent checks and local commit 8eacb44',
        'current_goal_turn_classification':'progress',
        'formal_best_Total':.82940,'platform_policy':'Latest user direction: local models only, no publish/push/upload',
        'completed':['A58 failed startup preserved and repaired study completed',
            'Six ordinary/counterfactual assimilation models across three panel/fold cases; all choices cal sealed',
            'A64 matching of original inputs and solver grid, all comparisons and negative results retained',
            'Six studies all14-day independent cal replay and real-subset sparse direct references',
            'A65 full TRAIN label/Mode A pair cache; zero extra buffer anchors contradict initial local hypothesis',
            'Full-coverage public density loss kernel with independent hand-calculated value/gradient test; not trained',
            '59 research/51 CPU tests, documentation and exact artifact/report hashes'],
        'terminal_jobs':[{'session_id':sid,'task':task,'exit_code':code} for sid,task,code in [
            (35981,'A58 initial startup failure',1),(82031,'A58 repaired',0),(54064,'A59',0),
            (98505,'A60',0),(80287,'A61',0),(95529,'A62',0),(32986,'A63',0),
            (54890,'A59 replay',0),(12355,'A58 replay',0),(65891,'A60 replay',0),
            (43603,'A61 replay',0),(79440,'A62 replay',0),(82754,'A63 replay',0),
            (29638,'A65 label preparation',0),(80121,'intermediate research pytest',0),(14092,'CPU pytest',0),
            (12522,'Full loss test boundary editing failure',1),(86805,'final research pytest',0)]],
        'active_jobs_at_checkpoint':[],'new_State_model_promoted':False,'new_formal_score':None,
        'formal_best_preserved':True,'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'next_research_priorities':['Matched strict12/vq density loss with public Mode A pair coverage and full denominator; no input label changes',
            'Queue observation/latent label-noise and event sampling consistency, causal rolling cross-panel validation',
            'Robust OD count inconsistency/prior mismatch; retain formal K01 counterevidence'],
        'limitations':'Current models gain <=.000124. Cal/assessment are previously inspected public train. No formal Physics or total gain proved; no goal completion claim.'}
    for name,value in zip(output_names,[manifest,tests,checkpoint]):
        with (reports/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,allow_nan=False)
    print('previous_unchanged',len(parent['artifacts']),'total_artifacts',len(artifacts),'experiment_channel_solves',solver_calls,flush=True)


if __name__=='__main__':main()
