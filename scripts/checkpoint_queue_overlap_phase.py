"""Hash completed matched overlap controls, including unsuccessful verification."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import checked, read


def main():
    reports=Path('reports/research')
    names=['model_artifacts_checkpoint_queue_overlap_v1.json','queue_overlap_test_record_v1.json',
           'research_checkpoint_queue_overlap_v1.json']
    if any((reports/name).exists() for name in names):raise FileExistsError('Never overwrite checkpoint evidence')
    parent_path=reports/'model_artifacts_checkpoint_full_queue_v1.json';parent=read(parent_path)
    artifacts=[checked(row) for row in parent['artifacts']]
    studies=['C42_I5_N_autumn_window_overlap_seed2','C43_I5_N_winter_window_overlap_seed2',
             'C47_I5_N_autumn_window_overlap_seed1','C48_I5_N_winter_window_overlap_seed1']
    extra=['C44_released_TRAIN_Queue_window_audit','C45_TRAIN_Queue_quality_projection',
           'C46_fixed_alpha_overlap_comparison','C46_overlap_deployment_first_three',
           'C46_overlap_deployment_first_three_v2','C48_overlap_deployment']
    for name in studies+extra:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.npz','.pt','.pkl','.zip','.csv'}:
                artifacts.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),
                                  'phase':'queue_overlap_and_observation_audit'})
    if len({row['path'] for row in artifacts})!=len(artifacts):raise ValueError('Duplicate artifacts')
    preserved=[checked(row) for row in parent['preserved_complete_submissions']]
    if {name:sha256_file(name) for name in parent['dependency_locks']}!=parent['dependency_locks']:
        raise ValueError('Dependency locks changed')
    runs={name:read(reports/(name+'.json')) for name in studies}
    for name,run in runs.items():
        if not run['matched_initialization_and_batches'] or len(run['training'])!=3:raise ValueError('Incomplete training arms')
        if len({r['initial_tensor_sha256'] for r in run['training'].values()})!=1 or len({
            r['batch_sequence_sha256'] for r in run['training'].values()})!=1:raise ValueError('Unmatched training controls')
        folder=Path('artifacts/research')/name
        if sha256_file(folder/'selection_sealed.json')!=run['selection_seal_sha256']:raise ValueError('Selection seal changed')
        for kind,model in run['exported_models'].items():
            if sha256_file(model['path'])!=model['sha256'] or sha256_file(folder/(kind+'.pt'))!=run['training'][kind]['checkpoint_sha256']:
                raise ValueError('Selected model changed')
        if run['formal_Queue'] is not None or run['formal_Total'] is not None or run['uploaded']:
            raise ValueError('Unexpected formal action/claim')
    replay_paths=[reports/'C46_overlap_deployment_first_three_v2.json',reports/'C48_overlap_deployment.json']
    replay_rows=[row for path in replay_paths for row in read(path)['studies']]
    if len(replay_rows)!=4 or {Path(r['source_report']).stem for r in replay_rows}!=set(studies):
        raise ValueError('Incomplete independent replay')
    calls=0;controls=0
    for row in replay_rows:
        if sha256_file(row['source_report'])!=row['source_report_sha256']:raise ValueError('Replayed report changed')
        if not all(row[key] for key in ['all_raw_neural_probabilities_exact','all_portable_full_template_decisions_exact','fresh_legal_history_inputs_exact']):
            raise ValueError('Independent inference failed')
        calls+=sum(row['model_window_calls'].values())
        if row['original_seed_training_control'] is not None:
            if not row['original_seed_training_control']['all_tensors_exact']:raise ValueError('Original control differs')
            controls+=1
    if calls!=660 or controls!=2:raise ValueError('Incomplete call/control counts')
    audit=read(reports/'C44_released_TRAIN_Queue_window_audit.json')
    projection=read(reports/'C45_TRAIN_Queue_quality_projection.json')
    if len(audit['rows'])!=80 or sum(not r['future_observation_positive'] for r in audit['rows'])!=1:
        raise ValueError('Window audit evidence differs')
    if not all(r['index_future_coverage']==1. for r in audit['rows']):raise ValueError('Coverage metadata claim differs')
    if projection['summary']['all_finite_positives_discarded_windows']!=0:raise ValueError('Quality explanation differs')
    comparison=read(reports/'C46_fixed_alpha_overlap_comparison.json')
    if len(comparison['studies'])!=4:raise ValueError('Incomplete fixed-alpha comparisons')
    for path,digest in comparison['prediction_archives_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Compared probability archive changed')
    failed=read(reports/'C46_overlap_deployment_first_three.failed.json')
    if failed['exit_code']!=1 or failed['scores_generated']:raise ValueError('Failure record changed')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'dependency_locks':parent['dependency_locks'],'preserved_complete_submissions':preserved,
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'purpose':'Frozen model/cache/source, failed replay and unchanged complete candidates; hashes only',
        'limitations':'Two optimizer seeds share old TRAIN event data. No new model promoted or formal score.'}
    tests={'created_UTC':now,'research_pytest':{'passed':71,'seconds':14.62,'session_id':47481,
            'basetemp':'E:/ieee/.cache/pytest_queue_overlap_phase_research'},
        'CPU_pytest':{'passed':59,'skipped_optional_torch_modules':4,'seconds':8.33,
            'basetemp':'E:/ieee/.cache/pytest_queue_overlap_phase_cpu'},'compileall_passed':True,
        'lovasz_discrete_truth_error_pairs_independently_checked':64,
        'raw_neural_cal_window_calls_exact':calls,'portable_full_template_cal_window_calls_exact':calls,
        'original_BCE_training_controls_all_tensors_exact':controls,
        'replay_report_sha256':{path.name:sha256_file(path) for path in replay_paths},
        'first_failed_replay_record_sha256':sha256_file(reports/'C46_overlap_deployment_first_three.failed.json'),
        'formal_best_and_K06_rehashed_unchanged':True,'formal_Kaggle_actions_this_phase':False,
        'scope':'Observed tests and independent causal inference/control replays, not formal performance evidence.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'objective_status':'active; formal 0.95000+ unachieved',
        'previous_goal_turn_classification':'progress: 5ce3efb, full-domain Queue calibration and K06 controlled candidate',
        'current_goal_turn_classification':'progress','formal_best_Total':.82940,
        'platform_policy':'User directs local model research; no publication, push or formal upload',
        'completed':['Twelve matched BCE/soft-IoU/Lovasz models, two seasons/two optimizer seeds; no stable primary gain',
            'All choices sealed before assessment cache reads;16 fixed-alpha secondary comparisons retained without promotion',
            '80 released TRAIN windows audited;one onset with no threshold observation positives, not explained by eligibility',
            '660 raw probability/portable forecast calls exact;two original BCE controls all tensors exact',
            '71 research/59 CPU tests,4 optional Torch skips; unchanged formal baseline/K06 and exact artifacts'],
        'terminal_jobs':[{'session_id':sid,'task':task,'exit_code':code} for sid,task,code in [
            (42361,'C42',0),(75598,'C43',0),(40470,'C47',0),(50979,'C48',0),
            (7692,'first deployment numpy.str_ failure',1),(83469,'first3 deployment v2',0),
            (5773,'C48 deployment',0),(75296,'fixed-alpha comparison',0),(47481,'71 research tests',0)]],
        'active_jobs_at_checkpoint':[],'new_model_promoted':False,'new_formal_score':None,'formal_best_preserved':True,
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'next_research_priorities':['Cross-seed probability ensembles and calibrated selection uncertainty before new architecture grids',
            'Known-latent CTM event/observation noise and future availability diagnostics, future mask never as input',
            'Cross-direction Queue paired generalization after stable cal policy',
            'State latent/noisy dynamics and OD prior/count mismatch retain all negative evidence'],
        'limitations':'Fixed overlap strength/old Brier checkpoint controls only; secondary positive effects do not generalize. One direction, old observed-proxy development periods. True hidden task scores unknown.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (reports/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('previous_unchanged',len(parent['artifacts']),'total_artifacts',len(artifacts),'raw_and_portable_calls',calls,'original_controls',controls,flush=True)


if __name__=='__main__':main()
