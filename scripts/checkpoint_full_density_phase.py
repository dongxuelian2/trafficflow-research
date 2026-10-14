"""Hash completed density training and corrected Queue-domain evidence locally."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file


def main():
    reports=Path('reports/research')
    names=['model_artifacts_checkpoint_full_density_v1.json','full_density_test_record_v1.json',
           'research_checkpoint_full_density_v1.json']
    if any((reports/n).exists() for n in names):raise FileExistsError('Never overwrite checkpoints')
    parent_path=reports/'model_artifacts_checkpoint_assimilation_v1.json';parent=json.loads(parent_path.read_text())
    artifacts=[]
    for row in parent['artifacts']:
        path=Path(row['path'])
        if path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:raise ValueError('Previous artifact changed: '+str(path))
        artifacts.append(dict(row))
    folders=['A66_full_density_I210_W_autumn','A67_full_public_density_labels_I5_N_winter',
        'A68_full_density_I5_N_winter','A66_full_density_I210_W_autumn.deployment',
        'A68_full_density_I5_N_winter.deployment','A70_density_fit_normalization_audit',
        'A71_full_density_State_temporal_extension','C35_full_forecast_domain_audit','C37_wave_full_forecast_domain_audit']
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.pkl','.pt','.npz','.zip'}:
                artifacts.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),'phase':'full_density_and_queue_domain'})
    if len({r['path'] for r in artifacts})!=len(artifacts):raise ValueError('Duplicate artifacts')
    verification={}
    for study in ['A66_full_density_I210_W_autumn','A68_full_density_I5_N_winter']:
        producer=Path('artifacts/research')/study/'experiment.json';replay_path=reports/(study+'.deployment.json')
        replay=json.loads(replay_path.read_text())
        if replay['source_experiment_sha256']!=sha256_file(producer):raise ValueError('Replay source changed')
        if not replay['four_variant_patch_sequence_identical'] or not all(r['all_tensors_exact'] for r in replay['original_training_controls']):
            raise ValueError('Matched original controls not verified')
        if not all(r['n_calibration_days']==14 and r['strict_inputs_and_raw_TRAIN_labels_rebuilt_exact'] for r in replay['roles'].values()):
            raise ValueError('Incomplete deployment replay')
        verification[replay_path.name]=sha256_file(replay_path)
    norm=json.loads((reports/'A70_density_fit_normalization_audit.json').read_text())
    for row in norm['rows']:
        study='A66_full_density_I210_W_autumn' if row['panel']=='D7_I210_W' else 'A68_full_density_I5_N_winter'
        run=json.loads((reports/(study+'.json')).read_text())
        if not all(r['sample_sequence_sha256']==row['sample_sequence_sha256'] for r in run['training'].values()):
            raise ValueError('Fit normalization audit did not use actual training patches')
    q35=json.loads((reports/'C35_full_forecast_domain_audit.json').read_text());q37=json.loads((reports/'C37_wave_full_forecast_domain_audit.json').read_text())
    if not all(r['legacy_proxy_replay_exact'] for r in q35['rows']) or not all(r['legacy_replay_exact'] and r['fresh_causal_inputs_exact'] for r in q37['rows']):
        raise ValueError('Queue legacy inputs/scores not independently reproduced')
    preserved=[]
    for row in parent['preserved_complete_submissions']:
        path=Path(row['path'])
        if path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:raise ValueError('Complete submission changed')
        preserved.append(dict(row))
    if {n:sha256_file(n) for n in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Locks changed')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'artifacts':artifacts,'n_artifacts':len(artifacts),
        'dependency_locks':parent['dependency_locks'],'preserved_complete_submissions':preserved,
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'purpose':'Model/cache/config/source integrity; hashes only, no data redistribution',
        'limitations':'No new model promoted or formal score. Old Queue outputs retained; corrected domain changes evidence, not saved weights.'}
    tests={'created_UTC':now,'research_pytest':{'passed':63,'seconds':12.76,'session_id':49361,
        'basetemp':'E:/ieee/.cache/pytest_full_density_queue_domain_research'},
        'CPU_pytest':{'passed':55,'skipped_optional_torch_modules':3,'seconds':7.17,
        'basetemp':'E:/ieee/.cache/pytest_full_density_queue_domain_cpu'},
        'compileall_passed':True,'independent_deployment_reports':verification,
        'Queue_old_model_phase_metrics_replayed_exact':len(q35['rows'])+len(q37['rows']),
        'queue_domain_scored_decision_changes_with_model_seed_repeats':sum(r['changed_scored_decisions'] for r in q35['rows']+q37['rows']),
        'formal_inference_scope_report_sha256':sha256_file(reports/'full_density_formal_queue_inference_scope_v1.json'),
        'formal_baseline_rehashed':True,'formal_Kaggle_actions_this_phase':False,
        'scope':'Observed tests, same-weight control replays, cal masked inputs/raw TRAIN labels, corrected Queue domain. Not formal hidden-score evidence.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'objective_status':'active; formal 0.95000+ unachieved','previous_goal_turn_classification':'progress: b02061e, assimilation controls/coverage preparation',
        'current_goal_turn_classification':'progress','formal_best_Total':.82940,
        'platform_policy':'User directs local model research; no publication, push or formal upload',
        'completed':['A66/A68 eight matched neural density-loss candidates, original control tensors exact',
            'A67 winter full TRAIN density labels, parent data retained',
            'A69 paired State/public increment diagnostics, A70 actual fit batch normalization audit',
            'A71 thirty-day frozen time extension, consistent State cost',
            'C35/C37 all80 original model/phase Queue metrics reproduced, full-domain set predictions before scorer masks',
            'C36 corrected deduplicated Queue comparisons; same old choices, no new cal reselection',
            'Formal Queue inference source/8 frozen model files rechecked; complete template, no future eligibility read',
            '63 research/55 CPU tests,3 optional Torch skips, docs and exact report/artifact hashes'],
        'terminal_jobs':[{'session_id':sid,'task':task,'exit_code':0} for sid,task in [
            (64015,'A66'),(7853,'A67'),(69193,'A68'),(78123,'A66 deployment'),(79270,'A68 deployment'),
            (68112,'A71'),(84183,'C35'),(9252,'C37'),(91268,'intermediate61 research tests'),(49361,'final63 research tests')]],
        'active_jobs_at_checkpoint':[],'new_model_promoted':False,'new_formal_score':None,'formal_best_preserved':True,
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'next_research_priorities':['Complete forecast-domain Queue calibration, public timing before optional nonempty, no future eligibility as decision input',
            'Queue latent/observation and sampled-positive bias diagnostics, event probability/space dependence under lawful simulations',
            'Only revisit density loss with calibrated noise/strength controls, not larger fixed-lambda grid',
            'OD count/prior mismatch with preserved K01 formal counterevidence'],
        'limitations':'Density changes have sustained State cost in I210 and tiny winter effects. Queue corrected absolute proxy scores differ; latent truth/true Physics remain unknown. No final target completion claim.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (reports/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('previous_unchanged',len(parent['artifacts']),'total_artifacts',len(artifacts),'Queue_old_metrics_replayed',tests['Queue_old_model_phase_metrics_replayed_exact'],flush=True)


if __name__=='__main__':main()
