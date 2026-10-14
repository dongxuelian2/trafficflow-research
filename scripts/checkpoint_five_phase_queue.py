"""Hash completed five-phase studies,sampler failures and preserved baselines."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import checked,read
from trafficflow.queue_phase_split import PHASES


def main():
    reports=Path('reports/research');names=['model_artifacts_checkpoint_five_phase_v1.json',
        'five_phase_test_record_v1.json','research_checkpoint_five_phase_v1.json']
    if any((reports/n).exists() for n in names):raise FileExistsError('Never overwrite five-phase checkpoints')
    parent_path=reports/'model_artifacts_checkpoint_queue_ensemble_v1.json';parent=read(parent_path)
    artifacts=[checked(r) for r in parent['artifacts']]
    registration=reports/'C55_C58_five_phase_campaign_balanced_v2.registered.json';campaign=read(registration)
    balanced=[j['name'] for j in campaign['protocol']['jobs']]
    completed=['C55_I5_S_autumn_five_phase']+balanced
    folders=completed+[n+'.deployment' for n in completed]+['C56_I5_S_winter_five_phase',
        'C59_legacy_sampler_metadata_replay','C60_I5_S_autumn_same_population_sampler_control']
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.pkl','.pt','.npz','.parquet','.zip','.csv'}:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'five_phase_Queue_and_condition_sampler'})
    if len({r['path'] for r in artifacts})!=len(artifacts):raise ValueError('Duplicate artifact paths')
    preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    calls=0;nn=0;trees=0;replays={}
    for name in completed:
        run=read(reports/(name+'.json'));folder=Path('artifacts/research')/name
        replay_path=reports/(name+'.deployment.json');replay=read(replay_path)
        if replay['source_experiment_sha256']!=sha256_file(folder/'experiment.json'):raise ValueError('Replay source changed')
        if not all(replay[k] for k in ['all_fresh_legal_inputs_exact','all_scores_exact','all_full_template_decisions_exact',
            'all_phase_reads_after_exact_prior_seal','all_phase_dates_disjoint_and_complete']):raise ValueError('Incomplete independent replay')
        calls+=sum(replay['model_window_calls'].values());replays[replay_path.name]=sha256_file(replay_path)
        flat=[d for phase in PHASES for d in run['dates'][phase]]
        if flat!=sorted(set(flat)):raise ValueError('Mechanism phases overlap')
        for seal,digest in run['seal_sha256'].items():
            if sha256_file(folder/(seal+'_sealed.json'))!=digest:raise ValueError('Seal changed')
        for row in run['cache_records']:
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Sequence cache changed')
            for f in row['feature_files']:
                if sha256_file(f['path'])!=f['sha256']:raise ValueError('Feature cache changed')
        for seed,row in run['training'].items():
            if sha256_file(folder/('seed_'+seed+'.pt'))!=row['selected_checkpoint_sha256']:raise ValueError('Neural checkpoint changed')
            nn+=1
        for row in run['exported_models'].values():
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Exported role changed')
        trees+=2
    if calls!=608 or nn!=15 or trees!=10:raise ValueError('Model/replay counts differ')
    summary=read(reports/'C61_balanced_five_phase_campaign_comparison.json')
    if summary['campaign_registered_sha256']!=sha256_file(registration) or summary['local_proxy_promotion_evidence_gate_passed']:
        raise ValueError('Unexpected campaign promotion/source claim')
    if not read(reports/'C55_C58_five_phase_campaign_balanced_v2.status.json')['complete']:
        raise ValueError('Balanced campaign is not terminal successful')
    old_status=read(reports/'C55_C58_five_phase_campaign.status.json')
    if old_status['complete'] or old_status['terminal_children'][-1]['exit_code']!=1:raise ValueError('Original failure not preserved')
    compatibility=read(reports/'C59_legacy_sampler_metadata_replay.json')
    if len(compatibility['rows'])!=8 or not all(r['original_metadata_exact'] and r['original_eligible_labels_exact'] for r in compatibility['rows']):
        raise ValueError('Default sampler compatibility failed')
    control=read(reports/'C60_I5_S_autumn_same_population_sampler_control.json')
    if not control['balanced_model_predictions_reproduced_exact']:raise ValueError('Same-population model control failed')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'purpose':'Model/source/config,complete causal caches including Parquet features,failed study and unchanged complete candidates; hashes only',
        'limitations':'No formal gain or promotion. Public noisy proxy,overlapping development calendars and shared event data.'}
    tests={'created_UTC':now,'research_pytest':{'passed':83,'seconds':16.32,'session_id':42194,
        'basetemp':'E:/ieee/.cache/pytest_balanced_five_phase_research'},
        'CPU_pytest':{'passed':71,'skipped_optional_torch_modules':4,'seconds':8.97,'session_id':43986,
        'basetemp':'E:/ieee/.cache/pytest_balanced_five_phase_cpu'},'compileall_passed':True,
        'completed_neural_training_models':nn,'completed_tree_training_models':trees,'failed_study_fit_only_trees_preserved':2,
        'decision_role_window_inputs_score_binary_exact':calls,'deployment_report_sha256':replays,
        'legacy_sampler_metadata_label_replays_exact':8,'same_population_sampler_policy_control_report_sha256':sha256_file(
            reports/'C60_I5_S_autumn_same_population_sampler_control.json'),
        'formal_best_and_K06_rehashed_unchanged':True,'formal_Kaggle_actions_this_phase':False,
        'scope':'Five disjoint mechanisms,default sampler backward compatibility and full-domain causal deployment; not official-score evidence.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'objective_status':'active; formal 0.95000+ unachieved',
        'previous_goal_turn_classification':'progress: 79d66f4,ensemble sensitivity and known-latent diagnostics',
        'current_goal_turn_classification':'progress','formal_best_Total':.82940,
        'platform_policy':'User directs local models; no publication,push or upload',
        'completed':['Five-phase weight/decision/model selection with independent dates and exact seals',
            'Original winter failed on zero selected onset despite15candidates,source/fitted trees retained',
            'New condition-balanced proxy four studies,two directions/two fits/three seeds; no model promotion',
            'All decisions retain incumbent; no nonzero challenger,so gate efficacy not established',
            '608 exact decision role calls,8 exact default-sampler replays,83 research/71CPU tests',
            'Same-population old/new frozen policy control shows no gain; raw unpaired scores not compared'],
        'terminal_jobs':[{'session_id':sid,'task':task,'exit_code':code} for sid,task,code in [
            (37159,'original campaign stopped after C56 failure',1),(78447,'four balanced-v2 studies',0),
            (90190,'original C55 deployment',0),(10133,'eight legacy sampler replays',0),
            (47349,'balanced C55 deployment',0),(83217,'same-population sampler control',0),
            (67431,'balanced C56 deployment',0),(63553,'balanced C57 deployment',0),
            (24352,'balanced C58 deployment',0),(42194,'83 research tests',0),(43986,'71CPU tests',0)]],
        'active_jobs_at_checkpoint':[],'new_model_promoted':False,'new_formal_score':None,'formal_best_preserved':True,
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'next_research_priorities':['Event location/latent-observation information rather than more current graph/alpha grids',
            'Keep same-population comparisons and evaluate a genuine nonzero challenger before gate efficacy claims',
            'Return substantial effort to State latent dynamics and probabilistic OD prior/demand mismatch',
            'Preserve frozen formal baseline and all negative evidence; formal performance remains unknown'],
        'limitations':'Scored-positive proxy bias persists. New mechanisms do not make previously viewed public calendars final/private tests.'}
    for name,value in zip(names,[manifest,tests,checkpoint]):
        with (reports/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    print('previous_unchanged',len(parent['artifacts']),'total_artifacts',len(artifacts),'exact_role_calls',calls,'NN/tree',nn,trees,flush=True)


if __name__=='__main__':main()
