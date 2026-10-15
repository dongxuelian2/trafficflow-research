"""Freeze positive OD demand research,negative evidence and local control."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import checked,read


def main():
    reports=Path('reports/research')
    names=['model_artifacts_checkpoint_log_demand_v1.json','log_demand_test_record_v1.json',
        'research_checkpoint_log_demand_v1.json','K07_log_demand_OD_LOCAL.controlled.json']
    if any((reports/n).exists() for n in names):raise FileExistsError('Never overwrite OD checkpoints')
    parent_path=reports/'model_artifacts_checkpoint_five_phase_v1.json';parent=read(parent_path)
    artifacts=[checked(row) for row in parent['artifacts']]
    folders=['D14_log_demand_OD_broad_control','D15_log_demand_OD_numeric_rank_verification',
        'D16_selected_log_demand_numeric_verification','D17_log_demand_OD_numeric_polish',
        'D18_polished_log_demand_numeric_verification','K07_log_demand_OD_LOCAL']
    for name in folders:
        folder=Path('artifacts/research')/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.pkl','.pt','.npz','.parquet','.zip','.csv'}:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),
                    'phase':'positive_OD_demand_and_numerical_repair'})
    if len({r['path'] for r in artifacts})!=len(artifacts):raise ValueError('Duplicate artifact paths')
    preserved=[checked(row) for row in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:
        raise ValueError('Dependency locks changed')
    candidate=read(reports/'K07_log_demand_OD_LOCAL.json')
    diff=read('artifacts/research/K07_log_demand_OD_LOCAL/controlled_diff.json')
    if not diff['full_row_diff_passed'] or diff['rows']!=6980503 or diff['changed_rows']!={'odme':70708}:
        raise ValueError('OD-only control not complete/exact')
    path=Path('artifacts/research/K07_log_demand_OD_LOCAL/submission.csv')
    if candidate['submission_sha256']!=sha256_file(path) or diff['candidate_sha256']!=sha256_file(path):
        raise ValueError('Candidate checksum differs')
    preserved.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),
        'scope':'OD-only LOCAL K07,formal score unknown,unuploaded'})
    before=read(reports/(folders[0]+'.json'));after=read(reports/(folders[3]+'.json'))
    repair=read(reports/'D19_log_demand_numeric_repair_comparison.json')
    comparison=read(reports/'D20_log_demand_matched_model_comparison.json')
    numeric=read(reports/(folders[4]+'.json'))
    if before['optimizer_case_failures']!=7 or after['optimizer_case_failures']!=0:
        raise ValueError('Numerical failures not retained/repaired as measured')
    if len(after['development'])!=1620 or len(after['assessment'])!=1080 or not after['selection']['original_choice_retained']:
        raise ValueError('Incomplete grid/cases or changed choice')
    if repair['n_recovered_recorded_cases']!=7 or not repair['all_reference_predictions_and_scores_exact']:
        raise ValueError('Incomplete numerical regression')
    selected=after['selection']['best_hypothesis']['index']
    rows=[r for r in numeric['rows'] if r['setting_index']==selected]
    if len(rows)!=6 or not all(r['independent_solutions_agree'] for r in rows):raise ValueError('Selected numeric check failed')
    for name in [folders[0],folders[3]]:
        run=read(reports/(name+'.json'));folder=Path('artifacts/research')/name
        if sha256_file(folder/'selection_sealed.json')!=run['selection_seal_sha256']:raise ValueError('Selection seal changed')
        if (folder/'experiment.json').read_bytes()!=(reports/(name+'.json')).read_bytes():raise ValueError('Experiment mirror differs')
    now=datetime.now(timezone.utc).isoformat()
    manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'purpose':'Full source,simulated cases,failures,independent local solutions and complete OD-only control; hashes only',
        'limitations':'No new formal score. Synthetic demand laws and local stationarity cannot prove actual path recovery.'}
    tests={'created_UTC':now,'research_pytest':{'passed':88,'seconds':20.78,'session_id':67641,
        'basetemp':'E:/ieee/.cache/pytest_log_demand_polish_research'},
        'CPU_pytest':{'passed':76,'skipped_optional_torch_modules':4,'seconds':9.58,'session_id':49881,
        'basetemp':'E:/ieee/.cache/pytest_log_demand_polish_cpu'},'compileall_passed':True,
        'selected_parameter_independent_local_oracles_passed':6,'all_polished_parameter_oracles_agree':16,
        'all_polished_parameter_oracles_tested':18,'unchanged_numerical_KKT_tolerance':2e-4,
        'full_row_control_diff':diff,'new_formal_actions':False,
        'scope':'Gradient,Gaussian/Huber trust-region oracle,gauge direction,invalid inputs,existing full suite and full submission control'}
    terminal=[(88323,'D14 original broad study,case failures retained'),(60179,'D15 independent local oracle/rank'),
        (95163,'D16 selected setting oracle'),(89151,'D17 same-case numerical repair'),
        (22621,'D18 polished independent local oracle/rank'),(67641,'88 research tests'),(49881,'76 CPU tests'),
        (55601,'K07 full local control build'),(54565,'K07 full-row OD-only baseline comparison')]
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'objective_status':'active; formal 0.95000+ unachieved','current_goal_turn_classification':'progress',
        'previous_goal_turn_classification':'progress: 2b1455e,disjoint-phase Queue studies and sampler repair',
        'formal_best_Total':.82940,'formal_best_preserved':True,'new_formal_score':None,
        'platform_policy':'User directs local modeling;no publication,push or upload',
        'completed':['Independent multiplicative positive origin/destination log-demand MAP,Gaussian/Huber count losses',
            'Nine known-truth laws,10 public networks,three development seeds,four settings and sealed choice',
            'Fresh research seed initial assessment exposed3selected-case numeric failures; full failure evidence retained',
            'Same-grid/seeds/objective/tolerance numerical polish replays all2700records and original choice',
            'Independent finite-difference trust-region oracle,local group rank,gauge and flat-solution diagnostics',
            'Complete K07 OD-only LOCAL control,full row baseline diff and dependency/baseline rehash'],
        'terminal_jobs':[{'session_id':sid,'task':task,'exit_code':0} for sid,task in terminal],
        'active_jobs_at_checkpoint':[],'new_formal_model_promoted':False,'local_control_candidate':'K07_log_demand_OD_LOCAL',
        'simulated_matched_comparison':comparison['paired_comparisons'],
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},
        'next_research_priorities':['Challenge fixed multiplicative prior with independently registered demand/noise laws',
            'Separate data-informed contrasts from prior-selected null directions;evaluate genuine uncertainty calibration',
            'Allow sparse/path residuals only if they improve the same strong entropy reference across families',
            'Return to State latent dynamics;new full-domain Queue graph grids have no reliable promotion evidence'],
        'limitations':'Repair evaluation reuses previously viewed research seeds;not a new blind test. '
        'Positive factor model cannot recover exact sparse zeros;local stationarity is not global optimality or identifiability.'}
    for name,value in zip(names,[manifest,tests,checkpoint,diff]):
        with (reports/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('previous_unchanged',len(parent['artifacts']),'total_artifacts',len(artifacts),'K07_SHA256',candidate['submission_sha256'],flush=True)


if __name__=='__main__':main()
