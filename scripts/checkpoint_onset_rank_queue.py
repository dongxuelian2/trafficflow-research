"""Freeze public onset-ranking experiments, negative results and diagnostics."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from lightgbm import Booster
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_onset_rank_v1.json','onset_rank_verification_record_v1.json','research_checkpoint_onset_rank_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite onset-rank checkpoint')
    parent_path=root/'model_artifacts_checkpoint_queue_latent_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency lock changed')
    stems=['C65_onset_query_rank_public','C66_onset_query_rank_public_verification','C67_onset_query_rank_cardinality_diagnostic'];study,audit,diagnostic=[read(root/(s+'.json')) for s in stems]
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or diagnostic['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or diagnostic['independent_audit_sha256']!=sha256_file(root/(stems[1]+'.json')):raise ValueError('Source audit changed')
    if len(study['cases'])!=4 or len(audit['cases'])!=4 or any(not r['all_query_rows_and_groups_exact'] or not r['all_decision_grids_and_gates_exact'] or not r['all_ongoing_decisions_unchanged'] for r in audit['cases']):raise ValueError('Case verification incomplete')
    if audit['n_fresh_public_window_occurrences']!=2345 or audit['n_direct_booster_calls']!=128 or audit['n_portable_window_calls']!=560 or audit['n_additional_shuffled_target_calls']!=140 or audit['n_future_history_poison_rejections']!=8:raise ValueError('Deployment verification incomplete')
    if not diagnostic['uses_public_assessment_labels'] or diagnostic['deployment_allowed'] or diagnostic['new_model_or_candidate']:raise ValueError('Label oracle must remain evaluator-only')
    parameter_checks=[]
    for case in study['cases']:
        if len(case['training'])!=4 or sha256_file(case['incumbent']['path'])!=case['incumbent']['sha256']:raise ValueError('Matched scope/reference changed')
        for record in case['training']:
            model=Booster(model_file=record['model']['path']);kind=record['kind'];objective='binary' if kind=='classifier' else 'lambdarank'
            if model.params['objective']!=objective or model.params['num_leaves']!=63 or model.params['feature_fraction']!=.9 or model.num_trees()!=record['trees'] or record['trees']>600:raise ValueError('Matched learner objective/capacity differs')
            if kind=='ranker' and (model.params['lambdarank_truncation_level']!=20 or not model.params['lambdarank_norm']):raise ValueError('Ranking objective parameters differ')
            parameter_checks.append({'source_study':case['source_study'],'kind':kind,'seed':record['seed'],'objective':objective,'rows':record['rows'],'queries':record['queries'],'trees':record['trees'],'header_parameters_exact':True})
    for path,digest in audit['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Public raw/masked source changed')
    for stem in stems:
        for p in sorted((Path('artifacts/research')/stem).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip','.txt','.parquet','.pkl'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'completed_public_onset_ranking'});known.add(p.as_posix())
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),
        'n_artifacts':len(artifacts),'artifacts':artifacts,'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],'scope':'Sixteen public-onset models,all cached evidence/portable roles and negative results;label-oracle diagnostic is not a candidate. Old complete submissions unchanged'}
    verification={'created_UTC':now,'new_unit_tests':{'passed':3,'seconds':1.15,'scope':'Negative/raw scores,ties,full-domain constraints,permutations,set-size caps and absence of future mask input'},
        'research_pytest':{'passed':131,'seconds':12.68,'session_id':52799,'basetemp':'.cache/pytest_onset_rank_research_v1'},
        'CPU_pytest':{'passed':110,'optional_Torch_modules_skipped':7,'seconds':7.34,'basetemp':'.cache/pytest_onset_rank_CPU_v1'},
        'compileall_passed':True,'parameter_checks':parameter_checks,'fresh_public_window_occurrences':2345,'direct_booster_calls':128,
        'portable_window_calls':560,'additional_shuffled_target_calls':140,'future_history_poison_rejections':8,
        'scope':'All32cached populations rebuilt;portable interface sampled first/last eachcondition per evaluation cache. Roles/origins repeat;no independent refit. Tests and verification partly concurrent.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:synthetic Queue teacher controls and independent verification completed,commit0bb0122','current_goal_turn_classification':'progress',
        'objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,
        'completed':['Sixteen matched public-onset classifier/ranker fits with frozen ongoing predictions','Four decision/selection/assessment mechanisms and full source/portable deployment audits','Evaluator-only cardinality headroom diagnostic;no oracle decoder exported'],
        'cases':[{'source_study':c['source_study'],'decision_selected':c['decision_selected'],'selected':c['selected'],'assessment_comparisons':c['comparisons']} for c in study['cases']],
        'active_jobs':[],'new_complete_submission':False,'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only',
        'limits':'Noisy public observation proxies and previously viewed dates,not official latent Queue or blind final evidence. Fixed-k/ranking changed onset only. Label-oracle metrics are unavailable to legal prediction. Four folds share training dates/events;no formal gain.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('onset_rank_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'models',len(parameter_checks),flush=True)


if __name__=='__main__':main()
