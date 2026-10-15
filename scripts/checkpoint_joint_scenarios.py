"""Freeze joint-scenario results, numerical-audit repair and negative evidence."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import numpy as np
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def disagreement_locations(cases):
    records=[]
    for case in cases:
        totals={k:{'onset_window_disagreements':0,'scoreable_changed_cells':0,'unscoreable_changed_cells':0} for k in ['full','empirical']};seen=set()
        for item in case['source_records']:
            r=item['source_record']
            if r['phase']!='assessment':continue
            p=next(p for p in case['prediction_records'] if p['phase']=='assessment' and p['seed']==r['seed'])
            with np.load(r['path']) as a,np.load(p['path']) as q:
                width=6*len(a['links'])
                for i,origin in enumerate(a['origins']):
                    if a['ongoing'][i] or str(origin) in seen:continue
                    seen.add(str(origin));ids=slice(i*width,(i+1)*width);mask=a['available'][i].ravel()
                    for name,d in totals.items():
                        changed=q['copula_'+name][ids]!=q['independent_'+name][ids];d['onset_window_disagreements']+=int(changed.any());d['scoreable_changed_cells']+=int((changed&mask).sum());d['unscoreable_changed_cells']+=int((changed&~mask).sum())
        records.append({'case':case['name'],'copula_versus_matched_independent':totals})
    return {'cases':records,'scope':'Evaluator masks used for posthoc error-location diagnostics ONLY;not any prediction/parameter/selection input.'}


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_joint_scenario_v1.json','joint_scenario_verification_record_v1.json','research_checkpoint_joint_scenario_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite joint-scenario checkpoint')
    parent_path=root/'model_artifacts_checkpoint_probabilistic_onset_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    stems=['C78_nonempty_joint_mask_stress','C79_joint_scenario_onset','C80b_joint_scenario_verification','C81_joint_scenario_matched_summary'];stress,study,audit,summary=[read(root/(s+'.json')) for s in stems];cfg=study['registered']['protocol']
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stems[1]+'.json')) or summary['source_study_sha256']!=sha256_file(root/(stems[1]+'.json')) or summary['independent_audit_sha256']!=sha256_file(root/(stems[2]+'.json')) or summary['known_law_stress_sha256']!=sha256_file(root/(stems[0]+'.json')):raise ValueError('New report chain differs')
    raw_proof=read(root/'C72b_rolling_onset_threshold_verification.json');probability_proof=read(root/'C75_probabilistic_onset_verification.json')
    if sha256_file(root/'C72b_rolling_onset_threshold_verification.json')!=cfg['source_raw_proof_sha256'] or sha256_file(root/'C75_probabilistic_onset_verification.json')!=cfg['source_audit_sha256']:raise ValueError('Reused proof changed')
    for path,digest in {**raw_proof['source_files_sha256'],**probability_proof['source_files_sha256'],**cfg['unchanged_source_constructors_sha256']}.items():
        if sha256_file(path)!=digest:raise ValueError('Audited public input changed')
    expected={'fit_banks_verified':4,'fit_onset_window_occurrences':289,'scenario_curve_checks':3076,'decision_onset_Booster_calls':8,'decision_reference_probability_calls':4,'portable_history_windows':109,'portable_role_calls':981,'additional_shuffled_calls':504,'future_history_rejections':36,'new_raw_window_rebuilds':0}
    if audit['totals']!=expected or len(study['cases'])!=4 or not all(c['fit_bank_member_curves_policy_gates_independently_verified'] for c in audit['cases']) or audit['maximum_matrix_cumsum_curve_difference']>1e-10:raise ValueError('Independent bank/curve/interface scope incomplete')
    if study['new_scenario_banks']!=4 or study['scenarios_per_bank']!=4096 or study['new_tree_models']!=0 or study['new_probability_calibrators']!=0 or len(summary['comparisons'])!=13 or len(summary['known_nonempty_stress_independent_checks'])!=9:raise ValueError('Study scope differs')
    if any(c['independent_replicate_maximum_difference']>1e-12 for c in summary['comparisons']) or any(c['maximum_difference']>1e-12 for c in summary['known_nonempty_stress_independent_checks']):raise ValueError('Independent summary differs')
    locations=disagreement_locations(study['cases'])
    for stem in stems+['C80_joint_scenario_verification']:
        for p in sorted((Path('artifacts/research')/stem).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip','.pkl','.txt','.parquet'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'joint_dependence_eligibility_scenario_study_complete'});known.add(p.as_posix())
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],'scope':'Four fit-only binary-rank/eligibility scenario banks,portable prefix models,exact nonempty-selection laws and independent audits;all old artifacts/complete submissions unchanged.'}
    verification={'created_UTC':now,'new_unit_tests':{'passed':3,'seconds':1.61,'scope':'Scalar weighted prefix sets/evidence/permutations;latent-nonempty versus scored-empty;fit-bank seeded replay/missing-label poison'},
        'research_pytest':{'passed':144,'seconds':12.53,'session_id':22370,'basetemp':'.cache/pytest_joint_scenario_research_v1'},'CPU_pytest':{'passed':123,'optional_Torch_modules_skipped':7,'seconds':7.94,'basetemp':'.cache/pytest_joint_scenario_CPU_v1'},'compileall_passed':True,
        'new_independent_audit':expected,'reused_full_raw_window_proof':2196,'reused_separate_probability_calibration_proof':'C75_probabilistic_onset_verification','maximum_matrix_cumsum_curve_difference':audit['maximum_matrix_cumsum_curve_difference'],
        'independent_full_MonteCarlo_curve_error_vs_exact_IID':audit['independent_full_MonteCarlo_curve_error_vs_exact_IID'],'joint_intervals':{'contrasts':13,'resamples_each':4000,'maximum_independent_difference':max(c['independent_replicate_maximum_difference'] for c in summary['comparisons'])},
        'known_nonempty_laws_independently_replayed':9,'posthoc_disagreement_locations':locations,'failed_audit_preserved':'reports/research/C80_joint_scenario_verification.failed.json',
        'scope':'Current production suites pass. First auditfloat32mean assumption repaired to integer ratio with bit-exact requirement unchanged;original failed source retained. Four banks/2MCmembers each,not new tree/calibration fits;old raw proofs reused,not rebuilt. Timing may include concurrent work.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'previous_goal_turn_classification':'progress:dfd24e7 completed C74-C77','current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved',
        'formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,'new_complete_submission':False,'active_jobs':[],
        'cases':[{'name':c['name'],'decision_selected':c['decision_selected'],'selected':c['selected'],'assessment_comparisons':c['comparisons']} for c in study['cases']],'evidence_gates':summary['evidence_gates'],
        'completed':['Exact latent-nonempty/future-mask laws quantify observed-positive selection bias with six-step evidence','Fit-only empirical binary-dependence/eligibility banks and four matched scenario laws;all primary retain incumbent','No stable gain over matched independent/full model;full-law changed cells all locally unscoreable','All scenario/member curves independently verified;restricted-prefix counterexample retained'],
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only','limits':'Observed noisy proxy with forced scored positives;unconditional fit masks/dependence and restricted prefix search may not generalize. MC members estimate simulation noise only;not training-model uncertainty or formal Queue/Total.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    print('joint_scenario_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),flush=True)


if __name__=='__main__':main()
