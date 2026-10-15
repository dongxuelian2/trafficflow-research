"""Freeze synthetic Queue teacher models and all independent evidence."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    root=Path('reports/research');names=['model_artifacts_checkpoint_queue_latent_v1.json','queue_latent_verification_record_v1.json','research_checkpoint_queue_latent_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite latent Queue checkpoint')
    parent_path=root/'model_artifacts_checkpoint_relative_position_third_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']]
    preserved=[checked(r) for r in parent['preserved_complete_submissions']];known={r['path'] for r in artifacts}
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    stems=['C62_CTM_Queue_hard_soft_teachers','C63_CTM_Queue_hard_soft_independent_verification','C64_CTM_Queue_hard_soft_summary'];study,audit,summary=[read(root/(s+'.json')) for s in stems]
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or summary['source_study_sha256']!=sha256_file(root/(stems[0]+'.json')) or summary['independent_audit_sha256']!=sha256_file(root/(stems[1]+'.json')):raise ValueError('Source evidence changed')
    if len(study['training'])!=6 or any(r['n_rows']!=272907 or r['n_trees']!=300 for r in study['training']):raise ValueError('Model capacity/exposure differs')
    if audit['n_regenerated_scenes']!=268 or audit['n_independent_feature_teacher_windows']!=7090 or audit['training_rows_and_weights_verified']!=272907 or audit['maximum_feature_difference']>2e-6 or audit['maximum_soft_teacher_difference']>1e-7:raise ValueError('Scene input/teacher verification incomplete')
    if audit['n_independent_booster_calls']!=18 or audit['maximum_probability_difference']>1e-12 or audit['n_cal_grid_entries_verified']!=300 or audit['n_heldout_window_scores_verified']!=57680 or audit['maximum_score_difference']>1e-12 or not audit['primary_selection_exact']:raise ValueError('Deployment/scoring verification incomplete')
    if summary['n_seed_intervals_independently_verified']!=8 or not all(r['four_seed_diagnostic_exact'] for r in summary['seed_diagnostics']):raise ValueError('Cluster uncertainty incomplete')
    for stem in stems:
        for p in sorted((Path('artifacts/research')/stem).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip','.txt'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'completed_known_synthetic_Queue_label_noise'});known.add(p.as_posix())
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),
        'n_artifacts':len(artifacts),'artifacts':artifacts,'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],
        'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],'scope':'Six paired synthetic-only fits,268rawscenes/4datasets/predictions and independent audits;old complete submissions unchanged'}
    verification={'created_UTC':now,'new_kernel_tests':{'passed':3,'seconds':1.91,'scope':'Analytic Gaussian target,13slot feature/permutation/missing values,full-domain timing/force versus evaluator mask'},
        'research_pytest':{'passed':128,'seconds':13.63,'session_id':77894,'basetemp':'.cache/pytest_latent_queue_research_v1'},
        'CPU_pytest':{'passed':107,'optional_Torch_modules_skipped':7,'seconds':8.79,'basetemp':'.cache/pytest_latent_queue_CPU_v1'},
        'compileall_passed':True,'source_scenes_regenerated':268,'independent_feature_teacher_windows':7090,'shared_fit_rows_weights':272907,
        'direct_portable_model_calls':18,'all_cal_grid_entries':300,'heldout_window_score_checks':57680,'independently_recomputed_seed_intervals':8,
        'maximum_feature_difference':audit['maximum_feature_difference'],'maximum_probability_difference':audit['maximum_probability_difference'],'maximum_score_difference':audit['maximum_score_difference'],
        'test_timing_scope':'Tests and CPU verification overlapped;not isolated performance benchmarks'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:third-family paired State study and precision-aware audit completed,commit62aec0f','current_goal_turn_classification':'progress',
        'objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,
        'completed':['Six equal-capacity hard/soft/known-latent synthetic-teacher models','Disjoint simulator seeds and unseen incident/FD/failure/noise stress','All scenes/causal features/labels/weights,decisions and cluster diagnostics independently replayed'],
        'synthetic_cal_primary':study['selection']['primary'],'soft_observed_cal_gain':study['selection']['calibration_observed_gain_soft_vs_hard'],'heldout_seed_diagnostics':study['comparisons'],
        'active_jobs':[],'new_complete_submission':False,'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only',
        'limits':'Own CTM law,not organizer latent truth. Known synthetic sigma soft teacher assumes flat prior;not a true posterior or public competition model. Four heldout seeds,overlapping windows and correlated counterfactuals. No public-data transfer or formal Queue/Total evidence.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('latent_queue_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'primary',study['selection']['primary'],flush=True)


if __name__=='__main__':main()
