"""Freeze joint neural models, causal trees and independent public diagnostics."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
from trafficflow.runtime import sha256_file
from checkpoint_full_queue_phase import read,checked


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,default=Path('reports/research/C93_joint_dynamics_model_verification.json'));args=parser.parse_args();root=Path('reports/research');names=['model_artifacts_checkpoint_joint_dynamics_v1.json','joint_dynamics_verification_record_v1.json','research_checkpoint_joint_dynamics_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite joint dynamics checkpoint')
    parent_path=root/'model_artifacts_checkpoint_dense_pretrain_v1.json';parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']];known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Dependency locks changed')
    study_path=root/'C91_joint_flow_latent_dynamics.json';data_path=root/'C92_joint_dynamics_data_verification.json';summary_path=root/'C94_joint_dynamics_matched_summary.json';study=read(study_path);data=read(data_path);audit=read(args.audit);summary=read(summary_path)
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(study_path) or audit['registered']['protocol']['source_data_proof_sha256']!=sha256_file(data_path) or summary['independent_audit_sha256']!=sha256_file(args.audit):raise ValueError('Joint source chain differs')
    expected={'network_fits':12,'checkpoint_window_forwards':792,'evaluation_event_windows':316,'event_member_forwards':1896,'tree_fit_state_contexts':34224,'tree_fit_rows':917650,'full_state_contexts':16560,'full_state_member_forwards':99360,'full_state_days':60,'decision_grid_scores':1128,'portable_history_windows':56,'portable_Queue_calls':336,'portable_State_calls':168,'target_shuffle_calls':168,'future_rejections':18}
    if audit['totals']!=expected or data['totals']!={'fit_day_packs':124,'fit_masked_state_contexts':34224,'fit_legal_Queue_origins':33480,'fit_CP_event_windows':742,'actual_poison_checks':744}:raise ValueError('Joint audit incomplete')
    if any(audit['maximum_differences'][k]>1e-5 for k in ['speed','flow','analysis']) or audit['maximum_differences']['probability']>1e-6 or audit['maximum_differences']['independent_tree_features']>2e-6:raise ValueError('Joint forward/feature tolerances exceeded')
    source_files=dict(data['source_files_sha256'])
    for case in study['cases']:source_files.update(case['source_files_sha256'])
    for p,d in source_files.items():
        if sha256_file(p)!=d:raise ValueError('Public data source changed')
    if study['new_network_fits']!=12 or study['new_State_tree_fits']!=4 or study['formal_model_promoted'] or study['new_complete_submission'] or any(c['selected']!='incumbent' for c in study['cases']):raise ValueError('Unexpected primary promotion')
    edge_folder=Path('artifacts/research/C93_state_condition_edge');edge_folder.mkdir(exist_ok=True)
    old_module=next(r for r in study['source_snapshot']['captured_files'] if r['path']=='src/trafficflow/joint_queue_model.py');new_module=next(r for r in audit['source_snapshot']['captured_files'] if r['path']=='src/trafficflow/joint_queue_model.py')
    if sha256_file('src/trafficflow/joint_queue_model.py')!=new_module['sha256']:raise ValueError('Corrected portable source changed')
    logs=[]
    for log in sorted(root.glob('C93_state_condition_edge*.log')):
        target=edge_folder/log.name
        if not target.exists():shutil.copyfile(log,target)
        if sha256_file(log)!=sha256_file(target):raise ValueError('Boundary log changed')
        logs.append({'path':target.as_posix(),'sha256':sha256_file(target)})
    edge={'status':'corrected;allmodelsreplayedagain','original_wrapper_source':old_module,'corrected_wrapper_source':new_module,
        'issue':'Scaledfloat32speed multipliedbackby25 then comparedwithfloat64cut can flip causalStatecondition atthreshold. Training uses normalizedfloat32cut. Boundary67.86 reproduced queue_onset versus expectedqueue_ongoing.',
        'before_fix_actual_regression':{'failed':1,'seconds':3.62,'environment':'.venv-research'},'initial_probe':{'cut':67.62,'passed':1,'seconds':3.71},
        'initial_test_placement':'CPUenvironment lacks optionalTorch and didnot exerciseboundary;failure retained,testmovedtoTorchoptionalmodule. ProductionCPUdependencycontractunchanged.',
        'correction':'PortableState condition uses same x_speed<=float32(v_cut)/25 as maskedState training. No weights,losses,batches,thresholdselection or savedarray changes.',
        'prior_C93':'Original sampledalldata/modelaudit passed beforeadditionalboundarytest;doesnot prove untestedthresholdedges. Preserve priorverification/source.',
        'accepted_replay':{'path':args.audit.as_posix(),'sha256':sha256_file(args.audit)},'regression_file_sha256':sha256_file('tests/test_joint_graph_queue.py'),'logs':logs,'formal_Total':None}
    for target in [edge_folder/'experiment.json',root/'C93_state_condition_edge.json']:
        if target.exists():
            if read(target)!=edge:raise ValueError('Boundary evidence changed')
        else:
            with target.open('x',encoding='utf-8') as f:json.dump(edge,f,indent=2,allow_nan=False)
    folders=sorted(p for p in Path('artifacts/research').iterdir() if p.is_dir() and (p.name.startswith('C91_') or p.name.startswith('C92_') or p.name.startswith('C93') or p.name.startswith('C94_')))
    # Preserve observed terminal streams alongside the source/model evidence.
    for folder in folders:
        log=root/(folder.name+'.run.log')
        if log.exists():
            target=folder/'terminal.run.log'
            if target.exists():
                if sha256_file(log)!=sha256_file(target):raise ValueError('Saved terminal stream differs')
            else:shutil.copyfile(log,target)
        for failure in root.glob(folder.name+'.failed.json'):
            target=folder/'failure.json'
            if not target.exists():shutil.copyfile(failure,target)
    for folder in folders:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip','.pkl','.pt','.txt','.parquet','.log'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'joint_flow_and_latent_dynamics_no_promotion'});known.add(p.as_posix())
    if len(known)!=len(artifacts):raise ValueError('Duplicate artifact')
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,'formal_best_submission_sha256':parent['formal_best_submission_sha256'],
        'scope':'12jointNN,4causalStateboosters,allphase inputs/current/futureteachers,48weight/optimizer/snapshot forecasts,60fullStatedays,portable interfaces and independent audits/source snapshots. Prior complete submissions unchanged;no formal promotion.'}
    verification={'created_UTC':now,'initial_kernel_tests':{'passed':6,'seconds':4.84,'scope':'MaskedState versus lawfulQueuehistory isolation,units/latest/fallback/permutation,legacyzerooutputcompatibility,finite flow/GRU gradients,unknownteacher loss masking,Statepointformula/missingregimeunknown'},
        'boundary_regression':'Seventhnewtest included in158suite aftersinglebeforefixfailure;logs retained. Earlier157suite and132CPU passed beforeadditionaledgeprobe.',
        'research_pytest':{'passed':158,'seconds':14.92,'session_id':10867,'basetemp':'.cache/pytest_joint_dynamics_research_v2'},'CPU_pytest':{'passed':132,'optional_Torch_modules_skipped':9,'seconds':11.41,'session_id':65135,'basetemp':'.cache/pytest_joint_dynamics_CPU_v2'},'compileall_passed':True,
        'independent_data_checks':data['totals'],'independent_model_checks':expected,'maximum_differences':audit['maximum_differences'],'source_files_rehashed':len(source_files),'Queue_joint_comparisons':7,'State_joint_comparisons':7,
        'scope':'Initial/batches/exposure/weights/optimizer/inference and choices replayed;allfullState contexts and newteachers rebuilt. Exacttreefeaturehash uses canonical generator plus independentsemantic tolerance2e-6. No independent NN/tree re-optimization or formal/private/latent score.'}
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'previous_goal_turn_classification':'progress:13a982c C87-C90 complete','current_goal_turn_classification':'progress','objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,'new_complete_submission':False,'active_jobs':[],
        'completed':['Explicitjointflow/currentreconstructionheads and6stepgraphGRU residualrollout,matched12NNexperiment','Strictlyseparate maskedStateinputs and lawfulQueue rawpast,no rawteacher State input;4strong causal State trees','Full30daymaskedState coverage,rare drop/recovery teacher strata,independent allinference/choice/data audits;negative outcomes preserved'],
        'cases':[{'name':c['name'],'decision_selected':c['decision_selected'],'selected':c['selected'],'State_subset':{k:c['state_assessment_metrics'][k]['public_subset_regime_macro'] for k in ['causal_carry','causal_tree']+['direct_no_flow','direct_flow','latent_flow']}} for c in study['cases']],
        'next_research':'Audit current-target exposure and task-gradient conflict before larger GRU/flowweight sweeps. Compare data-matched State-specific learning or targeted teacher-noise/rare-event representation on predeclared independent phases. Keep strong Queue/State references;do not treat auxiliary flow forecast or public subset State as formal Physics/Total.',
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Localmodels and Git only','limits':'Previouslyseenpublicnoisydevelopment;causalState subset excludes firsthour andboth-fieldsmissingonly. More tree labels thanNN is explicit confound. No private/latent/officialmodule scores;formal0.95000+unmet.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    print('joint_dynamics_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'sources',len(source_files),flush=True)


if __name__=='__main__':main()
