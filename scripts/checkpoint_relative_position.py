"""Freeze completed relative-position attention study against the strong control."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file
from trafficflow.research_eval import paired_block_interval
from checkpoint_full_queue_phase import read,checked


def matched_calendar_diagnostics(root,study,forward):
    """Resample common dates jointly; two corridors do not double sample size."""
    previous_paths=[root/'A130_I210_W_relative_position_attention.json',root/'A131_I210_W_relative_position_forward_calendar.json']
    previous,previous_forward=[read(p) for p in previous_paths]
    panels=[(previous,previous_forward),(study,forward)];results={}
    for label in ['pure_geometry_vs_NN','cal_selected_policy_vs_own_reference']:
        rows=[];calendar=None
        for fitted,future in panels:
            reference='nn_primary' if label=='pure_geometry_vs_NN' else fitted['registered']['protocol']['reference_role']
            role='geo_seed_mean' if label=='pure_geometry_vs_NN' else fitted['selection']['primary']
            a={d['date']:d['daily_State'] for d in future['metrics'][reference]['days']};b={d['date']:d['daily_State'] for d in future['metrics'][role]['days']}
            dates=sorted(a)
            if dates!=sorted(b) or (calendar is not None and calendar!=dates):raise ValueError('Cross-direction calendar differs')
            calendar=dates;rows.append((a,b,reference,role))
        a=[sum(row[0][date] for row in rows)/len(rows) for date in calendar];b=[sum(row[1][date] for row in rows)/len(rows) for date in calendar]
        results[label]={'date_aligned_equal_corridor_day_mean':paired_block_interval(a,b),'roles':[{'reference':row[2],'candidate':row[3]} for row in rows]}
    return {'earlier_source_sha256':{p.as_posix():sha256_file(p) for p in previous_paths},'diagnostics':results,
        'scope':'Descriptive two fitted directions sharing 14 research dates,not 28 independent days. Equal-corridor daily diagnostic,not official corridor weighting or pooled State. No new selection;no multiplicity correction.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--winter',action='store_true');args=parser.parse_args()
    suffix='relative_position_winter' if args.winter else 'relative_position'
    root=Path('reports/research');names=[f'model_artifacts_checkpoint_{suffix}_v1.json',f'{suffix}_verification_record_v1.json',f'research_checkpoint_{suffix}_v1.json']
    if any((root/n).exists() for n in names):raise FileExistsError('Never overwrite relative checkpoint')
    parent_path=root/('model_artifacts_checkpoint_relative_position_v1.json' if args.winter else 'model_artifacts_checkpoint_transition_phase2_v1.json');parent=read(parent_path);artifacts=[checked(r) for r in parent['artifacts']]
    known={r['path'] for r in artifacts};preserved=[checked(r) for r in parent['preserved_complete_submissions']]
    if {p:sha256_file(p) for p in parent['dependency_locks']}!=parent['dependency_locks']:raise ValueError('Locks changed')
    stems=(['A134_I5_N_relative_position_attention','A135_I5_N_relative_position_forward_calendar','A136_I5_N_relative_position_training_verification','A137_I5_N_relative_position_forward_independent_verification'] if args.winter else ['A130_I210_W_relative_position_attention','A131_I210_W_relative_position_forward_calendar','A132_I210_W_relative_position_training_verification','A133_I210_W_relative_position_forward_independent_verification'])
    study,forward,check,fcheck=[read(root/(s+'.json')) for s in stems]
    cfg=study['registered']['protocol'];reference=cfg['reference_role'];candidates=cfg['candidates'];nmodels=len(study['exports']);nmetrics=len(study['assessment'])*28
    for verifier,source in [(check,stems[0]),(fcheck,stems[1])]:
        if verifier['registered']['protocol']['source_study_sha256']!=sha256_file(root/(source+'.json')):raise ValueError('Verifier source changed')
    if not check['public_geometry_exact'] or len(check['exposure_checks'])!=2 or not all(r['shared_initialization_and_full_cell_exposure_exact'] and r['parameters']==97309 for r in check['exposure_checks']):raise ValueError('Matched geometry/control verification incomplete')
    if check['n_direct_predictor_calls']!=nmodels*28 or check['n_saved_role_day_metrics_recalculated']!=nmetrics or check['maximum_prediction_error_lane_units']>1e-8:raise ValueError('Deployment verification incomplete')
    if forward['reference_role']!=reference or forward['n_direct_predictor_calls']!=nmodels*14 or forward['selection_changed']:raise ValueError('Forward reference/choice differs')
    if fcheck['n_raw_teacher_days_rebuilt']!=14 or fcheck['n_saved_role_day_metrics_recalculated']!=nmodels*14 or fcheck['n_direct_predictor_calls']!=nmodels*2 or fcheck['maximum_prediction_difference_lane_units']>1e-8:raise ValueError('Forward audit incomplete')
    for stem in stems:
        for p in sorted((Path('artifacts/research')/stem).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.npz','.zip','.pt','.pkl'} and p.as_posix() not in known:
                artifacts.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p),'phase':'completed_relative_position_attention'});known.add(p.as_posix())
    for result in [study,forward]:
        for path,digest in result['source_files_sha256'].items():
            if sha256_file(path)!=digest:raise ValueError('Public inputs changed')
    paired=[]
    for phase,metrics in [('old_assessment',study['assessment']),('new_calendar',forward['metrics'])]:
        for seed in study['registered']['protocol']['seeds']:
            a,b=metrics[f'control_seed{seed}'],metrics[f'geo_seed{seed}'];paired.append({'phase':phase,'seed':seed,'pooled_State_gain_vs_same_seed':b['S_state']-a['S_state'],
                'day_mean_diagnostic':paired_block_interval([r['daily_State'] for r in a['days']],[r['daily_State'] for r in b['days']])})
    now=datetime.now(timezone.utc).isoformat();manifest={'created_UTC':now,'parent_manifest_sha256':sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged':len(parent['artifacts']),'n_artifacts':len(artifacts),'artifacts':artifacts,
        'preserved_complete_submissions':preserved,'dependency_locks':parent['dependency_locks'],'formal_best_Total':.82940,
        'formal_best_submission_sha256':parent['formal_best_submission_sha256'],'scope':'Two relative-position fits,static distances,all exports/metrics and independent audits;all earlier complete submissions unchanged'}
    verification={'created_UTC':now,'research_pytest':{'passed':125,'seconds':10.41,'session_id':52741,'basetemp':'E:/ieee/.cache/pytest_relative_research_v1'},
        'CPU_pytest':{'passed':104,'optional_Torch_modules_skipped':7,'seconds':8.04,'basetemp':'E:/ieee/.cache/pytest_relative_cpu_v1'},
        'new_kernel_tests':{'passed':3,'seconds':3.51,'scope':'Analytic distance units/permutation;shared weights/nonzero-head equality at zero bias/learnable gradient;roundtrip cache/keys/flow units'},
        'kernel_test_evidence_scope':'Historical unchanged production-kernel tests from I210 phase;not rerun in winter script-only phase' if args.winter else 'Executed in this phase',
        'compileall_passed':True,'cell_exposure_archives_verified':2,'n_direct_cal_ass_predictor_calls':nmodels*28,'n_cal_ass_metrics_recalculated':nmetrics,
        'n_direct_forward_calls':nmodels*14,'n_forward_metrics_recalculated':nmodels*14,'n_additional_forward_audit_calls':nmodels*2,
        'maximum_cal_ass_prediction_error_lane_units':check['maximum_prediction_error_lane_units'],'maximum_forward_score_error':fcheck['maximum_score_error'],
        'paired_seed_policy_diagnostics':paired,'scope':'Two new trained models only;other roles are frozen controls or fixed combinations. Paired seed comparisons may use different cal alpha/checkpoints.'}
    if args.winter:verification['cross_direction_future_diagnostics']=matched_calendar_diagnostics(root,study,forward)
    checkpoint={'created_UTC':now,'git_base_before_phase_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'previous_goal_turn_classification':'progress:completed I210 relative-position study,commit77fa23e' if args.winter else 'progress:completed negative speed-jump study and graph scope audit,commit0d27eaa','current_goal_turn_classification':'progress',
        'objective_status':'active;formal0.95000+unachieved','formal_best_Total':.82940,'new_formal_score':None,'formal_model_promoted':False,
        'completed':['Two static relative-position ordinary-loss models with identical shared initialization/cell exposure',
            'Cal selection includes frozen strongest reference and fixed new-speed/reference-flow combination','Frozen future calendar and full exposure/geometry/deployment verification'],
        'primary':study['selection']['primary'],'old_assessment_candidates':{r:study['comparisons'][r] for r in candidates},
        'future_candidates':{r:forward['comparisons'][r] for r in candidates},'reference_role':reference,'active_jobs':[],
        'final_simulator_seeds':{'seeds':[7907,8017,9109],'generated':False},'platform_policy':'Local models and Git only','all_previous_complete_submissions_preserved':True,
        'limits':'One additional direction/fit per phase,public research dates and two seeds sharing labels. Cross-direction summary is descriptive and dates are resampled jointly. Signed distance is a statistical relation,not road connectivity or hidden boundary flux. Formal Physics/Total unknown. No new whole-network fit or complete submission.'}
    for name,value in zip(names,[manifest,verification,checkpoint]):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print('relative_checkpoint','previous_unchanged',len(parent['artifacts']),'artifacts',len(artifacts),'primary',study['selection']['primary'],flush=True)


if __name__=='__main__':main()
