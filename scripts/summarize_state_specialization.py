"""Posthoc factor comparisons on the rejected State decision screen."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.state_control_eval import paired_state_bootstrap


REGIMES=['R1','R2','R3'];SEEDS=[20261013,20261029]


def scalar_point(rows,role):
    values=[]
    for regime in REGIMES:
        n=sum(r['models'][role][regime]['n'] for r in rows)
        if not n:return None
        v=(sum(r['models'][role][regime]['speed_normalized_SSE'] for r in rows)/n)**.5;q=(sum(r['models'][role][regime]['flow_normalized_SSE'] for r in rows)/n)**.5;values.append(.54*max(0.,1-v)+.46*max(0.,1-q))
    return sum(values)/3.


def compared(cases,baseline,candidate):
    result=paired_state_bootstrap(cases,baseline,candidate);n=len(cases[0]['dates']);rng=np.random.default_rng(20261015);starts=rng.integers(0,n,(4000,int(np.ceil(n/7))));ix=((starts[:,:,None]+np.arange(7))%n).reshape(4000,-1)[:,:n];samples=[]
    for indices in ix:
        values=[]
        for case in cases:
            rows=[case['records'][int(i)] for i in indices];a=scalar_point(rows,baseline);b=scalar_point(rows,candidate)
            if a is None or b is None:break
            values.append(b-a)
        if len(values)==len(cases):samples.append(sum(values)/len(values))
    point=sum(scalar_point(c['records'],candidate)-scalar_point(c['records'],baseline) for c in cases)/len(cases);interval=np.quantile(samples,[.025,.975]).tolist() if samples else None
    if abs(point-result['gain'])>1e-12 or len(samples)!=result['valid_resamples'] or not np.allclose(interval,result['ci95'],atol=1e-12,rtol=0):raise ValueError('Independent scalar factor interval differs')
    result['independent_all_retained_scalar_replicates_checked']=True;return result


def exposure(case):
    packs=[dict(np.load(r['path'])) for r in case['dataset_description']['fit_day_records']];result=[]
    for r in case['matched_trees']:
        if sha256_file(r['teacher_archive']['path'])!=r['teacher_archive']['sha256']:raise ValueError('Exposure data changed')
        a=dict(np.load(r['teacher_archive']['path']));coordinates=a['context_coordinates'];domain=[]
        for day,at in coordinates:
            p=packs[int(day)];domain.append(p['available'][int(at)][None,:]&~(p['masked_channels'][int(at),:,2:4].T>.5))
        active=np.array([d.any(axis=1) for d in domain]);counts=a['context_stage_counts'];draws=counts.sum(axis=1);row_regime=np.array([packs[int(day)]['mask_regime'][int(at),int(node)] for day,at,node in a['coordinates']]);regimes={}
        for name in REGIMES:
            use=row_regime==name;regimes[name]={'unique_teacher_rows':int(use.sum()),'normalized_weight_share_by_field':[float(a['normalized_weights'][use,i].sum()/a['normalized_weights'][:,i].sum()) for i in range(2)]}
        result.append({'seed':r['seed'],'n_unique_contexts':len(coordinates),'n_teacher_rows_by_field':a['domain'].sum(axis=0).tolist(),'n_context_draws':int(draws.sum()),'draws_with_current_label_by_field':(draws[:,None]*active).sum(axis=0).tolist(),
            'nominal_weight_sum_by_field':a['nominal_weights'].sum(axis=0).tolist(),'normalized_weight_quantiles_by_field':{str(i):np.quantile(a['normalized_weights'][:,i][a['domain'][:,i]],[0,.5,.9,.99,1]).tolist() for i in range(2)},'regime_exposure':regimes})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite State control summary')
    study_path=Path('reports/research/C95_state_specialization_controls.json');study=json.loads(study_path.read_text());audit=json.loads(args.audit.read_text())
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(study_path):raise ValueError('Accepted State audit source differs')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');tic=time.perf_counter();cases=[c['decision'] for c in study['cases']];pairs=[('full_tree','joint_queue_cp'),('joint_queue_cp','joint_state_event_cp'),('joint_state_event_cp','joint_state_full_cp'),('joint_queue_cp','joint_state_full_cp'),('joint_state_full_cp','state_only'),('full_tree','matched_l2'),('matched_l2','matched_l2_weighted'),('matched_l2_weighted','matched_l1_weighted'),('matched_l1_weighted','state_only'),('full_tree','state_only'),('state_only_event_cp','state_only'),('causal_carry','state_only')];comparisons=[compared(cases,a,b) for a,b in pairs];member=[]
    for seed in SEEDS:
        for a,b in [('full_tree','matched_l2'),('joint_state_full_cp','state_only'),('matched_l2','matched_l2_weighted'),('matched_l2_weighted','matched_l1_weighted')]:member.append(compared(cases,a if a=='full_tree' else a+'_'+str(seed),b+'_'+str(seed)))
    points={role:float(np.mean([c['decision']['metrics'][role]['public_subset_regime_macro'] for c in study['cases']])) for role in ['full_tree','matched_l2','matched_l2_weighted','matched_l1_weighted','state_only']};chain=['full_tree','matched_l2','matched_l2_weighted','matched_l1_weighted','state_only'];steps=[{'from':a,'to':b,'difference':points[b]-points[a]} for a,b in zip(chain,chain[1:])];total=points['state_only']-points['full_tree']
    if abs(sum(r['difference'] for r in steps)-total)>1e-12:raise ValueError('Factor diagnostic chain does not telescope')
    gradients=[r for c in study['cases'] for g in c['gradient_diagnostics'] for r in g['rows']];cos=np.array([r['cosine'] for r in gradients if r['cosine'] is not None]);case_results=[]
    for case in study['cases']:
        choices={seed:{role:r['step'] for role,r in rows.items()} for seed,rows in case['choices'].items()};case_results.append({'name':case['name'],'decision_metrics':case['decision']['metrics'],'actual_exposure':exposure(case),'checkpoint_choice_steps':choices})
    fp_cases=[c['full_FP32_phases']['decision'] for c in audit['cases']];fp_check={}
    for role in ['full_tree','matched_l2','matched_l2_weighted','matched_l1_weighted','state_only','joint_state_full_cp']:fp_check[role]={'recorded_TF32_point':float(np.mean([scalar_point(c['records'],role) for c in cases])),'corrected_FP32_point':float(np.mean([scalar_point(c['records'],role) for c in fp_cases]))}
    result={'source_study_sha256':sha256_file(study_path),'source_audit_sha256':sha256_file(args.audit),'source_snapshot':snapshot,'decision_factor_comparisons':comparisons,'single_seed_control_comparisons':member,'diagnostic_chain':{'steps':steps,'sum':total,'scope':'Algebraicsequentialcontrolchain,not causalpercentage/Shapley decomposition. Fulltreevs2seedmatchedtree alsoincludesensemble;single-seedcoverage contrastsreportedseparately. RemainingNN/treegap includesfeatureengineering/optimizer,notpurearchitecture.'},
        'gradient_diagnostic_summary':{'n_pairs':len(gradients),'n_defined_cosines':len(cos),'n_negative':int((cos<0).sum()),'mean_cosine':float(cos.mean()),'minmax_cosine':[float(cos.min()),float(cos.max())],'scope':'32frozenFITbatches,localalignmentonly;removingtasksdidnotshowconsistentStategain. Not proof ofcausalnegative-transfer mechanism.'},'full_FP32_point_check':fp_check,'cases':case_results,'seconds':time.perf_counter()-tic,'formal_State':None,'formal_Queue':None,'formal_Physics':None,'formal_Total':None,
        'scope':'PosthocdecisionperiodpublicnoisycausalState factor diagnostics,not independenttest or officialState. All12mean+8single-seedintervals uncorrected/conditional onfittedmodels. No new rawlabels/laterperiods,modelchoice or formalpromotion. Screen rejectedall;selection/assessment/late_transferunknown fornewcandidates.'};raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'experiment.json').write_text(raw,encoding='utf-8');print('State_control_summary_COMPLETE','comparisons',len(comparisons)+len(member),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
