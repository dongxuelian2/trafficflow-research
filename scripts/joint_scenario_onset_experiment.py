"""Public FIT binary copula/mask banks with frozen rolling probabilities."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import Booster
from trafficflow.queue import QueueModel
from trafficflow.expected_iou_queue import apply_platt,onset_expected_iou_decisions
from trafficflow.onset_cardinality_queue import onset_threshold_decisions
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.joint_queue_scenarios import build_scenario_bank,bank_prefix_set,ScenarioOnsetQueueModel
from trafficflow.queue_domain import score_full_forecast
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from compare_queue_events import comparison


LAWS=[('independent_full','independent','full'),('independent_empirical','independent','empirical'),('copula_full','copula','full'),('copula_empirical','copula','empirical')]
ORDER=['incumbent','threshold_raw','iou_cal_nonempty']+[r[0] for r in LAWS]


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def file_record(path):return {'path':Path(path).as_posix(),'sha256':sha256_file(path)}


def decode(probability,window,condition,lead,links,baseline,bank,member=None):
    """Only public metadata/causal probability and fitted scenarios are inputs."""
    result={role:baseline.copy() for role,_,_ in LAWS}
    for key in pd.unique(window):
        ids=np.flatnonzero(window==key)
        if condition[ids[0]]!='queue_onset':continue
        last=ids[lead[ids]==30];lookup=pd.Index(links[last].astype(str)).get_indexer(bank['links'].astype(str))
        if len(last)!=len(bank['links']) or (lookup<0).any():raise ValueError('Scenario bank requires complete node domain')
        ordered=last[lookup]
        for role,dependence,availability in LAWS:
            result[role][ids]=0;result[role][ordered]=bank_prefix_set(probability[ordered],bank['links'],bank,dependence,availability,member)[0]
    return result


def metric(a,p):return score_full_forecast(a['labels'].ravel(),p,a['window'],a['condition'],a['available'].ravel())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();report=Path('reports/research')/(args.output.name+'.json')
    if report.exists() or args.output.exists():raise FileExistsError('Never overwrite joint scenario study')
    root=Path('reports/research');source_path=root/'C74_probabilistic_onset_sets.json';source=json.loads(source_path.read_text());rolling_path=root/'C71_rolling_onset_threshold_refits.json';rolling=json.loads(rolling_path.read_text());audit_path=root/'C75_probabilistic_onset_verification.json';audit=json.loads(audit_path.read_text());raw_proof_path=root/'C72b_rolling_onset_threshold_verification.json';raw_proof=json.loads(raw_proof_path.read_text());parent_path=root/'model_artifacts_checkpoint_probabilistic_onset_v1.json';known={r['path']:r['sha256'] for r in json.loads(parent_path.read_text())['artifacts']}
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(source_path) or raw_proof['registered']['protocol']['source_study_sha256']!=sha256_file(rolling_path):raise ValueError('Reused proof source changed')
    constructors=source['registered']['protocol']['unchanged_source_constructors_sha256']
    if any(sha256_file(p)!=h for p,h in constructors.items()):raise ValueError('Reused public input constructors changed')
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Fit_only_binary_dependence_and_eligibility_scenario_Queue','previous_goal_turn':'progress:dfd24e7 completed C74-C77 probability set study','source_study_sha256':sha256_file(source_path),'source_rolling_study_sha256':sha256_file(rolling_path),'source_audit_sha256':sha256_file(audit_path),'source_raw_proof_sha256':sha256_file(raw_proof_path),'source_manifest_sha256':sha256_file(parent_path),'unchanged_source_constructors_sha256':constructors,
        'fit':'Only C71 fit onset windows/eligible observed labels/all6step eligibility. Empirical node positive rates from eligible rows,missing labels ignored;no fitted-classifier in-sample residuals.',
        'scenario_bank':'2fixedMonteCarlo seeds20261013/20261029,2048draws each. Bootstrap same fit windows/masks;independent uniforms and randomized binary-tie empirical dependence use same base jitters. Missing label->independent uniform. No tuned shrinkage/rho/distance hyperparameter.',
        'laws':LAWS,'reference_controls':ORDER[:3],'scope':'Only onset last step changes,ongoing copied. Rank calibrated p and search all prefixes k0..N. General joint global optimality not claimed. Six-step evidence from hypothetical fit masks;no actualfuture mask API.',
        'decision':'C71 decision arrays,old references/control probabilities verified by C75. Choose best ORDER,incumbent first. Fit bank sealed before decision;MonteCarlo member policies are diagnostics only.',
        'selection':'Later15days same3eventseeds,origin deduplicated gain>=.01 and7dayblock95CI lower>0;otherwise incumbent. Seal before assessment.',
        'assessment':'Previously viewed public observation proxy calendars,not latent/private/final. Fit mask law unconditional on current history and observation/latent mismatch are unverified assumptions.',
        'reuse':'C72b2196raw input rebuilds and C75probability/calibration proof explicitly reused;no new raw windows or probability calibrator/classifier fit in this study.',
        'algorithm_reference':'https://arxiv.org/abs/1302.7149;own binary scenario adaptation,no external code copied',
        'formal_Queue':None,'formal_Total':None,'platform':'Local models/Git only','final_simulator_seeds_generated':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter();cases=[];booster_calls=0
    for case,old in zip(source['cases'],rolling['cases']):
        if case['name']!=old['name']:raise ValueError('Source case mismatch')
        folder=args.output/case['name'];folder.mkdir();fit_record=next(r for r in old['cache_records'] if r['phase']=='fit')
        if sha256_file(fit_record['path'])!=fit_record['sha256'] or known[fit_record['path']]!=fit_record['sha256']:raise ValueError('Audited fit data changed')
        fit=dict(np.load(fit_record['path']));onset=np.flatnonzero(~fit['ongoing']);bank=build_scenario_bank(fit['labels'][onset,-1],fit['available'][onset]);bank['links']=fit['links'];bank['fit_onset_indices']=onset;path=folder/'fitted_scenario_bank.npz';np.savez_compressed(path,**bank);bank_record=file_record(path);bank_seal=folder/'bank_sealed.json';save(bank_seal,{'bank':bank_record,'fit_source':fit_record,'n_fit_onset_windows':len(onset),'seeds':bank['seeds'].tolist(),'draws_per_seed':int(bank['draws_per_seed'])})
        print('joint_scenario_bank',case['name'],'fit_onset',len(onset),'scenarios',len(bank['member']),flush=True)
        portable_controls={}
        for role in ORDER[:3]:
            r=case['exports'][role]
            if sha256_file(r['path'])!=r['sha256'] or known[r['path']]!=r['sha256']:raise ValueError('Frozen control policy changed')
            portable_controls[role]=QueueModel.load(Path(r['path']))
        reference=portable_controls['incumbent'];base=portable_controls['iou_cal_nonempty'].base;arm=case['feature_arm'];state=case['calibration_states'][arm];boosters=[]
        for t in old['training']:
            if t['role']=='onset' and t['arm']==arm:
                if sha256_file(t['model']['path'])!=t['model']['sha256']:raise ValueError('Frozen onset tree changed')
                boosters.append(Booster(model_file=t['model']['path']))
        reads=[];source_records=[];prediction_records=[]
        def load(item,prior):
            r=item['source_record'];f=item['feature_file']
            for row in [r,f]:
                if sha256_file(row['path'])!=row['sha256'] or known[row['path']]!=row['sha256']:raise ValueError('Audited eval cache changed')
            source_records.append(item);reads.append({'phase':r['phase'],'seed':r['seed'],'prior_seal_sha256':sha256_file(prior)});return dict(np.load(r['path'])),pd.read_parquet(f['path'])
        def candidates(a,p,controls,probes=False):
            links=np.tile(a['links'],len(a['ongoing'])*6);arguments=(p,a['window'],a['condition'],a['lead_minutes'],links,controls['incumbent'],bank);pred={**controls,**decode(*arguments)}
            if probes:
                for member in [0,1]:pred.update({role+'_member'+str(member):values for role,values in decode(*arguments,member=member).items()})
            return pred
        item=next(r for r in case['source_records'] if r['source_record']['phase']=='decision');a,x=load(item,bank_seal);raw=np.mean([m.predict(x,num_threads=4) for m in boosters],axis=0);booster_calls+=2;p=apply_platt(state,raw);b=reference.baseline.probability(x);baseline=full_queue_decisions(b,a['window'],a['condition'],a['lead_minutes'],reference.setting['thresholds'],reference.setting['require_nonempty'],reference.setting['timing'])
        use=a['condition']=='queue_onset';links=np.tile(a['links'],len(a['ongoing'])*6);meta=(a['window'][use],a['lead_minutes'][use],links[use]);threshold=baseline.copy();threshold[use]=onset_threshold_decisions(raw[use],*meta,old['threshold_choice']['setting']['threshold'],old['threshold_choice']['setting']['force']);nonempty=baseline.copy();nonempty[use]=onset_expected_iou_decisions(p[use],*meta,allow_empty=False);controls={'incumbent':baseline,'threshold_raw':threshold,'iou_cal_nonempty':nonempty}
        pred=candidates(a,p,controls);decision={role:metric(a,pred[role]) for role in ORDER}
        for role in ORDER[:3]:
            if decision[role]!=case['decision_metrics'][role]:raise ValueError('C74 decision control differs')
        challenger=max(ORDER,key=lambda role:decision[role]['proxy_IoU']);decision_seal=folder/'decision_sealed.json';save(decision_seal,{'decision_metrics':decision,'decision_selected':challenger})
        def evaluate(phase,prior,selected=None):
            result={'dates':{'assessment':case['dates'][phase]},'assessment':[]}
            for item in case['source_records']:
                r=item['source_record']
                if r['phase']!=phase:continue
                a,_=load(item,prior);probability_record=next(q for q in case['prediction_records'] if q['phase']==phase and q['seed']==r['seed'])
                if sha256_file(probability_record['path'])!=probability_record['sha256'] or known[probability_record['path']]!=probability_record['sha256']:raise ValueError('Frozen probability proof changed')
                with np.load(probability_record['path']) as q:p=q['calibrated_probability'];controls={role:q[role] for role in ORDER[:3]}
                pred=candidates(a,p,controls,probes=True)
                if selected is not None:pred['selected']=pred[selected]
                metrics={role:metric(a,pred[role]) for role in ORDER+(['selected'] if selected is not None else [])};result['assessment'].append({'seed':r['seed'],'models':metrics});path=folder/(phase+'_'+str(r['seed'])+'_predictions.npz');np.savez_compressed(path,**pred);prediction_records.append({'phase':phase,'seed':r['seed'],'source_probability_record':probability_record,**file_record(path)})
            return result
        selection=evaluate('selection',decision_seal);gate=comparison(selection,'incumbent',challenger);selected=challenger if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent';selection_seal=folder/'selection_sealed.json';save(selection_seal,{'selected':selected,'decision_selected':challenger,'gate':gate})
        portable={**portable_controls,**{role:ScenarioOnsetQueueModel(base,state,bank,dependence,availability) for role,dependence,availability in LAWS}};portable['decision_selected']=portable[challenger];portable['selected']=portable[selected];exports={}
        for role,model in portable.items():
            path=folder/(role+'.pkl');model.save(path);exports[role]={**file_record(path),'candidate':challenger if role=='decision_selected' else selected if role=='selected' else role}
        assessment=evaluate('assessment',selection_seal,selected);new_case={'name':case['name'],'panel':case['panel'],'dates':case['dates'],'feature_arm':arm,'fit_record':fit_record,'fitted_bank':bank_record,'n_fit_onset_windows':len(onset),'calibration_state':state,
            'decision_metrics':decision,'decision_selected':challenger,'selection_gate':gate,'selected':selected,'selection':selection['assessment'],'assessment':assessment['assessment'],'comparisons':{role:comparison(assessment,'incumbent',role) for role in ORDER+['selected']},
            'source_records':source_records,'phase_reads':reads,'prediction_records':prediction_records,'exports':exports,'seals':{n:sha256_file(folder/(n+'_sealed.json')) for n in ['bank','decision','selection']}}
        cases.append(new_case);save(folder/'case_completed.json',new_case);print('joint_scenario_case_COMPLETE',case['name'],'decision',challenger,'selected',selected,flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'new_scenario_banks':4,'MonteCarlo_members_per_bank':2,'scenarios_per_bank':4096,'new_decision_classifier_booster_calls':booster_calls,'reused_evaluation_probability_records':24,'reused_raw_window_proof':2196,'new_raw_window_rebuilds':0,
        'new_tree_models':0,'new_probability_calibrators':0,'seconds':time.perf_counter()-start,'source':evidence(),'formal_Queue':None,'formal_Total':None,'formal_model_promoted':False,'new_complete_submission':False,
        'scope':'Empirical fit-label/mask scenario law and prefix search,not a general global Bayes solution or latent-truth model. Observed-positive sampling and unconditional mask templates can shift;MC members are sensitivity probes,not independent training replicates.'}
    save(report,result);save(args.output/'experiment.json',result);print('joint_scenario_onset_COMPLETE','cases',len(cases),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
