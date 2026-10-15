"""Decision-only screen: bounded joint search, independently simulated check."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import Booster
from trafficflow.queue import QueueModel
from trafficflow.joint_queue_scenarios import build_scenario_bank,bank_prefix_set
from trafficflow.scenario_local_search import bank_search,ValidatedLocalScenarioQueueModel
from trafficflow.expected_iou_queue import apply_platt,onset_expected_iou_decisions
from trafficflow.onset_cardinality_queue import onset_threshold_decisions
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from joint_scenario_onset_experiment import save,file_record,metric


ORDER=['incumbent','threshold_raw','iou_cal_nonempty','prefix','validated_local']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();report=Path('reports/research')/(args.output.name+'.json')
    if report.exists() or args.output.exists():raise FileExistsError('Never overwrite decision screen')
    root=Path('reports/research');source_path=root/'C79_joint_scenario_onset.json';source=json.loads(source_path.read_text());audit_path=root/'C80b_joint_scenario_verification.json';audit=json.loads(audit_path.read_text());manifest_path=root/'model_artifacts_checkpoint_joint_scenario_v1.json';known={r['path']:r['sha256'] for r in json.loads(manifest_path.read_text())['artifacts']}
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(source_path):raise ValueError('Source audit binding differs')
    snapshot_hashes={r['path']:r['sha256'] for r in audit['source_snapshot']['captured_files']};constructors={p:sha256_file(p) for p in ['src/trafficflow/expected_iou_queue.py','src/trafficflow/joint_queue_scenarios.py']}
    if any(snapshot_hashes[p]!=h for p,h in constructors.items()):raise ValueError('Frozen probability/bank kernels changed')
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Bounded_local_search_separate_MC_decision_screen','previous_goal_turn':'progress:1a6a921 completed C78-C81 joint scenario/no-gain evidence','source_study_sha256':sha256_file(source_path),'source_audit_sha256':sha256_file(audit_path),'source_manifest_sha256':sha256_file(manifest_path),'unchanged_kernel_sha256':constructors,
        'search':'Same C79copula/empirical bank4096scenarios,all ranked prefixes initialization,at most8single-bit improving moves,min_step1e-6,prefer removal/linkID on tie. Not globally optimal.',
        'conditional_MC_validation':'Separate fit-template banks seeds20261107/20261119,2048each. Accept proposal iff paired simulation mean gain>=.001 and approximate normal95CI lower>0. This is model-law numerical screening,not real label/test confidence.',
        'source_data':'Only existing fit labels/masks to build independent simulation bank;decision features/labels for screen. Frozen trees,calibrators,incumbent/rawthreshold/nonempty/prefix controls. No selection/assessment arrays read by worker.',
        'screen_gate':'Shortlist validated_local only if decision proxy>=incumbent+.005 and>=prefix+.005. If no case qualifies,stop this decoder expansion;do not evaluate new selection/assessment. If any qualifies,a separate frozen followup protocol is needed.',
        'scope':'Previously viewed public noisy observation proxies;not blind/final/private. Actual future labels/masks only score completed decisions. No tuning maxmoves/MCseeds/margins.',
        'formal_Queue':None,'formal_Total':None,'platform':'Local models/Git only','final_simulator_seeds_generated':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter();cases=[]
    for old in source['cases']:
        folder=args.output/old['name'];folder.mkdir();r=old['fit_record']
        if sha256_file(r['path'])!=r['sha256'] or known[r['path']]!=r['sha256']:raise ValueError('Audited fit cache changed')
        fit=dict(np.load(r['path']));onset=np.flatnonzero(~fit['ongoing']);validation=build_scenario_bank(fit['labels'][onset,-1],fit['available'][onset],seeds=(20261107,20261119));validation['links']=fit['links'];validation['fit_onset_indices']=onset;path=folder/'independent_validation_bank.npz';np.savez_compressed(path,**validation);validation_record=file_record(path);seal=folder/'validation_bank_sealed.json';save(seal,{'validation_bank':validation_record,'fit_record':r})
        for role in ['incumbent','threshold_raw','copula_empirical']:
            q=old['exports'][role]
            if sha256_file(q['path'])!=q['sha256'] or known[q['path']]!=q['sha256']:raise ValueError('Frozen model changed')
        prefix=QueueModel.load(Path(old['exports']['copula_empirical']['path']));incumbent=QueueModel.load(Path(old['exports']['incumbent']['path']));raw_threshold=QueueModel.load(Path(old['exports']['threshold_raw']['path']));models=[Booster(model_str=s) for s in prefix.base.model_strings]
        item=next(q for q in old['source_records'] if q['source_record']['phase']=='decision');record=item['source_record'];features=item['feature_file']
        for q in [record,features]:
            if sha256_file(q['path'])!=q['sha256'] or known[q['path']]!=q['sha256']:raise ValueError('Decision data changed')
        a=dict(np.load(record['path']));x=pd.read_parquet(features['path']);raw=np.mean([m.predict(x,num_threads=4) for m in models],axis=0);p=apply_platt(prefix.calibration,raw);b=incumbent.baseline.probability(x);base=full_queue_decisions(b,a['window'],a['condition'],a['lead_minutes'],incumbent.setting['thresholds'],incumbent.setting['require_nonempty'],incumbent.setting['timing']);use=a['condition']=='queue_onset';links=np.tile(a['links'],len(a['ongoing'])*6);meta=(a['window'][use],a['lead_minutes'][use],links[use]);threshold=base.copy();threshold[use]=onset_threshold_decisions(raw[use],*meta,raw_threshold.setting['threshold'],raw_threshold.setting['force']);nonempty=base.copy();nonempty[use]=onset_expected_iou_decisions(p[use],*meta,allow_empty=False)
        preds={'incumbent':base,'threshold_raw':threshold,'iou_cal_nonempty':nonempty,'prefix':base.copy(),'local_proposal':base.copy(),'validated_local':base.copy()};diagnostics=[]
        for window in pd.unique(a['window']):
            ids=np.flatnonzero(a['window']==window)
            if a['condition'][ids[0]]!='queue_onset':continue
            last=ids[a['lead_minutes'][ids]==30];indices=pd.Index(links[last]).get_indexer(prefix.bank['links']);last=last[indices];validated,proposal,d=bank_search(p[last],prefix.bank['links'],prefix.bank,validation)
            for role,values in [('prefix',d['prefix_prediction']),('local_proposal',proposal),('validated_local',validated)]:preds[role][ids]=0;preds[role][last]=values
            diagnostics.append({'window':str(window),**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in d.items()},'proposal_changed_cells':int(np.count_nonzero(proposal!=d['prefix_prediction'])),'validated_changed_cells':int(np.count_nonzero(validated!=d['prefix_prediction']))})
        metrics={role:metric(a,pred) for role,pred in preds.items()}
        for role,old_role in [('incumbent','incumbent'),('threshold_raw','threshold_raw'),('iou_cal_nonempty','iou_cal_nonempty'),('prefix','copula_empirical')]:
            if metrics[role]!=old['decision_metrics'][old_role]:raise ValueError('Frozen decision control differs')
        best=max(ORDER,key=lambda role:metrics[role]['proxy_IoU']);passes=metrics['validated_local']['proxy_IoU']>=metrics['incumbent']['proxy_IoU']+.005 and metrics['validated_local']['proxy_IoU']>=metrics['prefix']['proxy_IoU']+.005
        pred_path=folder/'decision_predictions.npz';np.savez_compressed(pred_path,raw_probability=raw,calibrated_probability=p,**preds);decision_seal=folder/'decision_screen_sealed.json';save(decision_seal,{'metrics':metrics,'decision_best':best,'local_shortlisted':bool(passes)})
        candidate=ValidatedLocalScenarioQueueModel(prefix,validation);exports={}
        for role,model in [('incumbent',incumbent),('prefix',prefix),('validated_local',candidate)]:
            q=folder/(role+'.pkl');model.save(q);exports[role]=file_record(q)
        case={'name':old['name'],'panel':old['panel'],'dates':old['dates'],'feature_arm':old['feature_arm'],'fit_record':r,'source_decision_record':item,'validation_bank':validation_record,'source_prefix_model':old['exports']['copula_empirical'],'source_controls':old['exports'],
            'decision_metrics':metrics,'decision_best':best,'local_shortlisted':bool(passes),'onset_diagnostics':diagnostics,'prediction_record':file_record(pred_path),'exports':exports,'seals':{'validation_bank':sha256_file(seal),'decision_screen':sha256_file(decision_seal)},
            'phase_reads':[{'phase':'fit','path':r['path']},{'phase':'decision','path':record['path'],'prior_seal_sha256':sha256_file(seal)}]}
        cases.append(case);save(folder/'case_completed.json',case);print('local_decision_screen_case',old['name'],'proposals',sum(bool(d['moves']) for d in diagnostics),'validated',sum(d['accepted'] for d in diagnostics),'shortlisted',passes,flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'campaign_local_shortlisted':any(c['local_shortlisted'] for c in cases),'selection_arrays_read':0,'assessment_arrays_read':0,'new_classifiers':0,'new_calibrators':0,'new_validation_simulation_banks':4,
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_Queue':None,'formal_Total':None,'formal_model_promoted':False,'new_complete_submission':False,
        'scope':'Decision-only preregistered resource screen. New MC verification is conditional on fitted hypothetical law,not independent traffic/test labels. Never promote or claim assessment gain from this screen.'}
    save(report,result);save(args.output/'experiment.json',result);print('local_decision_screen_COMPLETE','shortlisted',result['campaign_local_shortlisted'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
