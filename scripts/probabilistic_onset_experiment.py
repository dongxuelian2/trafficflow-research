"""Frozen rolling trees: separate calibration and expected-IoU set decisions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import Booster
from trafficflow.data import Release,eligible
from trafficflow.experiments import partition_date,raw_for
from trafficflow.queue import QueueModel,queue_features
from trafficflow.queue_events import sample_events
from trafficflow.causal_queue import released_origin_rows
from trafficflow.expected_iou_queue import fit_monotone_platt,apply_platt,onset_expected_iou_decisions,ProbabilisticOnsetQueueModel
from trafficflow.onset_cardinality_queue import onset_threshold_decisions
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.queue_domain import score_full_forecast
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from compare_queue_events import comparison


ORDER=['incumbent','threshold_raw','threshold_cal','iou_raw','iou_cal','iou_cal_nonempty']
CONSTRUCTORS=['src/trafficflow/queue.py','src/trafficflow/full_domain_queue.py','src/trafficflow/onset_rank_queue.py','src/trafficflow/causal_queue.py','src/trafficflow/data.py','src/trafficflow/queue_events.py']


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def record(path):return {'path':Path(path).as_posix(),'sha256':sha256_file(path)}


def calibration_cache(release,case,links,folder,feature_names):
    dates=case['dates']['checkpoint'];paths=[p for p in release.files(case['panel'],'train') if partition_date(release,p) in dates];frames=list(raw_for(release,case['panel'],paths));net=release.network(case['panel']);topology=release.topology(case['panel'])
    def origin_reader(t):return released_origin_rows(release,case['panel'],'train',t)
    try:
        _,meta=sample_events(frames,net,topology,case['panel'],4129,per_condition=30,spacing_minutes=360,origin_reader=origin_reader,include_occupancy=True,onset_last_step=True,balanced_conditions=True)
    except ValueError as error:
        if str(error)!='No eligible proxy events':raise
        meta={'windows':[],'selected_counts':{'queue_onset':0,'queue_ongoing':0},'reason':'No eligible proxy events'}
    by_date={str(f.timestamp.iloc[0])[:10]:f for f in frames};features={a:[] for a in ['plain','rich']};labels=[];available=[];origins=[];ongoing=[];windows=[];conditions=[];leads=[]
    for row in sorted(meta['windows'],key=lambda r:r['origin']):
        origin=pd.Timestamp(row['origin']);raw=by_date[origin.date().isoformat()];times=sorted(raw.timestamp.unique());at=row['at']
        if pd.Timestamp(times[at])!=origin or raw.link_id.drop_duplicates().tolist()!=links:raise ValueError('Calibration domain changed')
        history=pd.concat([raw.loc[raw.timestamp.isin(times[at-12:at])],origin_reader(origin)],ignore_index=True);target=pd.MultiIndex.from_product([times[at+1:at+7],links],names=['timestamp','link_id']).to_frame(index=False)
        for arm in features:
            x=queue_features(history,target,net,topology,origin,allow_origin=True,include_occupancy=arm=='rich')
            if arm=='rich':x['condition_ongoing']=float(row['condition']=='queue_ongoing')
            if list(x.columns)!=feature_names[arm]:raise ValueError('Frozen model calibration schema differs')
            features[arm].append(x)
        future=raw.loc[raw.timestamp.isin(times[at+1:at+7])].copy();future.loc[~eligible(future),'speed_kmh']=np.nan
        v=future.pivot_table(index='timestamp',columns='link_id',values='speed_kmh').reindex(index=times[at+1:at+7],columns=links).to_numpy(float)
        labels.append((v<=net.v_cut.reindex(links).to_numpy(float)).astype(np.float32));available.append(np.isfinite(v));origins.append(origin.isoformat());ongoing.append(row['condition']=='queue_ongoing')
        width=len(target);windows.extend([f'{case["panel"]}:{times[at]}:4129']*width);conditions.extend([row['condition']]*width);leads.extend(np.repeat([5,10,15,20,25,30],len(links)))
    a={'labels':np.stack(labels) if labels else np.empty((0,6,len(links)),np.float32),'available':np.stack(available) if available else np.empty((0,6,len(links)),bool),
        'origins':np.array(origins),'ongoing':np.array(ongoing,bool),'window':np.array(windows),'condition':np.array(conditions),'lead_minutes':np.array(leads,int),'links':np.array(links)}
    path=folder/'calibration_4129.npz';np.savez_compressed(path,**a);files=[]
    for arm in features:
        features[arm]=pd.concat(features[arm],ignore_index=True) if features[arm] else pd.DataFrame(columns=feature_names[arm],dtype=float)
        p=folder/('calibration_4129_'+arm+'.parquet');features[arm].to_parquet(p,index=False);files.append(record(p))
    return a,features,{'phase':'probability_calibration','seed':4129,**record(path),'feature_files':files,'sampler':meta,'n_windows':len(origins),'dates':dates}


def decode(a,raw,cal,baseline,raw_setting,cal_setting):
    use=a['condition']=='queue_onset';links=np.tile(a['links'],len(a['ongoing'])*6);meta=(a['window'][use],a['lead_minutes'][use],links[use]);pred={'incumbent':baseline.copy()}
    for role,p,setting in [('threshold_raw',raw,raw_setting),('threshold_cal',cal,cal_setting)]:
        pred[role]=baseline.copy();pred[role][use]=onset_threshold_decisions(p[use],*meta,setting['threshold'],setting['force'])
    for role,p,allow in [('iou_raw',raw,True),('iou_cal',cal,True),('iou_cal_nonempty',cal,False)]:
        pred[role]=baseline.copy();pred[role][use]=onset_expected_iou_decisions(p[use],*meta,allow_empty=allow)
    return pred


def metric(a,p):return score_full_forecast(a['labels'].ravel(),p,a['window'],a['condition'],a['available'].ravel())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();report=Path('reports/research')/(args.output.name+'.json')
    if report.exists() or args.output.exists():raise FileExistsError('Never overwrite probabilistic Queue evidence')
    source_path=Path('reports/research/C71_rolling_onset_threshold_refits.json');source=json.loads(source_path.read_text());proof_path=Path('reports/research/C72b_rolling_onset_threshold_verification.json');proof=json.loads(proof_path.read_text());parent_path=Path('reports/research/model_artifacts_checkpoint_rolling_onset_v1.json');parent=json.loads(parent_path.read_text());known={r['path']:r['sha256'] for r in parent['artifacts']}
    if proof['registered']['protocol']['source_study_sha256']!=sha256_file(source_path):raise ValueError('Reused raw audit source changed')
    old_source={r['path']:r['sha256'] for r in proof['source_snapshot']['captured_files']};constructors={p:sha256_file(p) for p in CONSTRUCTORS}
    if any(old_source[p]!=h for p,h in constructors.items()):raise ValueError('Reused feature/sampler constructor changed')
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Frozen_rolling_trees_calibrated_independent_expected_IoU','source_study_sha256':sha256_file(source_path),'source_manifest_sha256':sha256_file(parent_path),
        'previous_goal_turn':'progress:c480022 completed four rolling refits/audits,formal0.950+unmet','reused_raw_input_proof':proof_path.as_posix(),'reused_raw_input_proof_sha256':sha256_file(proof_path),'unchanged_source_constructors_sha256':constructors,
        'frozen_models':'C71 all four onset classifiers/case and strong reference;no new tree/neural fitting',
        'calibration':'Previously unused15day checkpoint buffer,both arms before decision,same balanced30/condition spacing360 seed4129 public-observation sampler. Fit only eligible onset last-step rows,window-equal weighted logloss +.01/2*((slope-1)^2+intercept^2),slope[.05,5],intercept[-8,8],clip1e-6. Fewer3windows/missingclass ->identity. No tuning regularization/bounds.',
        'decoders':ORDER,'IoU':'Own exact suffix Bernoulli convolution,all k0..N;empty/empty=1. Conditional independence and full-domain risk are modeling assumptions,not true traffic label independence or future scoring-mask knowledge. Nonempty arm forbidsk0 only,does not claim scored positives guaranteed.',
        'algorithm_source':'https://www.jmlr.org/papers/volume24/22-0712/22-0712.pdf, Lemma13;own implementation,no source-code copying',
        'decision':'Reuse C71 decision arrays/featurearm,raw threshold is frozen C71. Cal threshold31grid.05..8 x forceFalse/True,prefer noforce/closer.5. Choose max ORDER,incumbent first.',
        'selection':'Same later15days and3eventseeds:challenger needs origin macro gain>=.01,7dayblock95CI lower>0;else incumbent. Seal before assessment.',
        'limits':'Previously available/partly viewed public proxy calendars;new calibration data and fit,not blind/private/final. Old2196raw audit explicitly reused;new calibration raw will be verified separately. No future eligibility input to decoding.',
        'formal_Queue':None,'formal_Total':None,'platform':'Local models/Git only','final_simulator_seeds_generated':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');release=Release('data/kaggle_public');start=time.perf_counter();cases=[];calls=0
    for old in source['cases']:
        folder=args.output/old['name'];folder.mkdir();models={a:[] for a in ['plain','rich']};feature_names={}
        for t in old['training']:
            if t['role']!='onset':continue
            row=t['model']
            if sha256_file(row['path'])!=row['sha256'] or known[row['path']]!=row['sha256']:raise ValueError('Frozen classifier changed')
            models[t['arm']].append(Booster(model_file=row['path']));feature_names[t['arm']]=t['features']
        exports=old['exports']
        for row in exports.values():
            if sha256_file(row['path'])!=row['sha256'] or known[row['path']]!=row['sha256']:raise ValueError('Frozen portable reference changed')
        incumbent=QueueModel.load(Path(exports['incumbent']['path']));base=QueueModel.load(Path(exports['fixed_k']['path']))
        a,x,calibration_record=calibration_cache(release,old,list(base.links),folder,feature_names);use=(a['condition']=='queue_onset')&(a['lead_minutes']==30)&a['available'].ravel();states={};calibration_predictions={}
        for arm in models:
            p=np.mean([m.predict(x[arm],num_threads=4) for m in models[arm]],axis=0) if len(x[arm]) else np.array([]);calls+=len(models[arm]) if len(x[arm]) else 0;calibration_predictions[arm]=p
            states[arm]=fit_monotone_platt(p[use],a['labels'].ravel()[use],a['window'][use])
        path=folder/'calibration_probabilities.npz';np.savez_compressed(path,**calibration_predictions);cal_prediction_record=record(path);cal_seal=folder/'calibration_sealed.json';save(cal_seal,states)
        print('probabilistic_calibration',old['name'],[(k,v['n_windows'],v['slope'],v['intercept']) for k,v in states.items()],flush=True)
        arm=old['feature_arm'];reads=[];source_records=[];prediction_records=[]
        def load(r,prior):
            f=next(f for f in r['feature_files'] if Path(f['path']).stem.endswith('_'+arm))
            for row in [r,f]:
                if sha256_file(row['path'])!=row['sha256'] or known[row['path']]!=row['sha256']:raise ValueError('Reused audited cache changed')
            reads.append({'phase':r['phase'],'seed':r['seed'],'prior_seal_sha256':sha256_file(prior)});source_records.append({'source_record':r,'feature_file':f});return dict(np.load(r['path'])),pd.read_parquet(f['path'])
        def probabilities(a,x):
            nonlocal calls
            raw=np.mean([m.predict(x,num_threads=4) for m in models[arm]],axis=0);p=incumbent.baseline.probability(x);calls+=3
            baseline=full_queue_decisions(p,a['window'],a['condition'],a['lead_minutes'],incumbent.setting['thresholds'],incumbent.setting['require_nonempty'],incumbent.setting['timing']);return raw,apply_platt(states[arm],raw),baseline
        r=next(r for r in old['cache_records'] if r['phase']=='decision');a,x=load(r,cal_seal);raw,cal,baseline=probabilities(a,x);use=a['condition']=='queue_onset';links=np.tile(a['links'],len(a['ongoing'])*6);grid=[]
        for force in [False,True]:
            for threshold in np.linspace(.05,.8,31):
                s={'threshold':float(threshold),'force':force};p=baseline.copy();p[use]=onset_threshold_decisions(cal[use],a['window'][use],a['lead_minutes'][use],links[use],s['threshold'],s['force']);grid.append({'setting':s,'metrics':metric(a,p)})
        choice=max(grid,key=lambda r:(r['metrics']['proxy_IoU'],not r['setting']['force'],-abs(r['setting']['threshold']-.5)));pred=decode(a,raw,cal,baseline,old['threshold_choice']['setting'],choice['setting']);decision={k:metric(a,p) for k,p in pred.items()}
        if decision['incumbent']!=old['decision_metrics']['incumbent'] or decision['threshold_raw']!=old['decision_metrics']['threshold']:raise ValueError('C71 controls differ')
        challenger=max(ORDER,key=lambda k:decision[k]['proxy_IoU']);decision_seal=folder/'decision_sealed.json';save(decision_seal,{'threshold_grid':grid,'threshold_choice':choice,'decision_metrics':decision,'decision_selected':challenger})
        def evaluate(phase,prior,selected=None):
            result={'dates':{'assessment':old['dates'][phase]},'assessment':[]}
            for r in old['cache_records']:
                if r['phase']!=phase:continue
                a,x=load(r,prior);raw,cal,baseline=probabilities(a,x);pred=decode(a,raw,cal,baseline,old['threshold_choice']['setting'],choice['setting'])
                old_prediction=next(p for p in old['prediction_records'] if p['phase']==phase and p['seed']==r['seed'])
                if sha256_file(old_prediction['path'])!=old_prediction['sha256']:raise ValueError('C71 control prediction changed')
                with np.load(old_prediction['path']) as previous:
                    if not np.array_equal(raw,previous['onset_score']) or not np.array_equal(pred['incumbent'],previous['incumbent']) or not np.array_equal(pred['threshold_raw'],previous['threshold']):raise ValueError('Frozen control differs')
                if selected is not None:pred['selected']=pred[selected]
                metrics={k:metric(a,p) for k,p in pred.items()};result['assessment'].append({'seed':r['seed'],'models':metrics});path=folder/(phase+'_'+str(r['seed'])+'_predictions.npz');np.savez_compressed(path,raw_probability=raw,calibrated_probability=cal,**pred);prediction_records.append({'phase':phase,'seed':r['seed'],**record(path)})
            return result
        selection=evaluate('selection',decision_seal);gate=comparison(selection,'incumbent',challenger);selected=challenger if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent';selection_seal=folder/'selection_sealed.json';save(selection_seal,{'selected':selected,'decision_selected':challenger,'gate':gate})
        portable={'incumbent':incumbent,'threshold_raw':QueueModel.load(Path(exports['threshold']['path'])),'threshold_cal':ProbabilisticOnsetQueueModel(base,states[arm],threshold=choice['setting']),
            'iou_raw':ProbabilisticOnsetQueueModel(base),'iou_cal':ProbabilisticOnsetQueueModel(base,states[arm]),'iou_cal_nonempty':ProbabilisticOnsetQueueModel(base,states[arm],allow_empty=False)}
        portable['decision_selected']=portable[challenger];portable['selected']=portable[selected];new_exports={}
        for role,model in portable.items():
            path=folder/(role+'.pkl');model.save(path);new_exports[role]={**record(path),'candidate':challenger if role=='decision_selected' else selected if role=='selected' else role}
        assessment=evaluate('assessment',selection_seal,selected);case={'name':old['name'],'panel':old['panel'],'dates':old['dates'],'feature_arm':arm,'calibration_record':calibration_record,'calibration_probabilities':cal_prediction_record,'calibration_states':states,
            'threshold_grid':grid,'threshold_choice':choice,'decision_metrics':decision,'decision_selected':challenger,'selected':selected,'selection_gate':gate,'selection':selection['assessment'],'assessment':assessment['assessment'],
            'comparisons':{role:comparison(assessment,'incumbent',role) for role in ORDER+['selected']},'source_records':source_records,'prediction_records':prediction_records,'phase_reads':reads,'exports':new_exports,'seals':{n:sha256_file(folder/(n+'_sealed.json')) for n in ['calibration','decision','selection']}}
        cases.append(case);save(folder/'case_completed.json',case);print('probabilistic_case_COMPLETE',old['name'],'decision',challenger,'selected',selected,flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'new_booster_calls':calls,'new_tree_models':0,'n_calibration_states':8,'reused_raw_window_audit':2196,'new_calibration_window_occurrences':sum(c['calibration_record']['n_windows'] for c in cases),
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_Queue':None,'formal_Total':None,'formal_model_promoted':False,'new_complete_submission':False,
        'scope':'Eight monotone calibrators or declared identity fallbacks from separate earlier phase. Exact full-domain independent-label IoU and nonempty sensitivity controls;assumptions unverified for correlated traffic/future eligibility. Public development only.'}
    save(report,result);save(args.output/'experiment.json',result);print('probabilistic_onset_COMPLETE','cases',len(cases),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
