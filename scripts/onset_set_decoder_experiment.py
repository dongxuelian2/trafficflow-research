"""Frozen onset probabilities: fixed k,threshold and fit-only count models."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import Booster
from trafficflow.queue import QueueModel
from trafficflow.onset_rank_queue import onset_topk_decisions
from trafficflow.onset_cardinality_queue import onset_summaries,fit_count_state,predict_set_sizes,onset_threshold_decisions,onset_count_decisions,OnsetSetDecoderModel
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.queue_domain import score_full_forecast
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from compare_queue_events import comparison


ORDER=['incumbent','fixed_k','count_prior','threshold','count_observed','count_adjusted']
SOURCE_CONSTRUCTORS=['src/trafficflow/queue.py','src/trafficflow/full_domain_queue.py','src/trafficflow/onset_rank_queue.py','src/trafficflow/causal_queue.py','src/trafficflow/data.py']


def summarize(data,features):
    use=data['condition']=='queue_onset';links=np.tile(data['links'],len(data['ongoing'])*6)
    return onset_summaries(features.loc[use].reset_index(drop=True),data['window'][use],data['lead_minutes'][use],links[use])


def decode(data,score,baseline,keys,summary,states,setting,fixed_k):
    use=data['condition']=='queue_onset';links=np.tile(data['links'],len(data['ongoing'])*6);meta=(data['window'][use],data['lead_minutes'][use],links[use]);preds={'incumbent':baseline.copy()};sizes={}
    pred=baseline.copy();pred[use]=onset_topk_decisions(score[use],*meta,fixed_k);preds['fixed_k']=pred
    pred=baseline.copy();pred[use]=onset_threshold_decisions(score[use],*meta,setting['threshold'],setting['force']);preds['threshold']=pred
    for role,state in states.items():
        k=predict_set_sizes(state,summary,summary[:,0]);sizes[role]=k;pred=baseline.copy();pred[use]=onset_count_decisions(score[use],*meta,dict(zip(keys,k)));preds[role]=pred
    return preds,sizes


def metric(data,pred):return score_full_forecast(data['labels'].ravel(),pred,data['window'],data['condition'],data['available'].ravel())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();report=Path('reports/research')/(args.output.name+'.json')
    if report.exists() or args.output.exists():raise FileExistsError('Never overwrite onset decoder study')
    source_path=Path('reports/research/C65_onset_query_rank_public.json');source=json.loads(source_path.read_text());proof_path=Path('reports/research/C66_onset_query_rank_public_verification.json');proof=json.loads(proof_path.read_text())
    if proof['registered']['protocol']['source_study_sha256']!=sha256_file(source_path):raise ValueError('Reused input proof source changed')
    recorded={r['path']:r['sha256'] for r in proof['source_snapshot']['captured_files']};constructor_hashes={p:sha256_file(p) for p in SOURCE_CONSTRUCTORS}
    if any(recorded[p]!=h for p,h in constructor_hashes.items()):raise ValueError('Previously audited input/score constructors changed')
    parent=Path('reports/research/model_artifacts_checkpoint_onset_rank_v1.json');known={r['path']:r['sha256'] for r in json.loads(parent.read_text())['artifacts']}
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Frozen_classifier_onset_threshold_and_fit_only_cardinality','source_study_sha256':sha256_file(source_path),
        'reused_input_proof':proof_path.as_posix(),'reused_input_proof_sha256':sha256_file(proof_path),'unchanged_source_constructors_sha256':constructor_hashes,'source_manifest_sha256':sha256_file(parent),
        'frozen_models':'C65 two-seed onset classifiers and original ongoing reference;no classifier/ranker refit',
        'candidates':ORDER,'count_features':'24 whole-window summaries of legal existing features,link-ID sorted,final lead30 metadata. No fit in-sample probabilities,labels or future quality input.',
        'count_targets':'Observed eligible positive count;adjusted count=count*N/eligible_count under unverified MCAR assumption. log1p Ridge alpha10,fit-only feature mean/std,no scaling grid;prior is adjusted-target intercept-only control.',
        'count_decoding':'expm1 clipped to[0,N],floor(value+.5),clamp1..N;predicted count is not latent truth. Three states/case,two Ridge and one scalar control.',
        'threshold_grid':np.linspace(.05,.8,31).tolist(),'threshold_force':[False,True],'threshold':'Decision only,prefer no force and threshold closer.5 on exact tie',
        'decision':'Max decision proxy over fixed candidate order,incumbent first;count models have no decision tuning',
        'selection':'Later15days:decision winner replaces incumbent only if origin-macro gain>=.01 and7day-block95CI lower>0',
        'assessment':'Previously viewed public observation proxy dates;C67 label-oracle diagnostics not read by this program or used in fitting/decoding. Not blind/final/private.',
        'scope':'Only onset final step changes;ongoing exactly frozen. Fit->count seal->decision->decision seal->selection->selection seal->assessment.',
        'formal_Queue':None,'formal_Total':None,'uploaded':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter();cases=[];calls=0
    for old in source['cases']:
        token=old['source_study'].split('_',1)[0];folder=args.output/token;folder.mkdir();models=[]
        for r in old['training']:
            if r['kind']=='classifier':
                if sha256_file(r['model']['path'])!=r['model']['sha256']:raise ValueError('Frozen classifier changed')
                models.append(Booster(model_file=r['model']['path']))
        incumbent=QueueModel.load(Path(old['incumbent']['path']));base=QueueModel.load(Path(old['exports']['classifier']['path']));records=[];reads=[]
        for row in [old['incumbent'],old['exports']['classifier']]:
            if sha256_file(row['path'])!=row['sha256'] or known[row['path']]!=row['sha256']:raise ValueError('Frozen portable model changed')
        def load(item,prior=None):
            record=item['source_record'];feature=item['used_feature_file']
            if record['phase']!='fit' and prior is None:raise ValueError('Prior choice seal required')
            for row in [record,feature]:
                if sha256_file(row['path'])!=row['sha256'] or known[row['path']]!=row['sha256']:raise ValueError('Reused audited input changed')
            reads.append({'phase':record['phase'],'seed':record['seed'],'prior_seal_sha256':sha256_file(prior) if prior else None});records.append(item)
            return dict(np.load(record['path'])),pd.read_parquet(feature['path'])
        item=next(r for r in old['source_records'] if r['source_record']['phase']=='fit');fit,x=load(item);keys,summary=summarize(fit,x);positive=[];eligible=[]
        for key in keys:
            use=(fit['window']==key)&(fit['lead_minutes']==30)&fit['available'].ravel();positive.append(float(fit['labels'].ravel()[use].sum()));eligible.append(float(use.sum()))
        target_path=folder/'training_windows.npz';np.savez_compressed(target_path,keys=keys,summary=summary,positive=positive,eligible=eligible,links=summary[:,0])
        tic=time.perf_counter();states={
            'count_prior':fit_count_state(summary,positive,eligible,summary[:,0],adjust=True,intercept_only=True),
            'count_observed':fit_count_state(summary,positive,eligible,summary[:,0]),
            'count_adjusted':fit_count_state(summary,positive,eligible,summary[:,0],adjust=True)}
        count_seconds=time.perf_counter()-tic;count_seal=folder/'count_models_sealed.json';count_seal.write_text(json.dumps(states,indent=2,allow_nan=False),encoding='utf-8')
        print('onset_set_count_fit',token,'windows',len(summary),flush=True)
        def predict(data,x):
            nonlocal calls
            score=np.mean([m.predict(x,num_threads=4) for m in models],axis=0);calls+=len(models);p=incumbent.baseline.probability(x)
            baseline=full_queue_decisions(p,data['window'],data['condition'],data['lead_minutes'],incumbent.setting['thresholds'],incumbent.setting['require_nonempty'],incumbent.setting['timing'])
            keys,summary=summarize(data,x);return score,baseline,keys,summary
        item=next(r for r in old['source_records'] if r['source_record']['phase']=='decision');data,x=load(item,count_seal);score,baseline,keys,summary=predict(data,x);grid=[]
        for force in [False,True]:
            for threshold in np.linspace(.05,.8,31):
                setting={'threshold':float(threshold),'force':force};preds,_=decode(data,score,baseline,keys,summary,{},setting,old['choices']['classifier']['k'])
                grid.append({'setting':setting,'metrics':metric(data,preds['threshold'])})
        chosen=max(grid,key=lambda r:(r['metrics']['proxy_IoU'],not r['setting']['force'],-abs(r['setting']['threshold']-.5)));setting=chosen['setting']
        preds,sizes=decode(data,score,baseline,keys,summary,states,setting,old['choices']['classifier']['k']);decision={role:metric(data,preds[role]) for role in ORDER}
        if decision['fixed_k']!=old['decision_metrics']['classifier']:raise ValueError('C65 fixed classifier control differs')
        challenger=max(ORDER,key=lambda r:decision[r]['proxy_IoU']);decision_seal=folder/'decision_sealed.json';decision_seal.write_text(json.dumps({'threshold_choice':chosen,'decision_metrics':decision,'challenger':challenger},indent=2,allow_nan=False),encoding='utf-8')
        pred_records=[]
        def evaluate(part,prior,selected=None):
            report={'dates':{'assessment':old['dates'][part]},'assessment':[]}
            for item in old['source_records']:
                record=item['source_record']
                if record['phase']!=part:continue
                data,x=load(item,prior);score,baseline,keys,summary=predict(data,x);preds,sizes=decode(data,score,baseline,keys,summary,states,setting,old['choices']['classifier']['k'])
                legacy=next(r for r in old['score_records'] if r['phase']==part and r['seed']==record['seed'])
                if sha256_file(legacy['path'])!=legacy['sha256']:raise ValueError('Frozen control prediction changed')
                with np.load(legacy['path']) as a:
                    if not np.array_equal(preds['fixed_k'],a['classifier']) or not np.array_equal(score,a['classifier_score']):raise ValueError('Reused classifier or fixed decoder differs')
                if selected is not None:preds['selected']=preds[selected]
                for pred in preds.values():
                    use=data['condition']=='queue_ongoing'
                    if not np.array_equal(pred[use],baseline[use]):raise ValueError('Frozen ongoing changed')
                metrics={role:metric(data,pred) for role,pred in preds.items()};report['assessment'].append({'seed':record['seed'],'models':metrics})
                path=folder/(part+'_'+str(record['seed'])+'_predictions.npz');np.savez_compressed(path,**preds,classifier_score=score,summary_keys=keys,summary=summary,**{k+'_sizes':v for k,v in sizes.items()})
                pred_records.append({'phase':part,'seed':record['seed'],'path':path.as_posix(),'sha256':sha256_file(path)})
            return report
        selection=evaluate('selection',decision_seal);gate=comparison(selection,'incumbent',challenger);selected=challenger if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent'
        selection_seal=folder/'selection_sealed.json';selection_seal.write_text(json.dumps({'selected':selected,'challenger':challenger,'gate':gate},indent=2,allow_nan=False),encoding='utf-8')
        candidates={'incumbent':incumbent,'fixed_k':base,'threshold':OnsetSetDecoderModel(base,'threshold',setting=setting),
            **{role:OnsetSetDecoderModel(base,role,count_state=state) for role,state in states.items()}};candidates['decision_selected']=candidates[challenger];candidates['selected']=candidates[selected];exports={}
        for role,model in candidates.items():
            path=folder/(role+'.pkl');model.save(path);exports[role]={'path':path.as_posix(),'sha256':sha256_file(path),'candidate':challenger if role=='decision_selected' else selected if role=='selected' else role}
        assessment=evaluate('assessment',selection_seal,selected)
        cases.append({'source_study':old['source_study'],'dates':old['dates'],'feature_arm':old['feature_arm'],'training_windows':{'path':target_path.as_posix(),'sha256':sha256_file(target_path)},
            'count_states':states,'count_fit_seconds':count_seconds,'threshold_grid':grid,'threshold_choice':chosen,'decision_metrics':decision,'decision_selected':challenger,'selected':selected,'selection_gate':gate,
            'selection':selection['assessment'],'assessment':assessment['assessment'],'comparisons':{role:comparison(assessment,'incumbent',role) for role in ORDER+['selected']},
            'source_records':records,'phase_read_order':reads,'prediction_records':pred_records,'exports':exports,
            'seals':{name:sha256_file(folder/(name+'_sealed.json')) for name in ['count_models','decision','selection']}})
        (args.output/'progress.json').write_text(json.dumps({'cases':cases},indent=2,allow_nan=False),encoding='utf-8');print('onset_set_case_COMPLETE',token,'decision',challenger,'selected',selected,flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'reused_raw_input_window_verification':2345,'new_raw_input_rebuild_in_this_experiment':0,
        'n_frozen_classifier_booster_calls':calls,'seconds':time.perf_counter()-start,'source':evidence(),'formal_Queue':None,'formal_Total':None,
        'limits':'Same frozen classifier and public already-viewed proxy dates. Count labels use future observations only in fit;MCAR correction not verified or latent truth. Twelve fitted count states,including4intercepts;8newRidge models. No oracle input,formal promotion or new full submission.'}
    raw=json.dumps(result,indent=2,allow_nan=False);report.write_text(raw,encoding='utf-8');(args.output/'experiment.json').write_text(raw,encoding='utf-8');print('onset_set_COMPLETE',len(cases),'cases','seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
