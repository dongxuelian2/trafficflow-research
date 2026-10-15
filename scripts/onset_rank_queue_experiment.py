"""Four public-train folds: matched onset classifiers and LambdaRank queries."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,LGBMRanker,Booster
from trafficflow.queue import QueueModel
from trafficflow.onset_rank_queue import onset_topk_decisions,OnsetTopKQueueModel
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.queue_domain import score_full_forecast
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from compare_queue_events import comparison


SOURCES=['C55_I5_S_autumn_five_phase_balanced_v2','C56_I5_S_winter_five_phase_balanced_v2','C57_I10_W_autumn_five_phase_balanced_v2','C58_I10_W_winter_five_phase_balanced_v2']
KGRID=[1,2,4,8,16]


def metric(data,prediction):
    return score_full_forecast(data['labels'].ravel(),prediction,data['window'],data['condition'],data['available'].ravel())


def apply_onset(data,score,k,baseline):
    prediction=baseline.copy();use=data['condition']=='queue_onset';links=np.tile(data['links'],len(data['ongoing'])*6)
    prediction[use]=onset_topk_decisions(score[use],data['window'][use],data['lead_minutes'][use],links[use],k)
    return prediction


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();report=Path('reports/research')/(args.output.name+'.json')
    if report.exists() or args.output.exists():raise FileExistsError('Never overwrite onset ranking evidence')
    parent_path=Path('reports/research/model_artifacts_checkpoint_queue_latent_v1.json');known={r['path']:r['sha256'] for r in json.loads(parent_path.read_text())['artifacts']}
    sources={name:json.loads((Path('reports/research')/(name+'.json')).read_text()) for name in SOURCES};seeds=[20261013,20261029]
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Public_train_onset_query_ranking_with_fixed_ongoing_reference','sources':SOURCES,
        'source_reports_sha256':{name:sha256_file(Path('reports/research')/(name+'.json')) for name in SOURCES},'source_manifest_sha256':sha256_file(parent_path),
        'seeds':seeds,'k_grid':KGRID,'new_models_per_case':4,
        'matched_data':'Identical existing fit onset T+30 eligible rows,labels,feature columns and unit weights;complete-window query groups for LambdaRank. No future eligibility in prediction.',
        'features':'Same rich/plain feature schema as the frozen decision-selected existing tree;no additional features or labels',
        'models':'Both600maxTrees/63leaves/lr.035/min_child60/lambda8/feature_fraction.9/4threads. Classifier binary logloss;ranker lambdarank truncation20 normtrue label_gain0,1. Two-seed equal mean;ranking scores are not probabilities.',
        'decision':'Separate decision period chooses top-k for each method;max overall proxy among incumbent/classifier/ranker,prefer incumbent then classifier on exact tie',
        'selection':'Later15days:chosen challenger replaces incumbent only if origin-deduplicated macro gain>=.01 and paired7day-block95CI lower>0',
        'ongoing':'Every candidate exactly copies all existing ongoing decisions. Only onset final step can change.',
        'timing':'Fixed tree iteration cap,no weight checkpoint selection. Existing checkpoint interval unused buffer. Fit->model seal->decision->decision seal->selection->selection seal->assessment.',
        'assessment':'Previously inspected public TRAIN development dates and observation labels;not latent truth or final/private. Source reports include historical outcomes;no new choices use assessment arrays.',
        'algorithm_sources':['https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html','https://lightgbm.readthedocs.io/en/latest/Parameters.html#lambdarank_truncation_level'],
        'formal_Queue':None,'formal_Total':None,'uploaded':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter();results=[]
    for name,source in sources.items():
        token=name.split('_',1)[0];folder=args.output/token;folder.mkdir();reference_record=source['exported_models']['incumbent'];reference_path=Path(reference_record['path'])
        if sha256_file(reference_path)!=reference_record['sha256'] or known[reference_path.as_posix()]!=reference_record['sha256']:raise ValueError('Frozen strong reference changed')
        incumbent=QueueModel.load(reference_path)
        if incumbent.alpha!=0:raise ValueError('CPU tree reference required')
        arm=source['decision_choices'][source['incumbent']]['baseline_arm'];records=[];reads=[];training=[];models={};model_files={}
        def load(record,prior=None):
            if record['phase']!='fit' and prior is None:raise ValueError('Prior model choice seal required')
            feature_record=next(r for r in record['feature_files'] if Path(r['path']).stem.endswith('_'+arm))
            for row in [record,feature_record]:
                if sha256_file(row['path'])!=row['sha256'] or known[row['path']]!=row['sha256']:raise ValueError('Source cached population changed')
            data=dict(np.load(record['path']));features=pd.read_parquet(feature_record['path'])
            if list(features.columns)!=incumbent.baseline.model.feature_name_ or len(features)!=len(data['labels'].ravel()):raise ValueError('Matched feature schema differs')
            reads.append({'phase':record['phase'],'seed':record['seed'],'prior_seal_sha256':sha256_file(prior) if prior else None})
            records.append({'source_record':record,'used_feature_file':feature_record});return data,features
        fit_record=next(r for r in source['cache_records'] if r['phase']=='fit');fit,features=load(fit_record)
        use=(fit['condition']=='queue_onset')&(fit['lead_minutes']==30)&fit['available'].ravel();indices=np.flatnonzero(use)
        windows=fit['window'][use];codes,queries=pd.factorize(windows,sort=False);groups=np.bincount(codes)
        if (np.diff(codes)<0).any():raise ValueError('Query rows are not contiguous')
        x=features.loc[use].reset_index(drop=True);y=fit['labels'].ravel()[use].astype(int)
        for group in range(len(groups)):
            if set(y[codes==group])!={0,1}:raise ValueError('Onset query requires positive and negative eligible labels')
        xpath=folder/'training_features.parquet';x.to_parquet(xpath,index=False);rpath=folder/'training_rows.npz';np.savez_compressed(rpath,source_flat_indices=indices,labels=y,groups=groups,query_windows=queries,weights=np.ones(len(y)))
        for kind,cls in [('classifier',LGBMClassifier),('ranker',LGBMRanker)]:
            models[kind]=[];model_files[kind]=[]
            for seed in seeds:
                params=dict(n_estimators=600,num_leaves=63,learning_rate=.035,min_child_samples=60,reg_lambda=8.,colsample_bytree=.9,n_jobs=4,
                    random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True)
                if kind=='ranker':params.update(objective='lambdarank',lambdarank_truncation_level=20,lambdarank_norm=True,label_gain=[0,1])
                tic=time.perf_counter();model=cls(**params)
                if kind=='ranker':model.fit(x,y,group=groups,sample_weight=np.ones(len(y)))
                else:model.fit(x,y,sample_weight=np.ones(len(y)))
                path=folder/(kind+'_'+str(seed)+'.txt');model.booster_.save_model(path);models[kind].append(model.booster_);model_files[kind].append({'path':path.as_posix(),'sha256':sha256_file(path)})
                training.append({'kind':kind,'seed':seed,'rows':len(x),'queries':len(groups),'trees':model.booster_.num_trees(),'seconds':time.perf_counter()-tic,'model':model_files[kind][-1]})
                print('onset_rank_fit',token,kind,seed,'rows',len(x),'queries',len(groups),flush=True)
        model_seal=folder/'models_sealed.json';model_seal.write_text(json.dumps(model_files,indent=2),encoding='utf-8')
        def predict(data,x):
            p=incumbent.baseline.probability(x);baseline=full_queue_decisions(p,data['window'],data['condition'],data['lead_minutes'],incumbent.setting['thresholds'],incumbent.setting['require_nonempty'],incumbent.setting['timing'])
            scores={kind:np.mean([model.predict(x,num_threads=4) for model in members],axis=0) for kind,members in models.items()}
            return baseline,scores
        decision_record=next(r for r in source['cache_records'] if r['phase']=='decision');data,x=load(decision_record,model_seal);baseline,scores=predict(data,x);grids={};choices={}
        base_metric=metric(data,baseline)
        if base_metric!=source['decision_choices'][source['incumbent']]['metrics']:raise ValueError('Original decision reference does not reproduce')
        for kind,score in scores.items():
            grids[kind]=[{'k':k,'metrics':metric(data,apply_onset(data,score,k,baseline))} for k in KGRID]
            choices[kind]=max(grids[kind],key=lambda r:(r['metrics']['proxy_IoU'],-r['k']))
        decision_metrics={'incumbent':base_metric,**{kind:choice['metrics'] for kind,choice in choices.items()}}
        order=['incumbent','classifier','ranker'];challenger=max(order,key=lambda r:decision_metrics[r]['proxy_IoU'])
        decision_seal=folder/'decision_sealed.json';decision_seal.write_text(json.dumps({'choices':choices,'challenger':challenger,'metrics':decision_metrics},indent=2,allow_nan=False),encoding='utf-8')
        prediction_records=[]
        def evaluate(part,seal,selected=None):
            result={'dates':{'assessment':source['dates'][part]},'assessment':[]}
            for record in source['cache_records']:
                if record['phase']!=part:continue
                data,x=load(record,seal);baseline,scores=predict(data,x);preds={'incumbent':baseline,**{kind:apply_onset(data,score,choices[kind]['k'],baseline) for kind,score in scores.items()}}
                if selected is not None:preds['selected']=preds[selected]
                for pred in preds.values():
                    if not np.array_equal(pred[data['condition']=='queue_ongoing'],baseline[data['condition']=='queue_ongoing']):raise ValueError('Ongoing reference changed')
                values={role:metric(data,pred) for role,pred in preds.items()};result['assessment'].append({'seed':record['seed'],'models':values})
                path=folder/(part+'_'+str(record['seed'])+'_predictions.npz');np.savez_compressed(path,**preds,**{kind+'_score':p for kind,p in scores.items()})
                prediction_records.append({'phase':part,'seed':record['seed'],'path':path.as_posix(),'sha256':sha256_file(path)})
            return result
        selection=evaluate('selection',decision_seal);gate=comparison(selection,'incumbent',challenger)
        selected=challenger if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent'
        selection_seal=folder/'selection_sealed.json';selection_seal.write_text(json.dumps({'selected':selected,'decision_challenger':challenger,'gate':gate},indent=2,allow_nan=False),encoding='utf-8')
        exported={}
        candidates={'incumbent':incumbent}
        for kind in ['classifier','ranker']:candidates[kind]=OnsetTopKQueueModel(incumbent,[m.model_to_string() for m in models[kind]],features.columns.tolist(),choices[kind]['k'],kind)
        candidates['decision_selected']=candidates[challenger];candidates['selected']=candidates[selected]
        for role,model in candidates.items():
            path=folder/(role+'.pkl');model.save(path);exported[role]={'path':path.as_posix(),'sha256':sha256_file(path),'candidate':challenger if role=='decision_selected' else selected if role=='selected' else role}
        assessment=evaluate('assessment',selection_seal,selected)
        results.append({'source_study':name,'source_study_sha256':registered['protocol']['source_reports_sha256'][name],'dates':source['dates'],'feature_arm':arm,
            'training':training,'training_table':{'path':xpath.as_posix(),'sha256':sha256_file(xpath)},'training_rows':{'path':rpath.as_posix(),'sha256':sha256_file(rpath)},
            'incumbent':reference_record,'choices':choices,'decision_grids':grids,'decision_metrics':decision_metrics,'decision_selected':challenger,'selected':selected,'selection_gate':gate,
            'selection':selection['assessment'],'assessment':assessment['assessment'],'comparisons':{role:comparison(assessment,'incumbent',role) for role in ['classifier','ranker','selected']},
            'score_records':prediction_records,'source_records':records,'phase_read_order':reads,'exports':exported,
            'seals':{key:sha256_file(folder/(key+'_sealed.json')) for key in ['models','decision','selection']}})
        (args.output/'progress.json').write_text(json.dumps({'cases':results},indent=2,allow_nan=False),encoding='utf-8')
        print('onset_rank_case_COMPLETE',token,'decision',challenger,'selected',selected,flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':results,'seconds':time.perf_counter()-start,'source':evidence(),'formal_Queue':None,'formal_Total':None,
        'limits':'Only public noisy proxy labels and already-inspected dates. Dedicated onset model/ranking/set decoding;ongoing frozen. Classifier-vs-ranker matches row/schema/budget,not internal gradients or query normalization. No formal promotion or new complete submission.'}
    raw=json.dumps(result,indent=2,allow_nan=False);report.write_text(raw,encoding='utf-8');(args.output/'experiment.json').write_text(raw,encoding='utf-8')
    print('onset_rank_COMPLETE','cases',len(results),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
