"""Independent cardinality summaries,augmented Ridge,decisions and deployment."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import Booster
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.causal_queue import released_origin_rows
from trafficflow.queue import QueueModel
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol
from verify_onset_rank_queue import manual_baseline,manual_metric,same_metric,paired,same_paired


def manual_summary(data,x):
    keys=[];rows=[];links=np.tile(data['links'],len(data['ongoing'])*6)
    for key in pd.unique(data['window']):
        ids=np.flatnonzero(data['window']==key)
        if data['condition'][ids[0]]!='queue_onset':continue
        ids=ids[data['lead_minutes'][ids]==30];ids=ids[np.argsort(links[ids].astype(str),kind='stable')];a=x.iloc[ids]
        gap=a.current.astype(float)-a.cut_ratio.astype(float);trend=a.trend_forecast.astype(float)-a.cut_ratio.astype(float);flow=a.flow_capacity_ratio.astype(float)
        def get(s,fn):
            s=s.replace([np.inf,-np.inf],np.nan).dropna();return float(fn(s)) if len(s) else 0.
        row=[len(a),get(gap,lambda s:s.min()),get(gap,lambda s:s.quantile(.1)),get(gap,lambda s:s.mean()),get(gap,lambda s:s.std(ddof=0)),float((gap<=0).mean()),
            get(trend,lambda s:s.min()),get(trend,lambda s:s.quantile(.1)),get(trend,lambda s:s.mean()),get(trend,lambda s:s.std(ddof=0)),float((trend<=0).mean()),
            get(a.slope.astype(float),lambda s:s.min()),get(a.slope.astype(float),lambda s:s.mean()),get(a.slope.astype(float),lambda s:s.std(ddof=0)),
            get(flow,lambda s:s.mean()),get(flow,lambda s:s.max()),float((~np.isfinite(flow)).mean()),get(a.speed_std.astype(float),lambda s:s.mean()),get(a.speed_std.astype(float),lambda s:s.max()),
            get(a.coverage.astype(float),lambda s:s.mean()),get(a.age.astype(float),lambda s:s.mean()),float(a.sin_tod.iloc[0]),float(a.cos_tod.iloc[0]),float(a.weekday.iloc[0])]
        keys.append(str(key));rows.append(row)
    return np.array(keys),np.array(rows,float)


def recompute_state(summary,y,m,n,adjusted,prior):
    mu=np.mean(summary,axis=0);scale=np.std(summary,axis=0);scale[scale<1e-12]=1.;z=np.log1p(y*n/m if adjusted else y)
    if prior:beta=np.r_[z.mean(),np.zeros(24)]
    else:
        d=np.column_stack([np.ones(len(summary)),(summary-mu)/scale]);penalty=np.column_stack([np.zeros(24),np.sqrt(10.)*np.eye(24)])
        beta=np.linalg.lstsq(np.vstack([d,penalty]),np.r_[z,np.zeros(24)],rcond=None)[0]
    return mu,scale,beta


def counts(state,summary):
    z=((summary-np.array(state['mean']))/np.array(state['scale']))@np.array(state['coefficient'])+state['intercept']
    return np.minimum(np.maximum(np.floor(np.expm1(np.clip(z,0,np.log1p(summary[:,0])))+.5),1),summary[:,0]).astype(int)


def manual_decode(data,score,base,keys,summary,states,setting,fixed_k):
    predictions={'incumbent':base.copy()};sizes={role:dict(zip(keys,counts(state,summary))) for role,state in states.items()};sizes['fixed_k']={k:fixed_k for k in keys}
    links=np.tile(data['links'],len(data['ongoing'])*6)
    for role in ['fixed_k','count_prior','threshold','count_observed','count_adjusted']:
        pred=base.copy()
        for key in keys:
            ids=np.flatnonzero(data['window']==key);allowed=ids[data['lead_minutes'][ids]==30];pred[ids]=0;order=sorted(allowed,key=lambda i:(-score[i],str(links[i])))
            if role=='threshold':
                pred[allowed]=(score[allowed]>=setting['threshold'])
                if setting['force'] and not pred[ids].any():pred[order[0]]=1
            else:pred[order[:int(sizes[role][key])]]=1
        predictions[role]=pred
    return predictions,sizes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite onset decoder verification')
    study=json.loads(args.study.read_text());cfg=study['registered']['protocol'];old_path=Path('reports/research/C65_onset_query_rank_public.json');old=json.loads(old_path.read_text());proof_path=Path(cfg['reused_input_proof']);proof=json.loads(proof_path.read_text())
    if sha256_file(old_path)!=cfg['source_study_sha256'] or sha256_file(proof_path)!=cfg['reused_input_proof_sha256']:raise ValueError('Reused frozen data audit changed')
    for path,digest in cfg['unchanged_source_constructors_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Audited legal input constructor changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_fit_only_cardinality_and_frozen_decoder_verification',
        'source_study_sha256':sha256_file(args.study),'reused_raw_input_audit_sha256':sha256_file(proof_path),'raw_input_window_occurrences_previously_verified':2345,
        'scope':'Reuses hash-bound C66 raw input/teacher audit,does not rebuild2345rawwindows again. Independently rebuilds all new onset summaries/fit targets,Ridge via augmented least squares,threshold grids,decisions,scoring/gates. Fresh public histories for portable first/last eachcondition per eval cache.',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public');source=Path('artifacts/research')/args.study.stem
    max_summary=max_state=0.;summary_windows=portable_calls=shuffle_calls=poison_reject=booster_calls=fresh_histories=0;records=[]
    for case,old_case in zip(study['cases'],old['cases']):
        if case['source_study']!=old_case['source_study']:raise ValueError('Case order differs')
        token=case['source_study'].split('_',1)[0];part_folder=source/token;original=json.loads((Path('reports/research')/(case['source_study']+'.json')).read_text());panel=original['registered']['protocol']['panel']
        for name,digest in case['seals'].items():
            if sha256_file(part_folder/(name+'_sealed.json'))!=digest:raise ValueError('New choice seal changed')
        for r in case['phase_read_order']:
            requirement={'decision':'count_models','selection':'decision','assessment':'selection'}.get(r['phase'])
            if r['prior_seal_sha256']!=(case['seals'][requirement] if requirement else None):raise ValueError('Choice read order differs')
        incumbent=QueueModel.load(Path(old_case['incumbent']['path']));boosters=[]
        for r in old_case['training']:
            if r['kind']=='classifier':
                if sha256_file(r['model']['path'])!=r['model']['sha256']:raise ValueError('Frozen classifier changed')
                boosters.append(Booster(model_file=r['model']['path']))
        portable={}
        for role,r in case['exports'].items():
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Portable decoder changed')
            portable[role]=QueueModel.load(Path(r['path']))
        net,topology=release.network(panel),release.topology(panel);raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};evaluated={phase:[] for phase in ['selection','assessment']};case_records=[];poisoned=False
        for item in case['source_records']:
            record=item['source_record'];feature=item['used_feature_file'];phase=record['phase']
            if sha256_file(record['path'])!=record['sha256'] or sha256_file(feature['path'])!=feature['sha256']:raise ValueError('Reused verified input cache changed')
            data=dict(np.load(record['path']));x=pd.read_parquet(feature['path']);keys,summary=manual_summary(data,x);summary_windows+=len(keys)
            if phase=='fit':
                row=case['training_windows']
                if sha256_file(row['path'])!=row['sha256']:raise ValueError('Training window summary changed')
                with np.load(row['path']) as saved:
                    error=float(np.abs(summary-saved['summary']).max());max_summary=max(max_summary,error)
                    if not np.array_equal(keys,saved['keys']) or error>1e-10:raise ValueError('Independent fit summary differs')
                    positive=[];eligible=[]
                    for key in keys:
                        ids=np.flatnonzero((data['window']==key)&(data['lead_minutes']==30)&data['available'].ravel());positive.append(data['labels'].ravel()[ids].sum());eligible.append(len(ids))
                    y,m=np.array(positive,float),np.array(eligible,float)
                    if not np.array_equal(y,saved['positive']) or not np.array_equal(m,saved['eligible']) or not np.array_equal(summary[:,0],saved['links']):raise ValueError('Fit count labels or coverage differs')
                for role,state in case['count_states'].items():
                    mu,scale,beta=recompute_state(summary,y,m,summary[:,0],role!='count_observed',role=='count_prior')
                    errors=[np.max(np.abs(mu-state['mean'])),np.max(np.abs(scale-state['scale'])),abs(beta[0]-state['intercept']),np.max(np.abs(beta[1:]-state['coefficient']))];max_state=max(max_state,float(max(errors)))
                    if max(errors)>1e-9 or state['alpha']!=10. or state['n_fit_windows']!=len(keys):raise ValueError('Independent augmented Ridge solution differs')
                case_records.append({'phase':phase,'seed':record['seed'],'onset_summary_windows':len(keys),'new_summary_and_fit_targets_verified':True});continue
            score=np.mean([b.predict(x,num_threads=4) for b in boosters],axis=0);booster_calls+=len(boosters);base=manual_baseline(data,incumbent.baseline.probability(x),incumbent.setting)
            setting=case['threshold_choice']['setting'];preds,sizes=manual_decode(data,score,base,keys,summary,case['count_states'],setting,old_case['choices']['classifier']['k'])
            if phase=='decision':
                for row in case['threshold_grid']:
                    # Threshold check is independent of count models or their labels.
                    pred=base.copy();links=np.tile(data['links'],len(data['ongoing'])*6)
                    for key in keys:
                        ids=np.flatnonzero(data['window']==key);allowed=ids[data['lead_minutes'][ids]==30];pred[ids]=0;pred[allowed]=(score[allowed]>=row['setting']['threshold'])
                        if row['setting']['force'] and not pred[ids].any():pred[sorted(allowed,key=lambda i:(-score[i],str(links[i])))[0]]=1
                    same_metric(manual_metric(data,pred),row['metrics'])
                choice=max(case['threshold_grid'],key=lambda r:(r['metrics']['proxy_IoU'],not r['setting']['force'],-abs(r['setting']['threshold']-.5)))
                if choice!=case['threshold_choice']:raise ValueError('Threshold choice differs')
                for role,m in case['decision_metrics'].items():same_metric(manual_metric(data,preds[role]),m)
                order=['incumbent','fixed_k','count_prior','threshold','count_observed','count_adjusted']
                if max(order,key=lambda r:case['decision_metrics'][r]['proxy_IoU'])!=case['decision_selected']:raise ValueError('Decision candidate choice differs')
            else:
                if phase=='assessment':preds['selected']=preds[case['selected']]
                row=next(r for r in case['prediction_records'] if r['phase']==phase and r['seed']==record['seed'])
                if sha256_file(row['path'])!=row['sha256']:raise ValueError('Saved decoder predictions changed')
                with np.load(row['path']) as saved:
                    error=float(np.abs(summary-saved['summary']).max());max_summary=max(max_summary,error)
                    if not np.array_equal(keys,saved['summary_keys']) or error>1e-10 or not np.array_equal(saved['classifier_score'],score):raise ValueError('Independent summary or frozen classifier differs')
                    for role,pred in preds.items():
                        if not np.array_equal(pred,saved[role]):raise ValueError('Independent decoder differs')
                    for role,state in case['count_states'].items():
                        if not np.array_equal(np.array([sizes[role][k] for k in keys]),saved[role+'_sizes']):raise ValueError('Count rounding differs')
                metrics={role:manual_metric(data,pred) for role,pred in preds.items()};wanted=next(r for r in case[phase] if r['seed']==record['seed'])
                for role,m in metrics.items():same_metric(m,wanted['models'][role])
                evaluated[phase].append({'seed':record['seed'],'models':metrics})
            sample=set();first_by_condition=set()
            for c in [False,True]:
                ids=np.flatnonzero(data['ongoing']==c)
                if len(ids):sample.update([int(ids[0]),int(ids[-1])]);first_by_condition.add(int(ids[0]))
            width=6*len(data['links'])
            for i in sorted(sample):
                origin=pd.Timestamp(str(data['origins'][i]));token_time=origin.strftime('%Y-%m-%dT%H:%M:%SZ');history=pd.read_parquet(raw_paths[origin.date().isoformat()],filters=[('timestamp','>=',(origin-pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ')),('timestamp','<',token_time)])
                for key in ['timestamp','link_id','station_id']:history[key]=history[key].astype(str)
                history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True);fresh_histories+=1
                condition='queue_ongoing' if data['ongoing'][i] else 'queue_onset';times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
                target=pd.MultiIndex.from_product([times,data['links']],names=['timestamp','link_id']).to_frame(index=False);offset=slice(i*width,(i+1)*width)
                for role,model in portable.items():
                    candidate=case['exports'][role]['candidate'];wanted=preds[candidate][offset];out=model.predict_window(history,target,net,topology,origin,condition);portable_calls+=1
                    if not out[['timestamp','link_id']].equals(target) or not np.array_equal(out.queue_pred,wanted):raise ValueError('Portable decoder differs from independent recipe')
                    if i in first_by_condition:
                        permutation=np.random.default_rng(91).permutation(width);out=model.predict_window(history,target.iloc[permutation],net,topology,origin,condition);shuffle_calls+=1
                        if not np.array_equal(out.queue_pred,wanted[permutation]):raise ValueError('Portable decoder depends on target order')
                if condition=='queue_onset' and not poisoned:
                    bad=pd.concat([history,history.iloc[[0]].assign(timestamp=times[0])],ignore_index=True)
                    for role in ['threshold','count_prior','count_observed','count_adjusted']:
                        try:portable[role].predict_window(bad,target,net,topology,origin,condition)
                        except ValueError:poison_reject+=1
                        else:raise ValueError('Future history accepted')
                    poisoned=True
            case_records.append({'phase':phase,'seed':record['seed'],'onset_summary_windows':len(keys),'new_summary_decisions_scores_verified':True})
        gate=paired(evaluated['selection'],case['dates']['selection'],'incumbent',case['decision_selected']);same_paired(gate,case['selection_gate'])
        selected=case['decision_selected'] if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent'
        if selected!=case['selected']:raise ValueError('Later selection gate differs')
        for role,comparison in case['comparisons'].items():same_paired(paired(evaluated['assessment'],case['dates']['assessment'],'incumbent',role),comparison)
        records.append({'source_study':case['source_study'],'new_summaries_fit_targets_ridge_decoders_gates_exact_within_tolerance':True,'records':case_records})
        print('onset_set_verified_case',case['source_study'],flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':records,'reused_prior_raw_window_verification':2345,'n_new_onset_summary_window_occurrences':summary_windows,
        'maximum_summary_difference':max_summary,'maximum_count_fit_parameter_difference':max_state,'n_direct_frozen_classifier_booster_calls':booster_calls,
        'n_new_fresh_public_history_windows':fresh_histories,'n_portable_role_window_calls':portable_calls,'n_additional_shuffled_target_calls':shuffle_calls,'n_future_history_poison_rejections':poison_reject,
        'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,'scope':'C66 raw full-cache proof explicitly reused;newderived count recipe independently reimplemented and portable interfaces freshly tested on sampled histories. No new classifier fit or hidden queue truth.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8')
    print('onset_set_verified','summary_windows',summary_windows,'portable',portable_calls,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
