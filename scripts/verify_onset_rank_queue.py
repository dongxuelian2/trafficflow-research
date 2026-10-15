"""Fresh public history/teacher replay and independent full-domain decisions."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import Booster
from trafficflow.data import Release,eligible
from trafficflow.experiments import partition_date
from trafficflow.causal_queue import released_origin_rows
from trafficflow.queue import QueueModel,queue_features
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol


def manual_baseline(data,probability,setting):
    if setting['timing']!='before':raise ValueError('Expected pre-threshold onset timing')
    p=np.zeros(len(probability),int)
    for w in np.unique(data['window']):
        ids=np.flatnonzero(data['window']==w);condition=str(data['condition'][ids[0]]);allowed=ids if condition=='queue_ongoing' else ids[data['lead_minutes'][ids]==30]
        threshold=setting['thresholds'].get(condition,setting['thresholds']['global']);p[allowed]=(probability[allowed]>=threshold)
        if setting['require_nonempty'] and not p[ids].any():p[allowed[np.argmax(probability[allowed])]]=1
    return p


def manual_topk(data,score,k,base):
    p=base.copy();links=np.tile(data['links'],len(data['ongoing'])*6)
    for w in np.unique(data['window']):
        ids=np.flatnonzero(data['window']==w)
        if data['condition'][ids[0]]!='queue_onset':continue
        p[ids]=0;allowed=ids[data['lead_minutes'][ids]==30];order=sorted(allowed,key=lambda i:(-score[i],str(links[i])))
        p[order[:min(k,len(order))]]=1
    return p


def manual_metric(data,prediction):
    rows=[];y=data['labels'].ravel().astype(bool);mask=data['available'].ravel();p=prediction.astype(bool)
    for w in np.unique(data['window']):
        ids=np.flatnonzero((data['window']==w)&mask);condition=str(data['condition'][data['window']==w][0]);union=np.count_nonzero(y[ids]|p[ids])
        value=np.count_nonzero(y[ids]&p[ids])/union if union else float(len(ids)>0)
        rows.append({'window':str(w),'condition':condition,'IoU':float(value)})
    means=pd.DataFrame(rows).groupby('condition').IoU.mean().to_dict()
    return {'proxy_IoU':float(np.mean(list(means.values()))),'conditions':means,'windows':rows}


def same_metric(a,b):
    if abs(a['proxy_IoU']-b['proxy_IoU'])>1e-12 or a['conditions'].keys()!=b['conditions'].keys() or any(abs(a['conditions'][k]-b['conditions'][k])>1e-12 for k in a['conditions']):raise ValueError('Condition metric differs')
    if a['windows']!=b['windows']:raise ValueError('Per-window IoU differs')


def paired(rows,dates,baseline,candidate):
    records=[]
    for run in rows:
        other={r['window']:r for r in run['models'][candidate]['windows']}
        for r in run['models'][baseline]['windows']:
            origin=r['window'].rsplit(':',1)[0];records.append({'origin':origin,'date':pd.Timestamp(origin.split(':',1)[1]).date().isoformat(),
                'condition':r['condition'],'a':r['IoU'],'b':other[r['window']]['IoU']})
    frame=pd.DataFrame(records)
    for _,group in frame.groupby('origin'):
        if group.condition.nunique()!=1 or not np.allclose(group[['a','b']],group[['a','b']].iloc[0],atol=1e-12,rtol=0):raise ValueError('Repeated origins disagree')
    unique=frame.drop_duplicates('origin');conditions=sorted(unique.condition.unique());count=np.zeros((len(dates),len(conditions)));delta=np.zeros_like(count)
    for r in unique.itertuples():
        i,j=dates.index(r.date),conditions.index(r.condition);count[i,j]+=1;delta[i,j]+=r.b-r.a
    gain=float((unique.groupby('condition').b.mean()-unique.groupby('condition').a.mean()).mean());rng=np.random.default_rng(20261013)
    starts=rng.integers(0,len(dates),(4000,int(np.ceil(len(dates)/7))));ix=((starts[:,:,None]+np.arange(7))%len(dates)).reshape(4000,-1)[:,:len(dates)]
    counts=count[ix].sum(axis=1);valid=(counts>0).all(axis=1);samples=(delta[ix].sum(axis=1)[valid]/counts[valid]).mean(axis=1)
    return {'paired_macro_gain':gain,'ci95':np.quantile(samples,[.025,.975]).tolist() if len(samples) else None,'valid_resamples':int(valid.sum()),'n_unique_origins':len(unique)}


def same_paired(a,b):
    if abs(a['paired_macro_gain']-b['paired_macro_gain'])>1e-12 or a['valid_resamples']!=b['valid_resamples'] or a['n_unique_origins']!=b['n_unique_origins'] or not np.allclose(a['ci95'],b['ci95'],atol=1e-12,rtol=0):raise ValueError('Independent day-block comparison differs')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite onset rank verification')
    study=json.loads(args.study.read_text());source=Path('artifacts/research')/args.study.stem
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Fresh_all_public_inputs_and_onset_query_ranking_deployment_verification',
        'source_study_sha256':sha256_file(args.study),'scope':'Allfit/decision/selection/assessment cached tree features and future evaluator labels rebuilt from public TRAIN. Independent query rows,topk/grid/ongoing/scoring/gates/day-block intervals. Portable interfaces on first/last of each condition per evaluation cache plus shuffled targets;not everywindow interface or independent tree refits.',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public');cases=[];source_hashes={}
    total_windows=booster_calls=portable_calls=shuffle_calls=poison_checks=0
    for case in study['cases']:
        old_path=Path('reports/research')/(case['source_study']+'.json')
        if sha256_file(old_path)!=case['source_study_sha256']:raise ValueError('Prior study changed')
        old=json.loads(old_path.read_text());panel=old['registered']['protocol']['panel'];token=case['source_study'].split('_',1)[0];model_folder=source/token
        for key,digest in case['seals'].items():
            if sha256_file(model_folder/(key+'_sealed.json'))!=digest:raise ValueError('Choice seal changed')
        for row in case['phase_read_order']:
            required={'decision':'models','selection':'decision','assessment':'selection'}.get(row['phase'])
            if row['prior_seal_sha256']!=(case['seals'][required] if required else None):raise ValueError('Phase read order differs')
        date_groups=[set(v) for v in old['dates'].values()]
        if any(a&b for i,a in enumerate(date_groups) for b in date_groups[i+1:]):raise ValueError('Time mechanisms overlap')
        incumbent=QueueModel.load(Path(case['incumbent']['path']));net,topology=release.network(panel),release.topology(panel);models={kind:[] for kind in ['classifier','ranker']};portable={}
        for t in case['training']:
            if sha256_file(t['model']['path'])!=t['model']['sha256']:raise ValueError('Model changed')
            model=Booster(model_file=t['model']['path'])
            if model.num_trees()!=t['trees'] or t['trees']>600 or model.params['num_leaves']!=63:raise ValueError('Model capacity differs')
            models[t['kind']].append(model)
        for role,record in case['exports'].items():
            if sha256_file(record['path'])!=record['sha256']:raise ValueError('Portable wrapper changed')
            portable[role]=QueueModel.load(Path(record['path']))
        raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};masked_paths={partition_date(release,p):p for p in release.files(panel,'train')}
        @lru_cache(maxsize=8)
        def raw_day(date):
            path=raw_paths[date];source_hashes[path.as_posix()]=sha256_file(path);return release.read(path)
        evaluated={p:[] for p in ['selection','assessment']};record_checks=[];local_poison=False
        for item in case['source_records']:
            record=item['source_record'];feature_file=item['used_feature_file'];phase=record['phase']
            if sha256_file(record['path'])!=record['sha256'] or sha256_file(feature_file['path'])!=feature_file['sha256']:raise ValueError('Original cached input changed')
            data=dict(np.load(record['path']));features=pd.read_parquet(feature_file['path']);links=data['links'].tolist();width=6*len(links)
            sample_ids=set()
            if phase!='fit':
                for condition in [False,True]:
                    ids=np.flatnonzero(data['ongoing']==condition)
                    if len(ids):sample_ids.update([int(ids[0]),int(ids[-1])])
            base_probability=incumbent.baseline.probability(features);base=manual_baseline(data,base_probability,incumbent.setting)
            scores={kind:np.mean([model.predict(features,num_threads=4) for model in members],axis=0) for kind,members in models.items()};booster_calls+=4
            preds={'incumbent':base,**{kind:manual_topk(data,score,case['choices'][kind]['k'],base) for kind,score in scores.items()}}
            if phase=='assessment':preds['selected']=preds[case['selected']]
            if phase=='fit':
                use=(data['condition']=='queue_onset')&(data['lead_minutes']==30)&data['available'].ravel();ix=np.flatnonzero(use);codes,queries=pd.factorize(data['window'][use],sort=False)
                with np.load(case['training_rows']['path']) as a:
                    if sha256_file(case['training_rows']['path'])!=case['training_rows']['sha256'] or not np.array_equal(a['source_flat_indices'],ix) or not np.array_equal(a['labels'],data['labels'].ravel()[use].astype(int)) or not np.array_equal(a['groups'],np.bincount(codes)) or not np.array_equal(a['query_windows'],queries) or not (a['weights']==1).all():raise ValueError('Onset query population differs')
                training_x=pd.read_parquet(case['training_table']['path'])
                if sha256_file(case['training_table']['path'])!=case['training_table']['sha256'] or not training_x.equals(features.loc[use].reset_index(drop=True)):raise ValueError('Training feature table differs')
            if phase=='decision':
                decision_metrics={'incumbent':manual_metric(data,base)}
                for kind in scores:
                    grid=[]
                    for entry in case['decision_grids'][kind]:
                        measured=manual_metric(data,manual_topk(data,scores[kind],entry['k'],base));same_metric(measured,entry['metrics']);grid.append(entry)
                    choice=max(grid,key=lambda r:(r['metrics']['proxy_IoU'],-r['k']))
                    if choice!=case['choices'][kind]:raise ValueError('Topk choice differs')
                    decision_metrics[kind]=choice['metrics']
                if max(['incumbent','classifier','ranker'],key=lambda r:decision_metrics[r]['proxy_IoU'])!=case['decision_selected']:raise ValueError('Decision winner differs')
                for role,m in decision_metrics.items():same_metric(m,case['decision_metrics'][role])
            if phase in ['selection','assessment']:
                saved=next(r for r in case['score_records'] if r['phase']==phase and r['seed']==record['seed'])
                if sha256_file(saved['path'])!=saved['sha256']:raise ValueError('Stored prediction changed')
                with np.load(saved['path']) as a:
                    if any(not np.array_equal(a[kind+'_score'],score) for kind,score in scores.items()) or any(not np.array_equal(a[role],pred) for role,pred in preds.items()):raise ValueError('Independent raw score or decision differs')
                metrics={role:manual_metric(data,pred) for role,pred in preds.items()};wanted=next(r for r in case[phase] if r['seed']==record['seed'])
                for role,m in metrics.items():same_metric(m,wanted['models'][role])
                evaluated[phase].append({'seed':record['seed'],'models':metrics})
            for i,origin_text in enumerate(data['origins']):
                origin=pd.Timestamp(str(origin_text));date=origin.date().isoformat();raw=raw_day(date);ts=pd.to_datetime(raw.timestamp,utc=True)
                history=raw.loc[(ts>=origin-pd.Timedelta(minutes=60))&(ts<origin)].copy();history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
                if masked_paths[date].as_posix() not in source_hashes:source_hashes[masked_paths[date].as_posix()]=sha256_file(masked_paths[date])
                condition='queue_ongoing' if data['ongoing'][i] else 'queue_onset';times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
                target=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id']).to_frame(index=False);offset=slice(i*width,(i+1)*width)
                rebuilt=queue_features(history,target,net,topology,origin,allow_origin=True,include_occupancy=case['feature_arm']=='rich')
                if case['feature_arm']=='rich':rebuilt['condition_ongoing']=float(condition=='queue_ongoing')
                expected=features.iloc[offset].reset_index(drop=True)
                if list(rebuilt.columns)!=list(expected.columns) or not np.array_equal(rebuilt.to_numpy(),expected.to_numpy(),equal_nan=True):raise ValueError('Fresh legal tree features differ')
                future=raw.loc[raw.timestamp.isin(times)].copy();future.loc[~eligible(future),'speed_kmh']=np.nan
                v=future.pivot_table(index='timestamp',columns='link_id',values='speed_kmh').reindex(index=times,columns=links).to_numpy(float)
                if not np.array_equal(data['labels'][i],(v<=net.v_cut.reindex(links).to_numpy(float)).astype(np.float32)) or not np.array_equal(data['available'][i],np.isfinite(v)):raise ValueError('Fresh public observation labels/domain differ')
                if i in sample_ids:
                    for role,model in portable.items():
                        candidate=case['exports'][role]['candidate'];wanted=preds[candidate][offset]
                        out=model.predict_window(history,target,net,topology,origin,condition);portable_calls+=1
                        if not out[['timestamp','link_id']].equals(target) or not np.array_equal(out.queue_pred,wanted):raise ValueError('Portable model differs')
                        if i==min(sample_ids):
                            order=np.random.default_rng(81).permutation(width);shuffled=model.predict_window(history,target.iloc[order],net,topology,origin,condition);shuffle_calls+=1
                            if not np.array_equal(shuffled.queue_pred,wanted[order]):raise ValueError('Target permutation changes prediction')
                    if condition=='queue_onset' and not local_poison:
                        poisoned=pd.concat([history,history.iloc[[0]].assign(timestamp=times[0])],ignore_index=True)
                        for role in ['classifier','ranker']:
                            try:portable[role].predict_window(poisoned,target,net,topology,origin,condition)
                            except ValueError:poison_checks+=1
                            else:raise ValueError('Future history was accepted')
                        local_poison=True
                total_windows+=1
            record_checks.append({'phase':phase,'seed':record['seed'],'n_fresh_windows':len(data['origins']),'tree_features_observation_labels_domain_exact':True})
            print('onset_rank_verified_cache',token,phase,record['seed'],'windows',len(data['origins']),flush=True)
        gate=paired(evaluated['selection'],case['dates']['selection'],'incumbent',case['decision_selected']);same_paired(gate,case['selection_gate'])
        selected=case['decision_selected'] if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent'
        if selected!=case['selected']:raise ValueError('Later selection gate differs')
        for role in ['classifier','ranker','selected']:same_paired(paired(evaluated['assessment'],case['dates']['assessment'],'incumbent',role),case['comparisons'][role])
        cases.append({'source_study':case['source_study'],'all_query_rows_and_groups_exact':True,'all_decision_grids_and_gates_exact':True,'all_ongoing_decisions_unchanged':True,'record_checks':record_checks})
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'n_fresh_public_window_occurrences':total_windows,'n_direct_booster_calls':booster_calls,
        'n_portable_window_calls':portable_calls,'n_additional_shuffled_target_calls':shuffle_calls,'n_future_history_poison_rejections':poison_checks,
        'source_files_sha256':source_hashes,'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,
        'scope':'Allcached populations/labels/scoring/raw learner matrices verified;portable interface tested on first/last of each condition per evaluation cache,not everywindow or independent tree training. Repeated origins/roles are not independent events.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8')
    print('onset_rank_verified',total_windows,'fresh_window_occurrences','portable_calls',portable_calls,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
