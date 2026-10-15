"""Fresh rolling inputs and independent calibration/decoder/gate replay."""
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
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot
from verify_onset_rank_queue import manual_baseline,manual_topk,manual_metric,same_metric,paired,same_paired


def manual_threshold(a,score,base,setting):
    pred=base.copy();links=np.tile(a['links'],len(a['ongoing'])*6)
    for key in pd.unique(a['window']):
        ids=np.flatnonzero(a['window']==key)
        if a['condition'][ids[0]]!='queue_onset':continue
        pred[ids]=0;last=ids[a['lead_minutes'][ids]==30];pred[last]=score[last]>=setting['threshold']
        if setting['force'] and not pred[ids].any():pred[sorted(last,key=lambda i:(-score[i],str(links[i])))[0]]=1
    return pred


def independent_scalar_score(a,pred):
    """Integer set counts per window, NumPy mean order of calibration scorer."""
    y=a['labels'].ravel().astype(bool);m=a['available'].ravel();rows=[];conditions=[]
    for key in np.unique(a['window']):
        ids=np.flatnonzero((a['window']==key)&m);p=pred[ids].astype(bool);truth=y[ids];union=int(np.sum(p|truth))
        rows.append(float(np.sum(p&truth))/union if union else float(len(ids)>0));conditions.append(a['condition'][a['window']==key][0])
    rows=np.array(rows);conditions=np.array(conditions)
    return float(np.mean([rows[conditions==c].mean() for c in np.unique(conditions)]))


def verify_baseline_grid(a,p,calibration):
    generated=[];n=0
    for force in [False,True]:
        best={}
        for c in np.unique(a['condition']):
            use=a['condition']==c;part={k:a[k][use] for k in ['window','condition','lead_minutes']};part['labels']=a['labels'].ravel()[use];part['available']=a['available'].ravel()[use];values=[]
            for t in np.linspace(.05,.8,31):
                setting={'thresholds':{'global':float(t)},'require_nonempty':force,'timing':'before'}
                values.append((independent_scalar_score(part,manual_baseline(part,p[use],setting)),float(t)));n+=1
            best[str(c)]=max(values,key=lambda r:(r[0],-abs(r[1]-.5)))[1]
        for mode in ['global','condition']:
            candidates=[{'global':float(t)} for t in np.linspace(.05,.8,31)] if mode=='global' else [{'global':.5,**best}]
            for thresholds in candidates:
                s={'mode':mode,'thresholds':thresholds,'require_nonempty':force,'timing':'before'};s['calibration_proxy_IoU']=independent_scalar_score(a,manual_baseline(a,p,s));generated.append(s);n+=1
    if len(generated)!=len(calibration['settings']):raise ValueError('Baseline grid length differs')
    for left,right in zip(generated,calibration['settings']):
        if any(left[k]!=right[k] for k in ['mode','thresholds','require_nonempty','timing']) or abs(left['calibration_proxy_IoU']-right['calibration_proxy_IoU'])>1e-12:raise ValueError('Independent baseline calibration grid differs')
    selected=max(calibration['settings'],key=lambda r:(r['calibration_proxy_IoU'],r['mode']=='global',not r['require_nonempty']))
    if selected!=calibration['selected']:raise ValueError('Baseline calibration winner differs')
    return n


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite rolling verification')
    study=json.loads(args.study.read_text());registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_rolling_onset_decoder_verification','source_study_sha256':sha256_file(args.study),
        'scope':'Fresh raw/masked-T histories and both tree feature arms for ALL selected cached window occurrences;selected sampler guards,fit rows/model capacity,all baseline/threshold/k grids,gates/scores and sampled portable interfaces. No classifier retraining or independent recreation of candidate universe/RNG.',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public');source_hashes={};cases=[]
    totals={'raw_window_occurrences':0,'feature_arm_window_rebuilds':0,'direct_booster_calls':0,'portable_window_calls':0,'additional_shuffled_calls':0,'future_history_rejections':0,'baseline_calibration_score_calls':0,'new_training_models_checked':0}
    for case in study['cases']:
        base_folder=Path('artifacts/research')/args.study.stem/case['name'];panel=case['panel'];net,topology=release.network(panel),release.topology(panel)
        for part,digest in case['seals'].items():
            if sha256_file(base_folder/(part+'_sealed.json'))!=digest:raise ValueError('Choice seal changed')
        phase_sets=[set(d) for d in case['dates'].values()]
        if any(a&b for i,a in enumerate(phase_sets) for b in phase_sets[i+1:]):raise ValueError('Phase date overlap')
        for read in case['phase_reads']:
            required={'decision':'checkpoint','selection':'decision','assessment':'selection'}.get(read['phase'])
            if read['prior_seal_sha256']!=(case['seals'][required] if required else None):raise ValueError('Causal read order differs')
        if any(r['phase']=='checkpoint' for r in case['cache_records']):raise ValueError('Unused buffer was loaded')
        decision_seal=json.loads((base_folder/'decision_sealed.json').read_text());baseline_boosters={};onset_boosters={a:[] for a in ['plain','rich']}
        for record in case['training']:
            row=record['booster'] if record['role']=='baseline' else record['model']
            for item in [record['model'],row]:
                if sha256_file(item['path'])!=item['sha256']:raise ValueError('Fitted model changed')
            model=Booster(model_file=row['path']);totals['new_training_models_checked']+=1
            if model.num_trees()!=record['trees'] or not 0<record['trees']<=600 or model.params['num_leaves']!=63 or model.feature_name()!=record['features']:raise ValueError('Matched model capacity/schema differs')
            if record['role']=='baseline':baseline_boosters[record['arm']]=model
            else:onset_boosters[record['arm']].append(model)
        if len(case['training'])!=6 or any(len(v)!=2 for v in onset_boosters.values()):raise ValueError('Missing matched refit')
        portable={}
        for role,row in case['exports'].items():
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Portable model changed')
            portable[role]=QueueModel.load(Path(row['path']))
        raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};masked_paths={partition_date(release,p):p for p in release.files(panel,'train')}
        for phase,days in case['dates'].items():
            if phase=='checkpoint':continue
            for date in days:
                path=raw_paths[date]
                if path.as_posix() not in source_hashes:source_hashes[path.as_posix()]=sha256_file(path)
        for path in (release.root/'corridors'/panel/'network').glob('*.csv'):source_hashes[path.as_posix()]=sha256_file(path)
        @lru_cache(maxsize=8)
        def raw_day(date):
            path=raw_paths[date];source_hashes[path.as_posix()]=sha256_file(path);return release.read(path)
        evaluated={'selection':[],'assessment':[]};checks=[];poisoned=False
        for record in case['cache_records']:
            phase=record['phase'];seed=record['seed']
            if sha256_file(record['path'])!=record['sha256']:raise ValueError('Input cache changed')
            a=dict(np.load(record['path']));features={}
            for f in record['feature_files']:
                if sha256_file(f['path'])!=f['sha256']:raise ValueError('Cached features changed')
                features['rich' if Path(f['path']).stem.endswith('_rich') else 'plain']=pd.read_parquet(f['path'])
            links=a['links'].tolist();width=len(links)*6;arm=case['feature_arm'];y=a['labels'].ravel();eligible_rows=a['available'].ravel()
            if len(a['origins'])!=len(set(a['origins'])) or any(pd.Timestamp(str(t)).date().isoformat() not in case['dates'][phase] for t in a['origins']):raise ValueError('Origins outside phase or duplicated')
            spacing=30 if phase=='fit' else 360;ordered=pd.to_datetime(a['origins'],utc=True)
            if len(ordered)>1 and (ordered[1:]-ordered[:-1]).min()<pd.Timedelta(minutes=spacing):raise ValueError('Sampler spacing violated')
            pred=None;sample_ids=set();first_ids=set()
            if phase=='fit':
                mask=eligible_rows&(a['condition']=='queue_onset')&(a['lead_minutes']==30);row=case['training_rows']
                if sha256_file(row['path'])!=row['sha256']:raise ValueError('Training row file changed')
                with np.load(row['path']) as saved:
                    if not np.array_equal(saved['baseline_indices'],np.flatnonzero(eligible_rows)) or not np.array_equal(saved['onset_indices'],np.flatnonzero(mask)) or not np.array_equal(saved['onset_labels'],y[mask].astype(int)) or not (saved['onset_weights']==1).all():raise ValueError('Actual supervised rows differ')
                for r in case['training']:
                    if r['rows']!=int((eligible_rows if r['role']=='baseline' else mask).sum()) or list(features[r['arm']].columns)!=r['features']:raise ValueError('Fit feature population differs')
            else:
                probabilities={}
                for name in (['plain','rich'] if phase=='decision' else [arm]):probabilities[name]=baseline_boosters[name].predict(features[name],num_threads=4);totals['direct_booster_calls']+=1
                score=np.mean([b.predict(features[arm],num_threads=4) for b in onset_boosters[arm]],axis=0);totals['direct_booster_calls']+=2
                base=manual_baseline(a,probabilities[arm],case['baseline_setting']);pred={'incumbent':base,'fixed_k':manual_topk(a,score,case['k_choice']['k'],base),'threshold':manual_threshold(a,score,base,case['threshold_choice']['setting'])}
                if phase=='decision':
                    for name,p in probabilities.items():totals['baseline_calibration_score_calls']+=verify_baseline_grid(a,p,decision_seal['baseline_calibration'][name])
                    expected_arm=max(['plain','rich'],key=lambda name:decision_seal['baseline_calibration'][name]['selected']['calibration_proxy_IoU'])
                    if arm!=expected_arm or case['baseline_setting']!=decision_seal['baseline_calibration'][arm]['selected']:raise ValueError('Decision baseline arm differs')
                    for r in decision_seal['threshold_grid']:same_metric(manual_metric(a,manual_threshold(a,score,base,r['setting'])),r['metrics'])
                    choice=max(decision_seal['threshold_grid'],key=lambda r:(r['metrics']['proxy_IoU'],not r['setting']['force'],-abs(r['setting']['threshold']-.5)))
                    if choice!=case['threshold_choice']:raise ValueError('Threshold choice differs')
                    for r in decision_seal['k_grid']:same_metric(manual_metric(a,manual_topk(a,score,r['k'],base)),r['metrics'])
                    if max(decision_seal['k_grid'],key=lambda r:(r['metrics']['proxy_IoU'],-r['k']))!=case['k_choice']:raise ValueError('Fixed-k control choice differs')
                    for role,m in case['decision_metrics'].items():same_metric(manual_metric(a,pred[role]),m)
                    if max(['incumbent','threshold'],key=lambda r:case['decision_metrics'][r]['proxy_IoU'])!=case['decision_selected']:raise ValueError('Decision challenger differs')
                else:
                    if phase=='assessment':pred['selected']=pred[case['selected']]
                    row=next(r for r in case['prediction_records'] if r['phase']==phase and r['seed']==seed)
                    if sha256_file(row['path'])!=row['sha256']:raise ValueError('Prediction cache changed')
                    with np.load(row['path']) as saved:
                        if not np.array_equal(saved['baseline_probability'],probabilities[arm]) or not np.array_equal(saved['onset_score'],score) or any(not np.array_equal(saved[k],p) for k,p in pred.items()):raise ValueError('Independent scores/decoders differ')
                    metrics={role:manual_metric(a,p) for role,p in pred.items()};wanted=next(r for r in case[phase] if r['seed']==seed)
                    for role,m in metrics.items():same_metric(m,wanted['models'][role])
                    evaluated[phase].append({'seed':seed,'models':metrics})
                for ongoing in [False,True]:
                    ids=np.flatnonzero(a['ongoing']==ongoing)
                    if len(ids):sample_ids.update([int(ids[0]),int(ids[-1])]);first_ids.add(int(ids[0]))
            for i,text in enumerate(a['origins']):
                origin=pd.Timestamp(str(text));date=origin.date().isoformat();raw=raw_day(date);ts=pd.to_datetime(raw.timestamp,utc=True)
                past=raw.loc[(ts>=origin-pd.Timedelta(minutes=60))&(ts<origin)].copy();history=pd.concat([past,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
                masked_path=masked_paths[date]
                if masked_path.as_posix() not in source_hashes:source_hashes[masked_path.as_posix()]=sha256_file(masked_path)
                condition='queue_ongoing' if a['ongoing'][i] else 'queue_onset';times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
                target=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id']).to_frame(index=False);offset=slice(i*width,(i+1)*width)
                for name in ['plain','rich']:
                    rebuilt=queue_features(history,target,net,topology,origin,allow_origin=True,include_occupancy=name=='rich')
                    if name=='rich':rebuilt['condition_ongoing']=float(condition=='queue_ongoing')
                    expected=features[name].iloc[offset].reset_index(drop=True)
                    if list(rebuilt.columns)!=list(expected.columns) or not np.array_equal(rebuilt.to_numpy(),expected.to_numpy(),equal_nan=True):raise ValueError('Fresh rolling features differ')
                    totals['feature_arm_window_rebuilds']+=1
                future=raw.loc[raw.timestamp.isin(times)].copy();future.loc[~eligible(future),'speed_kmh']=np.nan
                v=future.pivot_table(index='timestamp',columns='link_id',values='speed_kmh').reindex(index=times,columns=links).to_numpy(float);cut=net.v_cut.reindex(links).to_numpy(float)
                if not np.array_equal(a['labels'][i],(v<=cut).astype(np.float32)) or not np.array_equal(a['available'][i],np.isfinite(v)):raise ValueError('Fresh public teacher/eligibility differs')
                past_times=[(origin-pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in range(60,0,-5)]
                past.loc[~eligible(past),'speed_kmh']=np.nan;pv=past.pivot_table(index='timestamp',columns='link_id',values='speed_kmh').reindex(index=past_times,columns=links).to_numpy(float)
                actual_ongoing=bool(((pv<=cut).sum(axis=0)>=2).any())
                if pv.shape!=(12,len(links)) or np.isfinite(pv).mean()<.7 or actual_ongoing!=bool(a['ongoing'][i]) or not (v<=cut).any() or (not actual_ongoing and ((v[:-1]<=cut).any() or not (v[-1]<=cut).any())):raise ValueError('Selected proxy event safeguards differ')
                current=np.full(len(links),np.nan)
                for past_row in pv[::-1]:
                    use=~np.isfinite(current)&np.isfinite(past_row);current[use]=past_row[use]
                persistence=np.broadcast_to(current<=cut,v.shape);truth=(v<=cut);valid=np.isfinite(v);union=np.sum((persistence|truth)&valid);iou=float(np.sum((persistence&truth)&valid)/union) if union else 1.
                if iou>.9:raise ValueError('Persistence safeguard violated')
                if i in sample_ids:
                    for role,model in portable.items():
                        candidate=case['exports'][role]['candidate'];wanted=pred[candidate][offset];out=model.predict_window(history,target,net,topology,origin,condition);totals['portable_window_calls']+=1
                        if not out[['timestamp','link_id']].equals(target) or not np.array_equal(out.queue_pred,wanted):raise ValueError('Portable recipe differs')
                        if i in first_ids:
                            order=np.random.default_rng(93).permutation(width);shuffled=model.predict_window(history,target.iloc[order],net,topology,origin,condition);totals['additional_shuffled_calls']+=1
                            if not np.array_equal(shuffled.queue_pred,wanted[order]):raise ValueError('Portable prediction depends on target order')
                    if condition=='queue_onset' and not poisoned:
                        bad=pd.concat([history,history.iloc[[0]].assign(timestamp=times[0])],ignore_index=True)
                        for model in portable.values():
                            try:model.predict_window(bad,target,net,topology,origin,condition)
                            except ValueError:totals['future_history_rejections']+=1
                            else:raise ValueError('Future input accepted')
                        poisoned=True
                totals['raw_window_occurrences']+=1
            checks.append({'phase':phase,'seed':seed,'n_windows':len(a['origins']),'both_feature_arms_labels_eligibility_selected_event_guards_exact':True})
            print('rolling_verified_cache',case['name'],phase,seed,'windows',len(a['origins']),flush=True)
        gate=paired(evaluated['selection'],case['dates']['selection'],'incumbent',case['decision_selected']);same_paired(gate,case['selection_gate'])
        selected=case['decision_selected'] if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent'
        if selected!=case['selected']:raise ValueError('Selection gate differs')
        for role,r in case['comparisons'].items():same_paired(paired(evaluated['assessment'],case['dates']['assessment'],'incumbent',role),r)
        cases.append({'name':case['name'],'all_supervised_rows_capacity_schemas_grids_seals_gates_verified':True,'records':checks})
    for role,key in [('selected','selected_evidence_gate'),('threshold','threshold_secondary_gate')]:
        rows=[c['comparisons'][role] for c in study['cases']];n=sum(r['paired_macro_gain']>=.01 and r['ci95'] is not None and r['ci95'][0]>0 for r in rows);bounded=all(r['paired_macro_gain']>=-.01 for r in rows)
        if study[key]!={'n_cases_gain_01_positive_CI':n,'all_case_losses_bounded_01':bounded,'descriptive_gate_passed':n>=2 and bounded}:raise ValueError('Cross-case evidence gate differs')
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'totals':totals,'source_files_sha256':source_hashes,'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,
        'scope':'Fresh raw/masked selected-window input replay,both feature arms,teacher,proxy guards,independent grids/sets/scores/gates;sampled portable checks. No model retraining or independent candidate-universe/sampling RNG recreation. Unused neural cache x not a model input and not independently checked.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8');print('rolling_verified_COMPLETE',totals,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
