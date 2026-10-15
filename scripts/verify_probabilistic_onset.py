"""Independent quadrature IoU curves, convex calibration and public inputs."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from lightgbm import Booster
from trafficflow.data import Release,eligible
from trafficflow.experiments import partition_date
from trafficflow.causal_queue import released_origin_rows
from trafficflow.queue import QueueModel,queue_features
from trafficflow.expected_iou_queue import independent_iou_set
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol
from verify_onset_rank_queue import manual_baseline,manual_metric,same_metric,paired,same_paired
from verify_rolling_onset_decoders import manual_threshold


@lru_cache(maxsize=8)
def quadrature(n):
    x,w=np.polynomial.legendre.leggauss((n+1)//2);return (x+1)/2,w/2


def integral_curve(probability,links):
    """Integrate a degree N-1 polynomial, independently of suffix convolution."""
    p=np.asarray(probability,float);links=np.asarray(links).astype(str);order=np.lexsort((links,-p));q=p[order];nodes,weights=quadrature(len(q));scores=[float(np.prod(1-q))]
    for k in range(1,len(q)+1):
        outside=np.prod(1-q[k:,None]+q[k:,None]*nodes[None,:],axis=0)
        scores.append(float(np.sum(q[:k])*np.dot(weights,nodes**(k-1)*outside)))
    return order,np.array(scores)


def curve_decision(order,curve,allow_empty):
    start=0 if allow_empty else 1;ids=np.arange(start,len(curve));k=int(ids[np.flatnonzero(curve[ids]>=np.max(curve[ids])-1e-12)[0]]);out=np.zeros(len(order),int);out[order[:k]]=1;return out


def independent_calibration(probability,y,window,state):
    p=np.asarray(probability,float);y=np.asarray(y,float);keys,codes=np.unique(window.astype(str),return_inverse=True)
    if state['n_windows']!=len(keys) or state['n_rows']!=len(y) or state['n_positive']!=int(y.sum()) or state['regularization']!=.01 or state['clip']!=1e-6:raise ValueError('Calibration-only population/config differs')
    if not len(y):
        if state['fitted'] or state['slope']!=1 or state['intercept']!=0:raise ValueError('Empty calibration fallback differs')
        return 0.,0.
    weights=np.array([1/np.sum(window==w) for w in window]);weights/=weights.sum();clipped=np.minimum(np.maximum(p,1e-6),1-1e-6);z=np.log(clipped/(1-clipped))
    def objective(theta):
        linear=theta[0]*z+theta[1];residual=expit(linear)-y;delta=theta-np.array([1.,0.])
        return float(np.sum(weights*(np.logaddexp(0,linear)-y*linear))+.005*np.sum(delta**2)),np.array([np.sum(weights*residual*z),np.sum(weights*residual)])+.01*delta
    target=np.array([state['slope'],state['intercept']]);minimum_fitted=len(keys)>=3 and len(np.unique(y))==2
    if bool(state['fitted'])!=minimum_fitted:raise ValueError('Insufficient-window fallback differs')
    if not minimum_fitted:
        if not np.array_equal(target,[1.,0.]):raise ValueError('Identity fallback differs')
        return 0.,0.
    result=minimize(objective,[1.,0.],jac=True,method='SLSQP',bounds=[(.05,5.),(-8.,8.)],options={'ftol':1e-14,'maxiter':1000})
    error=float(np.max(np.abs(result.x-target)));loss,gradient=objective(target);projected=gradient.copy()
    for i,(left,right) in enumerate([(.05,5.),(-8.,8.)]):
        if target[i]<=left+1e-8 and gradient[i]>=0 or target[i]>=right-1e-8 and gradient[i]<=0:projected[i]=0.
    stationarity=float(np.max(np.abs(projected)))
    if not result.success or error>1e-5 or abs(loss-state['objective'])>1e-10 or stationarity>1e-7:raise ValueError('Independent convex calibration/KKT differs')
    loss_after=loss-.005*np.sum((target-[1.,0.])**2)
    if abs(objective(np.array([1.,0.]))[0]-state['loss_before'])>1e-12 or abs(loss_after-state['loss_after'])>1e-12:raise ValueError('Calibration loss record differs')
    return error,stationarity


def calibrated(state,p):
    clipped=np.minimum(np.maximum(p,1e-6),1-1e-6);return 1/(1+np.exp(-(state['slope']*np.log(clipped/(1-clipped))+state['intercept'])))


def decode(a,raw,cal,base,raw_setting,cal_setting):
    pred={'incumbent':base.copy(),'threshold_raw':manual_threshold(a,raw,base,raw_setting),'threshold_cal':manual_threshold(a,cal,base,cal_setting)}
    for role in ['iou_raw','iou_cal','iou_cal_nonempty']:pred[role]=base.copy()
    links=np.tile(a['links'],len(a['ongoing'])*6);checks=0;maximum=0.
    for key in pd.unique(a['window']):
        ids=np.flatnonzero(a['window']==key)
        if a['condition'][ids[0]]!='queue_onset':continue
        last=ids[a['lead_minutes'][ids]==30]
        for p,roles in [(raw,['iou_raw']),(cal,['iou_cal','iou_cal_nonempty'])]:
            order,curve=integral_curve(p[last],links[last]);actual=independent_iou_set(p[last],links[last])[1]['expected_by_size'];error=float(np.abs(curve-actual).max());maximum=max(maximum,error);checks+=1
            if error>1e-10:raise ValueError('Independent integral and DP curves differ')
            for role in roles:pred[role][ids]=0;pred[role][last]=curve_decision(order,curve,role!='iou_cal_nonempty')
    return pred,checks,maximum


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite probabilistic audit')
    study=json.loads(args.study.read_text());source=json.loads(Path('reports/research/C71_rolling_onset_threshold_refits.json').read_text());cfg=study['registered']['protocol'];proof=json.loads(Path(cfg['reused_raw_input_proof']).read_text())
    if sha256_file(cfg['reused_raw_input_proof'])!=cfg['reused_raw_input_proof_sha256']:raise ValueError('Reused input proof changed')
    for path,digest in {**cfg['unchanged_source_constructors_sha256'],**proof['source_files_sha256']}.items():
        if sha256_file(path)!=digest:raise ValueError('Previously audited public inputs/constructors changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_probabilistic_onset_audit','source_study_sha256':sha256_file(args.study),
        'scope':'Old2196raw proof reused explicitly;all new calibration windows rebuilt,all8calibration fits independently SLSQP/KKT;all new DP curves via exact-degree Gauss-Legendre integral,grids/policies/scores/gates and sampled portable checks. No tree refits.',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public');source_hashes={};records=[]
    totals={'new_calibration_raw_windows':0,'new_calibration_feature_arm_rebuilds':0,'new_calibration_states_verified':0,'quadrature_curve_checks':0,'direct_booster_calls':0,'portable_window_calls':0,'additional_shuffled_calls':0,'future_history_rejections':0}
    maximum_curve=maximum_coefficient=maximum_kkt=0.
    for case,old in zip(study['cases'],source['cases']):
        if case['name']!=old['name']:raise ValueError('Source case order differs')
        base_folder=Path('artifacts/research')/args.study.stem/case['name'];panel=case['panel'];arm=case['feature_arm'];net,topology=release.network(panel),release.topology(panel);models={a:[] for a in ['plain','rich']}
        for r in old['training']:
            if r['role']=='onset':
                if sha256_file(r['model']['path'])!=r['model']['sha256']:raise ValueError('Frozen classifier changed')
                models[r['arm']].append(Booster(model_file=r['model']['path']))
        for name,digest in case['seals'].items():
            if sha256_file(base_folder/(name+'_sealed.json'))!=digest:raise ValueError('New choice seal changed')
        for r in case['phase_reads']:
            requirement={'decision':'calibration','selection':'decision','assessment':'selection'}[r['phase']]
            if r['prior_seal_sha256']!=case['seals'][requirement]:raise ValueError('Choice read order differs')
        portable={}
        for role,r in case['exports'].items():
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Portable policy changed')
            portable[role]=QueueModel.load(Path(r['path']))
        incumbent=portable['incumbent'];raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};masked_paths={partition_date(release,p):p for p in release.files(panel,'train')}
        @lru_cache(maxsize=8)
        def raw_day(date):
            path=raw_paths[date];source_hashes[path.as_posix()]=sha256_file(path);return release.read(path)
        calibration=case['calibration_record']
        if calibration['dates']!=old['dates']['checkpoint']:raise ValueError('Probability calibration dates shifted')
        for date in calibration['dates']:
            p=raw_paths[date];source_hashes[p.as_posix()]=sha256_file(p)
        if sha256_file(calibration['path'])!=calibration['sha256']:raise ValueError('Calibration cache changed')
        a=dict(np.load(calibration['path']));features={}
        for f in calibration['feature_files']:
            if sha256_file(f['path'])!=f['sha256']:raise ValueError('Calibration feature cache changed')
            features['rich' if Path(f['path']).stem.endswith('_rich') else 'plain']=pd.read_parquet(f['path'])
        width=6*len(a['links']);times=pd.to_datetime(a['origins'],utc=True)
        if any(t.date().isoformat() not in calibration['dates'] for t in times) or len(times)>1 and (times[1:]-times[:-1]).min()<pd.Timedelta(minutes=360):raise ValueError('Calibration chronology/spacing differs')
        for i,text in enumerate(a['origins']):
            origin=pd.Timestamp(str(text));raw=raw_day(origin.date().isoformat());timestamp=pd.to_datetime(raw.timestamp,utc=True);history=raw.loc[(timestamp>=origin-pd.Timedelta(minutes=60))&(timestamp<origin)].copy();history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
            p=masked_paths[origin.date().isoformat()];source_hashes[p.as_posix()]=sha256_file(p);condition='queue_ongoing' if a['ongoing'][i] else 'queue_onset';future_times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]];target=pd.MultiIndex.from_product([future_times,a['links']],names=['timestamp','link_id']).to_frame(index=False);offset=slice(i*width,(i+1)*width)
            for name in ['plain','rich']:
                x=queue_features(history,target,net,topology,origin,allow_origin=True,include_occupancy=name=='rich')
                if name=='rich':x['condition_ongoing']=float(a['ongoing'][i])
                expected=features[name].iloc[offset].reset_index(drop=True)
                if list(x.columns)!=list(expected.columns) or not np.array_equal(x.to_numpy(),expected.to_numpy(),equal_nan=True):raise ValueError('Fresh calibration causal features differ')
                totals['new_calibration_feature_arm_rebuilds']+=1
            future=raw.loc[raw.timestamp.isin(future_times)].copy();future.loc[~eligible(future),'speed_kmh']=np.nan;v=future.pivot_table(index='timestamp',columns='link_id',values='speed_kmh').reindex(index=future_times,columns=a['links']).to_numpy(float)
            if not np.array_equal(a['labels'][i],(v<=net.v_cut.reindex(a['links']).to_numpy(float)).astype(np.float32)) or not np.array_equal(a['available'][i],np.isfinite(v)):raise ValueError('Fresh calibration observation teacher differs')
            totals['new_calibration_raw_windows']+=1
        use=(a['condition']=='queue_onset')&(a['lead_minutes']==30)&a['available'].ravel();row=case['calibration_probabilities']
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Calibration probability file changed')
        with np.load(row['path']) as saved:
            for name in ['plain','rich']:
                p=np.mean([b.predict(features[name],num_threads=4) for b in models[name]],axis=0) if len(features[name]) else np.array([]);totals['direct_booster_calls']+=2 if len(features[name]) else 0
                if not np.array_equal(p,saved[name]):raise ValueError('Calibration frozen probabilities differ')
                error,kkt=independent_calibration(p[use],a['labels'].ravel()[use],a['window'][use],case['calibration_states'][name]);maximum_coefficient=max(maximum_coefficient,error);maximum_kkt=max(maximum_kkt,kkt);totals['new_calibration_states_verified']+=1
        evaluated={'selection':[],'assessment':[]};poisoned=False;case_checks=[]
        for item in case['source_records']:
            r=item['source_record'];f=item['feature_file'];phase=r['phase']
            if sha256_file(r['path'])!=r['sha256'] or sha256_file(f['path'])!=f['sha256']:raise ValueError('Reused verified evaluation cache changed')
            a=dict(np.load(r['path']));x=pd.read_parquet(f['path']);raw=np.mean([b.predict(x,num_threads=4) for b in models[arm]],axis=0);cal=calibrated(case['calibration_states'][arm],raw);base=manual_baseline(a,incumbent.baseline.probability(x),incumbent.setting);totals['direct_booster_calls']+=3
            pred,count,error=decode(a,raw,cal,base,old['threshold_choice']['setting'],case['threshold_choice']['setting']);totals['quadrature_curve_checks']+=count;maximum_curve=max(maximum_curve,error)
            if phase=='decision':
                for entry in case['threshold_grid']:same_metric(manual_metric(a,manual_threshold(a,cal,base,entry['setting'])),entry['metrics'])
                chosen=max(case['threshold_grid'],key=lambda r:(r['metrics']['proxy_IoU'],not r['setting']['force'],-abs(r['setting']['threshold']-.5)))
                if chosen!=case['threshold_choice']:raise ValueError('Calibrated threshold winner differs')
                for role,m in case['decision_metrics'].items():same_metric(manual_metric(a,pred[role]),m)
                order=['incumbent','threshold_raw','threshold_cal','iou_raw','iou_cal','iou_cal_nonempty']
                if max(order,key=lambda k:case['decision_metrics'][k]['proxy_IoU'])!=case['decision_selected']:raise ValueError('New decision policy differs')
            else:
                if phase=='assessment':pred['selected']=pred[case['selected']]
                row=next(p for p in case['prediction_records'] if p['phase']==phase and p['seed']==r['seed'])
                if sha256_file(row['path'])!=row['sha256']:raise ValueError('Stored new forecast changed')
                with np.load(row['path']) as saved:
                    if not np.array_equal(saved['raw_probability'],raw) or not np.allclose(saved['calibrated_probability'],cal,atol=1e-15,rtol=0) or any(not np.array_equal(saved[role],p) for role,p in pred.items()):raise ValueError('Independent integration decoder differs')
                metrics={k:manual_metric(a,p) for k,p in pred.items()};wanted=next(p for p in case[phase] if p['seed']==r['seed'])
                for k,m in metrics.items():same_metric(m,wanted['models'][k])
                evaluated[phase].append({'seed':r['seed'],'models':metrics})
            sample=set();first=set();width=6*len(a['links'])
            for ongoing in [False,True]:
                ids=np.flatnonzero(a['ongoing']==ongoing)
                if len(ids):sample.update([int(ids[0]),int(ids[-1])]);first.add(int(ids[0]))
            for i in sorted(sample):
                origin=pd.Timestamp(str(a['origins'][i]));raw_day_frame=raw_day(origin.date().isoformat());ts=pd.to_datetime(raw_day_frame.timestamp,utc=True);history=raw_day_frame.loc[(ts>=origin-pd.Timedelta(minutes=60))&(ts<origin)].copy();history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
                future_times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]];target=pd.MultiIndex.from_product([future_times,a['links']],names=['timestamp','link_id']).to_frame(index=False);offset=slice(i*width,(i+1)*width);condition='queue_ongoing' if a['ongoing'][i] else 'queue_onset'
                for role,model in portable.items():
                    wanted=pred[case['exports'][role]['candidate']][offset];out=model.predict_window(history,target,net,topology,origin,condition);totals['portable_window_calls']+=1
                    if not out[['timestamp','link_id']].equals(target) or not np.array_equal(out.queue_pred,wanted):raise ValueError('Probabilistic portable forecast differs')
                    if i in first:
                        permutation=np.random.default_rng(97).permutation(width);out=model.predict_window(history,target.iloc[permutation],net,topology,origin,condition);totals['additional_shuffled_calls']+=1
                        if not np.array_equal(out.queue_pred,wanted[permutation]):raise ValueError('New model depends on target order')
                if condition=='queue_onset' and not poisoned:
                    bad=pd.concat([history,history.iloc[[0]].assign(timestamp=future_times[0])],ignore_index=True)
                    for model in portable.values():
                        try:model.predict_window(bad,target,net,topology,origin,condition)
                        except ValueError:totals['future_history_rejections']+=1
                        else:raise ValueError('Future measurement input accepted')
                    poisoned=True
            case_checks.append({'phase':phase,'seed':r['seed'],'new_calibration_quadrature_decisions_scores_exact':True})
        gate=paired(evaluated['selection'],case['dates']['selection'],'incumbent',case['decision_selected']);same_paired(gate,case['selection_gate']);selected=case['decision_selected'] if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent'
        if selected!=case['selected']:raise ValueError('New selection gate differs')
        for role,value in case['comparisons'].items():same_paired(paired(evaluated['assessment'],case['dates']['assessment'],'incumbent',role),value)
        records.append({'name':case['name'],'both_calibration_fits_and_new_decoders_grids_gates_verified':True,'records':case_checks});print('probabilistic_verified_case',case['name'],flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':records,'totals':totals,'reused_previous_raw_window_audit':2196,'source_files_sha256':source_hashes,'maximum_integral_DP_curve_difference':maximum_curve,'maximum_calibration_parameter_difference':maximum_coefficient,'maximum_projected_KKT_gradient':maximum_kkt,
        'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,'scope':'Independent mathematical integration/convex fitting and new calibration raw replay;old evaluation raw proof reused,not newly2196raw rebuilds. Portable checks use fresh sampled histories,not all evaluation windows. No independent classifier refit or full sampler RNG recreation.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8');print('probabilistic_verified_COMPLETE',totals,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
