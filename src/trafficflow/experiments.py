"""Chronological fit/calibration/test experiments on released train observations."""
from __future__ import annotations
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .data import Release,STATE_KEY,VALUES,eligible
from .metrics import state_score,grouped_iou,total_score
from .state import Profile,StateFeatures,StateModel
from .queue import QueueModel,proxy_windows
from .physics import diagnose,refine,grids,reconstruct
from .odme import predict as od_predict,operator,count_holdout

def partition_date(release,path):
    import re
    found=re.search(r'(20\d\d)[_-](\d\d)[_-](\d\d)',path.name)
    if found:
        return '-'.join(found.groups())
    return str(release.read(path).timestamp.iloc[0])[:10]

def groups(release,panel,profile_days=None):
    pairs=sorted((partition_date(release,p),p) for p in release.files(panel,'train',True))
    dates=sorted({d for d,p in pairs})
    if len(dates)<20:
        raise ValueError('At least 20 train days required for chronological validation')
    p=profile_days or max(7,len(dates)//6)
    fit_end=max(p+4,int(.65*len(dates)));cal_end=max(fit_end+3,int(.82*len(dates)))
    if cal_end>=len(dates):
        raise ValueError('Not enough dates after profile/fit/calibration boundaries')
    sets={'profile':dates[:p],'fit':dates[p:fit_end],'calibration':dates[fit_end:cal_end],'test':dates[cal_end:]}
    return {name:[path for d,path in pairs if d in ds] for name,ds in sets.items()},sets

def raw_for(release,panel,masked_paths):
    by_name={p.name:p for p in release.files(panel,'train',False)}
    for path in masked_paths:
        if path.name not in by_name:
            raise ValueError('Masked partition has no public train counterpart')
        yield release.read(by_name[path.name])

def day_template(release,panel,split,frame,template=None):
    if template is None:
        template=release.template(1,panel,split)
    return template[template.timestamp.isin(frame.timestamp.unique())].copy()

def target_truth(release,panel,template,raw):
    truth=template[STATE_KEY].merge(raw[['timestamp','station_id','link_id']+VALUES+['pct_observed']],
                      on=['timestamp','station_id','link_id'],how='left',validate='one_to_one')
    if not eligible(truth).all():
        raise ValueError('Task 1 template contains non-eligible or absent truth')
    return truth[STATE_KEY+VALUES]

def training_rows(release,panel,paths,profile,max_rows,seed,feature_version=2):
    net,topo=release.network(panel),release.topology(panel)
    template=release.template(1,panel,'train')
    rng=np.random.default_rng(seed);xs,ys,bs=[],[],[]
    per_day=max(1,max_rows//max(len(paths),1))
    for path,raw in zip(paths,raw_for(release,panel,paths)):
        masked=release.read(path);target=day_template(release,panel,'train',masked,template)
        if len(target)>per_day:
            target=target.iloc[np.sort(rng.choice(len(target),per_day,replace=False))]
        if not len(target):
            continue
        f=StateFeatures(masked,net,topo,profile,version=feature_version)
        idx=f.lookup(target)
        y=target_truth(release,panel,target,raw)[VALUES].to_numpy(float,copy=True)
        y[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
        xs.append(f.features.iloc[idx]);ys.append(y);bs.append(f.base.reshape(-1,2)[idx])
    if not xs:
        raise ValueError('No state training targets')
    return pd.concat(xs,ignore_index=True),np.concatenate(ys),np.concatenate(bs)

def evaluate_state(release,panel,paths,model,physics_strength=0.,fd_bounds=False):
    net,topo=release.network(panel),release.topology(panel)
    template=release.template(1,panel,'train')
    truths,predictions,lanes,diagnostics,daily_stats=[],[],[],[],[]
    for path,raw in zip(paths,raw_for(release,panel,paths)):
        frame=release.read(path);target=day_template(release,panel,'train',frame,template)
        if not len(target):
            continue
        features=StateFeatures(frame,net,topo,model.profile,version=getattr(model,'feature_version',1))
        pred=model.predict(features,target)
        full,_=reconstruct(frame,pred);times,links,_,_=grids(full,net)
        ramps=release.ramps(panel,'train',times,links)
        if physics_strength or fd_bounds:
            pred=refine(frame,pred,net,topo,ramps,physics_strength,fd_bounds)
        diagnostics.append(diagnose(frame,pred,net,topo,ramps))
        truth=target_truth(release,panel,target,raw)
        lane=release.station_lanes(panel,target)
        truths.append(truth);predictions.append(pred);lanes.append(lane)
        for regime in target.mask_regime.unique():
            selected=(target.mask_regime==regime).to_numpy()
            speed=pred.speed_kmh.to_numpy()[selected]-truth.speed_kmh.to_numpy()[selected]
            flow=(pred.flow_vph.to_numpy()[selected]-truth.flow_vph.to_numpy()[selected])/lane[selected]
            rv,rq=np.sqrt(np.mean(speed**2)),np.sqrt(np.mean(flow**2))
            daily_stats.append({'date':str(target.timestamp.iloc[0])[:10],'regime':str(regime),'n':len(speed),
                'speed_sse':float(np.square(speed).sum()),'lane_flow_sse':float(np.square(flow).sum()),
                'daily_State':float(.54*max(0,1-rv/25)+.46*max(0,1-rq/600))})
    truth,pred,lane=pd.concat(truths,ignore_index=True),pd.concat(predictions,ignore_index=True),np.concatenate(lanes)
    result=state_score(truth,pred,lane)
    # Congestion and missingness strata are diagnostic, not averaged into the official score.
    ratio=truth.speed_kmh.to_numpy()/truth.link_id.map(net.free_speed_kmh).to_numpy()
    for name,sel in [('congested_observation_proxy',ratio<=.6),('free_observation_proxy',ratio>.6)]:
        result[name]=state_score(truth.loc[sel],pred,lane[sel]) if sel.any() else None
    result['physics_diagnostics']=diagnostics
    result['n_targets']=len(truth)
    result['daily_sufficient_statistics']=daily_stats
    return result

def run_experiment(release,panel,output,seed=2026,iterations=250,max_rows=300000,methods=None):
    started=time.perf_counter();output.mkdir(parents=True,exist_ok=True)
    partitions,dates=groups(release,panel)
    profile=Profile.fit(raw_for(release,panel,partitions['profile']))
    net,topo=release.network(panel),release.topology(panel)
    result={'panel':panel,'dataset':'synthetic_fixture' if (release.root/'FIXTURE_ONLY.json').exists() else 'official_public_train',
       'seed':seed,'dates':dates,'validation':'chronological profile / fit / calibration / untouched test',
       'formal_queue':None,'formal_physics':None,'formal_ODME':None,'state':{},'queue':{},'odme':{},'selection':{}}
    methods=methods or ['historical','interpolation','kalman','lightgbm','catboost']
    x=y=base=None
    models={}
    def checkpoint():
        (output/'experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for method in methods:
        t=time.perf_counter();model=StateModel(profile,method,seed,iterations)
        if method in ['lightgbm','catboost','joint_lightgbm']:
            if x is None:
                x,y,base=training_rows(release,panel,partitions['fit'],profile,max_rows,seed)
            model.fit(x,y,base)
        model.save(output/f'state_{method}.pkl');models[method]=model
        cal=evaluate_state(release,panel,partitions['calibration'],model)
        test=evaluate_state(release,panel,partitions['test'],model)
        result['state'][method]={'calibration':cal,'test':test,'seconds':time.perf_counter()-t}
        checkpoint()
        print(f'{panel} state {method}: calibration={cal["S_state"]:.6f}, test={test["S_state"]:.6f}',flush=True)
    # The model and physics setting are chosen from calibration alone.
    best=max(methods,key=lambda m:result['state'][m]['calibration']['S_state'])
    result['selection']['state_base']=best
    for strength,bounds in [(0.,True),(.001,False),(.01,False)]:
        tag=f'{best}_physics_{strength}_bounds_{bounds}'
        t=time.perf_counter()
        try:
            cal=evaluate_state(release,panel,partitions['calibration'],models[best],strength,bounds)
            result['state'][tag]={'calibration':cal,'seconds':time.perf_counter()-t,'strength':strength,'fd_bounds':bounds}
        except RuntimeError as exc:
            result['state'][tag]={'success':False,'error':str(exc),'seconds':time.perf_counter()-t,
                                 'strength':strength,'fd_bounds':bounds}
        checkpoint()
    candidates=[best]+[k for k in result['state'] if k.startswith(best+'_physics') and 'calibration' in result['state'][k]]
    selected=max(candidates,key=lambda m:result['state'][m]['calibration']['S_state'])
    if selected!=best:
        settings=result['state'][selected]
        settings['test']=evaluate_state(release,panel,partitions['test'],models[best],settings['strength'],settings['fd_bounds'])
    result['selection']['state']=selected
    result['selection']['physics_strength']=result['state'][selected].get('strength',0.)
    result['selection']['fd_bounds']=result['state'][selected].get('fd_bounds',False)
    if not panel.startswith('D12_I405'):
        train_proxy=proxy_windows(raw_for(release,panel,partitions['fit']),net,topo,panel)
        cal_proxy=proxy_windows(raw_for(release,panel,partitions['calibration']),net,topo,panel)
        test_proxy=proxy_windows(raw_for(release,panel,partitions['test']),net,topo,panel)
        for method in ['persistence','wave','lightgbm']:
            t=time.perf_counter();model=QueueModel(method,seed,iterations).fit(train_proxy)
            tune=model.tune(cal_proxy) if method!='persistence' else {'selected_threshold':.5}
            cal=grouped_iou(cal_proxy.y,model.probability(cal_proxy.x)>=model.threshold,cal_proxy.window,cal_proxy.condition)
            test=grouped_iou(test_proxy.y,model.probability(test_proxy.x)>=model.threshold,test_proxy.window,test_proxy.condition)
            model.save(output/f'queue_{method}.pkl')
            result['queue'][method]={'calibration':cal,'test':test,'tuning':tune,'seconds':time.perf_counter()-t,
                       'n_training_windows':len(np.unique(train_proxy.window)),'n_test_windows':len(np.unique(test_proxy.window))}
            checkpoint()
            print(f'{panel} queue {method}: proxy test IoU={test["proxy_IoU"]:.6f}',flush=True)
        result['selection']['queue']=max(result['queue'],key=lambda m:result['queue'][m]['calibration']['proxy_IoU'])
    for method,reg in [('prior',0.),('ridge',.05),('ridge',.005),('relative',5.),('poisson',.05)]:
        t=time.perf_counter()
        try:
            _,info=od_predict(release,panel,'train',method,reg)
            A,c,b,_=operator(release,panel,'train')
            info['sensor_holdout']=count_holdout(A,c,b,reg,seed,method)
        except RuntimeError as exc:
            info={'success':False,'error':str(exc),'S_ODME':None}
        info['seconds']=time.perf_counter()-t
        result['odme'][f'{method}:{reg}']=info
        checkpoint()
    result['selection']['odme']={'method':'ridge','regularization':.05,
                                'reason':'Conservative default; link-fit alone cannot identify hidden OD truth.'}
    state=result['state'][selected]['test']['S_state']
    result['local_score_contract']=total_score({'state':state,'queue':None,'physics':None,'odme':None})
    result['formal_score_contract']=total_score({k:None for k in ['state','queue','physics','odme']})
    result['seconds']=time.perf_counter()-started
    from .runtime import evidence
    result['runtime']=evidence()
    (output/'experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result
