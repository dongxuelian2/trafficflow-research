"""Causal queue forecasting and observation-based train-only proxy labels."""
from __future__ import annotations
from dataclasses import dataclass
import pickle
import numpy as np
import pandas as pd
from .data import QUEUE_KEY,eligible
from .state import neighbors
from .metrics import grouped_iou

def queue_features(history,targets,net,topo,origin):
    origin=pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError('Forecast origin must include timezone')
    ts=pd.to_datetime(history.timestamp,utc=True)
    if (ts>=origin).any() or (ts<origin-pd.Timedelta(minutes=60)).any():
        raise ValueError('History outside [origin-60min, origin); future input leakage')
    tt=pd.to_datetime(targets.timestamp,utc=True)
    if (tt<=origin).any() or (tt>origin+pd.Timedelta(minutes=30)).any():
        raise ValueError('Queue targets must be after origin, within 30 minutes')
    d=history.copy()
    d.loc[~eligible(d),['speed_kmh','flow_vph']]=np.nan
    links=targets.link_id.drop_duplicates().tolist()
    last=history.timestamp.max()
    records=[]
    for link in links:
        h=d[d.link_id==link].sort_values('timestamp')
        vf=float(net.loc[link,'free_speed_kmh']);lanes=float(net.loc[link,'lanes'])
        good=h[h.speed_kmh.notna()]
        if len(good):
            t=pd.to_datetime(good.timestamp,utc=True)
            ratio=good.speed_kmh.to_numpy(float)/vf
            age=float((origin-t.iloc[-1]).total_seconds()/60)
            cur=ratio[-1]
            recent=t>=origin-pd.Timedelta(minutes=30)
            minutes=(t[recent]-origin).dt.total_seconds().to_numpy()/60
            slope=float(np.polyfit(minutes,ratio[recent],1)[0]) if recent.sum()>1 else 0.
            std=float(np.std(ratio[recent])) if recent.any() else .2
            flow=float(good.flow_vph.iloc[-1]/lanes)
            mean=float(ratio.mean())
        else:
            cur,mean,age,slope,std,flow=1.,1.,65.,0.,.2,0.
        final=h[h.timestamp==last]
        persist=bool(len(final) and (final.speed_kmh<=float(net.loc[link,'v_cut'])).any())
        records.append({'link_id':link,'current':cur,'mean60':mean,'slope':slope,'speed_std':std,
                        'age':age,'flow_per_lane':flow,'coverage':len(good)/max(len(h),1),
                        'queue_now':float(persist),'cut_ratio':float(net.loc[link,'v_cut']/vf)})
    stats=pd.DataFrame(records).set_index('link_id')
    for direction,label in [('incoming','up'),('outgoing','down')]:
        ns=neighbors(topo,links,direction)
        for field in ['current','slope','flow_per_lane']:
            values=stats[field].to_numpy()
            stats[label+'_'+field]=[float(np.mean(values[idx])) if idx else np.nan for idx in ns]
    x=stats.reindex(targets.link_id).reset_index(drop=True)
    lead=(tt-origin).dt.total_seconds().to_numpy()/60
    elapsed=lead+x.age.to_numpy()
    x['lead_minutes']=lead
    x['trend_forecast']=x.current+x.slope*elapsed
    x['free_speed']=targets.link_id.map(net.free_speed_kmh).to_numpy(float)
    x['capacity_per_lane']=targets.link_id.map(net.capacity_vph/net.lanes).to_numpy(float)
    x['length_km']=targets.link_id.map(net.length_km).to_numpy(float)
    x['density_per_lane']=x.flow_per_lane/np.maximum(x.current*x.free_speed,1.)
    # Backward wave arrival from downstream measured link; topology defines direction.
    x['wave_arrival_minutes']=x.length_km/20.*60.
    down_queued=x.down_current.fillna(1.)<=x.cut_ratio
    x['wave_reached']=(down_queued&(elapsed>=x.wave_arrival_minutes)).astype(float)
    minute=origin.hour*60+origin.minute
    x['sin_tod']=np.sin(2*np.pi*minute/1440.)
    x['cos_tod']=np.cos(2*np.pi*minute/1440.)
    x['weekday']=origin.weekday()
    return x.astype(np.float32)

def wave_probability(x):
    # Explicit arrival uncertainty avoids hard teleportation of a queue upstream.
    z=(x.cut_ratio.to_numpy()-x.trend_forecast.to_numpy())/.09
    own=1/(1+np.exp(-np.clip(z,-30,30)))
    downstream=1/(1+np.exp(-np.clip((x.cut_ratio-x.down_current.fillna(1.))/.08,-30,30)))
    arrival=1/(1+np.exp(-np.clip((x.lead_minutes+x.age-x.wave_arrival_minutes)/6.,-30,30)))
    return np.clip(1-(1-own)*(1-.65*downstream.to_numpy()*arrival.to_numpy()),0,1)

@dataclass
class ProxyData:
    x: pd.DataFrame
    y: np.ndarray
    window: np.ndarray
    condition: np.ndarray
    date: np.ndarray

def proxy_windows(frames,net,topo,panel,max_per_day=6,spacing_minutes=360):
    xs,ys,wids,conditions,dates=[],[],[],[],[]
    for raw in frames:
        times=sorted(raw.timestamp.unique())
        if len(times)<20:
            continue
        links=raw.link_id.drop_duplicates().tolist()
        good=raw.copy();good.loc[~eligible(raw),['speed_kmh','flow_vph']]=np.nan
        pivot=good.pivot_table(index='timestamp',columns='link_id',values='speed_kmh',aggfunc='mean').reindex(index=times,columns=links)
        speed=pivot.to_numpy(float)
        cutoff=net.v_cut.reindex(links).to_numpy(float)
        queued=speed<=cutoff
        last_by_condition={};chosen=0
        for at in range(12,len(times)-7):
            future=speed[at+1:at+7]
            if np.isfinite(speed[at-12:at]).mean()<.7 or not (future<=cutoff).any():
                continue
            # Horizon labels are public train observations; never true queue labels.
            cond='queue_ongoing' if queued[at-1].any() else 'queue_onset'
            if cond=='queue_onset' and queued[at-12:at].any():
                continue
            if at-last_by_condition.get(cond,-9999)<spacing_minutes/5:
                continue
            origin=pd.Timestamp(times[at])
            wanted=raw.timestamp.isin(times[at-12:at])
            history=raw.loc[wanted].copy()
            target=pd.MultiIndex.from_product([times[at+1:at+7],links],names=['timestamp','link_id']).to_frame(index=False)
            x=queue_features(history,target,net,topo,origin)
            y=(future<=cutoff).ravel().astype(int)
            usable=np.isfinite(future).ravel()
            wid=f'{panel}:{times[at]}'
            xs.append(x.loc[usable]);ys.extend(y[usable]);wids.extend([wid]*usable.sum())
            conditions.extend([cond]*usable.sum());dates.extend([times[at][:10]]*usable.sum())
            last_by_condition[cond]=at;chosen+=1
            if chosen>=max_per_day:
                break
    if not xs:
        raise ValueError('No eligible queue proxy windows; need public train days with congestion')
    return ProxyData(pd.concat(xs,ignore_index=True),np.array(ys),np.array(wids),np.array(conditions),np.array(dates))

class QueueModel:
    def __init__(self,method='wave',seed=2026,iterations=250,threads=4):
        self.method,self.seed,self.iterations,self.threads=method,seed,iterations,threads
        self.model=None;self.threshold=.5

    def fit(self,data:ProxyData):
        if len(np.unique(data.y))<2:
            raise ValueError('Proxy training must contain both queue classes')
        if self.method=='lightgbm':
            from lightgbm import LGBMClassifier
            self.model=LGBMClassifier(n_estimators=self.iterations,learning_rate=.035,num_leaves=23,
                 min_child_samples=60,reg_lambda=8.,random_state=self.seed,n_jobs=self.threads,
                 verbosity=-1,deterministic=True,force_col_wise=True)
            self.model.fit(data.x,data.y)
        elif self.method!='wave' and self.method!='persistence':
            raise ValueError('Unknown queue model')
        return self

    def probability(self,x):
        if self.method=='persistence':
            return x.queue_now.to_numpy()
        if self.method=='wave':
            return wave_probability(x)
        if self.model is None:
            raise ValueError('Queue model not trained')
        return self.model.predict_proba(x)[:,1]

    def tune(self,data:ProxyData):
        p=self.probability(data.x)
        rows=[]
        for t in np.linspace(.2,.8,13):
            score=grouped_iou(data.y,p>=t,data.window,data.condition)['proxy_IoU']
            rows.append({'threshold':float(t),'proxy_IoU':score})
        # Deterministic tie-break toward 0.5 avoids unnecessary threshold drift.
        best=max(rows,key=lambda r:(r['proxy_IoU'],-abs(r['threshold']-.5)))
        self.threshold=best['threshold']
        return {'selected_threshold':self.threshold,'calibration_grid':rows,
                'proxy_Brier':float(np.mean((p-data.y)**2)),
                'label_source':'eligible public train speed <= released cutoff; noisy observation proxy'}

    def save(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self,f)

    @classmethod
    def load(cls,path):
        with path.open('rb') as f:
            obj=pickle.load(f)
        if not isinstance(obj,cls):
            raise ValueError('Invalid queue artifact')
        return obj

def predict_windows(release,panel,split,model):
    from .data import csv
    folder=release.root/'task2'/panel/split
    windows=csv(folder/'window_index.csv')
    history=pd.read_parquet(folder/'window_history.parquet')
    for c in ['timestamp','link_id','window_id']:
        history[c]=history[c].astype(str)
    template=release.template(2,panel,split)
    net,topo=release.network(panel),release.topology(panel)
    pieces=[]
    for w in windows.itertuples(index=False):
        h=history[history.window_id==str(w.window_id)]
        target=template[template.window_id==str(w.window_id)].copy()
        if not len(h) or not len(target):
            raise ValueError('Window history/target missing')
        # Published [T-60,T) history ends at T-5; infer T without reading any corridor future.
        origin=pd.Timestamp(h.timestamp.max())+pd.Timedelta(minutes=5)
        for name in ('forecast_origin','forecast_origin_time','origin_timestamp','T'):
            if hasattr(w,name):
                origin=pd.Timestamp(getattr(w,name))
                break
        x=queue_features(h,target,net,topo,origin)
        target['queue_pred']=(model.probability(x)>=model.threshold).astype(int)
        pieces.append(target[QUEUE_KEY+['queue_pred']])
    result=pd.concat(pieces,ignore_index=True)
    if len(result)!=len(template):
        raise ValueError('Queue template/window mismatch')
    return result
