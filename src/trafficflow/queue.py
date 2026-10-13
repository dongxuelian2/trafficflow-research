"""Causal queue forecasting and observation-based train-only proxy labels."""
from __future__ import annotations
from dataclasses import dataclass
import pickle
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from .data import QUEUE_KEY,eligible
from .state import neighbors
from .metrics import grouped_iou

@lru_cache(maxsize=16)
def _road_distances(nodes,lengths,waves,edges):
    index={link:i for i,link in enumerate(nodes)}
    rr,cc,km,minutes=[],[],[],[]
    for source,destination in edges:
        if source not in index or destination not in index:continue
        a,b=index[source],index[destination]
        rr.append(a);cc.append(b);km.append(.5*(lengths[a]+lengths[b]))
        minutes.append(30*(lengths[a]/waves[a]+lengths[b]/waves[b]))
    shape=(len(nodes),len(nodes))
    return (shortest_path(sparse.csr_matrix((km,(rr,cc)),shape=shape),directed=True),
            shortest_path(sparse.csr_matrix((minutes,(rr,cc)),shape=shape),directed=True))

def road_context(stats,links,net,topo):
    nodes=tuple(net.index.astype(str));index={link:i for i,link in enumerate(nodes)}
    lengths=tuple(net.length_km.to_numpy(float))
    waves=tuple((net.capacity_vph/(net.k_jam-net.critical_density).clip(lower=1e-6)).clip(lower=1.).to_numpy(float))
    edges=[]
    for link,row in topo.iterrows():
        edges.extend((str(link),other.strip()) for other in str(row.get('outgoing_link_ids','')).split(';') if other.strip())
    km,minutes=_road_distances(nodes,lengths,waves,tuple(edges))
    idx=np.array([index[link] for link in links]);km=km[np.ix_(idx,idx)];minutes=minutes[np.ix_(idx,idx)]
    origin_ratio=np.clip(stats.current.to_numpy()+stats.slope.to_numpy()*stats.age.to_numpy(),.05,1.15)
    queued=origin_ratio<=stats.cut_ratio.to_numpy()
    stats['origin_ratio']=origin_ratio
    for matrix,label in [(km,'downstream_queue_km'),(minutes,'downstream_queue_arrival_min'),(km.T,'upstream_queue_km')]:
        stats[label]=np.minimum(np.min(matrix[:,queued],axis=1),1000.) if queued.any() else 1000.
    for radius in [5,15,30]:
        included=minutes<=radius
        values=np.where(included,origin_ratio[None,:],np.inf)
        stats[f'downstream_min_ratio_{radius}min']=np.min(values,axis=1)
        stats[f'downstream_mean_ratio_{radius}min']=(included*origin_ratio[None,:]).sum(axis=1)/np.maximum(included.sum(axis=1),1)
    stats['corridor_queue_fraction']=float(queued.mean())
    if 'order_index' in topo:
        order=pd.to_numeric(topo.order_index,errors='coerce').reindex(links).to_numpy(float)
        stats['position_fraction']=order/max(float(np.nanmax(order)),1.)
    else:
        stats['position_fraction']=np.arange(len(links))/max(len(links)-1,1)
    for kind in ['on','off']:
        field=kind+'_ramp_link_ids'
        stats[kind+'_ramp_count']=[len([s for s in str(topo.loc[link,field]).split(';') if s]) if link in topo.index and field in topo else 0 for link in links]
    return stats

def queue_features(history,targets,net,topo,origin,allow_origin=False,latest_valid_queue=False,include_occupancy=False):
    origin=pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError('Forecast origin must include timezone')
    ts=pd.to_datetime(history.timestamp,utc=True)
    after=(ts>origin) if allow_origin else (ts>=origin)
    if after.any() or (ts<origin-pd.Timedelta(minutes=60)).any():
        raise ValueError('History outside [origin-60min, origin); future input leakage')
    tt=pd.to_datetime(targets.timestamp,utc=True)
    if (tt<=origin).any() or (tt>origin+pd.Timedelta(minutes=30)).any():
        raise ValueError('Queue targets must be after origin, within 30 minutes')
    d=history.copy()
    d.loc[~eligible(d),['speed_kmh','flow_vph']]=np.nan
    links=targets.link_id.drop_duplicates().tolist()
    # Build each window grid once rather than scanning every link's history.
    history_times=sorted(d.timestamp.unique())
    vf=net.free_speed_kmh.reindex(links).to_numpy(float)
    lane=net.lanes.reindex(links).to_numpy(float)
    cut=net.v_cut.reindex(links).to_numpy(float)
    speeds=d.pivot_table(index='timestamp',columns='link_id',values='speed_kmh',aggfunc='mean').reindex(index=history_times,columns=links).to_numpy(float)
    flows=d.pivot_table(index='timestamp',columns='link_id',values='flow_vph',aggfunc='mean').reindex(index=history_times,columns=links).to_numpy(float)
    ratio=speeds/vf;valid=np.isfinite(ratio);counts=valid.sum(axis=0)
    rows=np.arange(len(history_times))[:,None]
    last_indices=np.max(np.where(valid,rows,-1),axis=0)
    safe_indices=np.maximum(last_indices,0);columns=np.arange(len(links))
    current=ratio[safe_indices,columns];flow=flows[safe_indices,columns]/lane
    dt=pd.to_datetime(history_times,utc=True)
    minutes=np.asarray((dt-origin).total_seconds()/60)
    age=-minutes[safe_indices]
    recent=valid&(minutes[:,None]>=-30)
    n=recent.sum(axis=0);xminutes=np.broadcast_to(minutes[:,None],ratio.shape)
    sx=np.where(recent,xminutes,0).sum(axis=0);sy=np.where(recent,ratio,0).sum(axis=0)
    sxx=np.where(recent,xminutes*xminutes,0).sum(axis=0)
    sxy=np.where(recent,xminutes*ratio,0).sum(axis=0)
    denominator=n*sxx-sx*sx
    slope=np.divide(n*sxy-sx*sy,denominator,out=np.zeros(len(links)),where=(n>1)&(denominator>0))
    mean_recent=np.divide(sy,n,out=np.zeros(len(links)),where=n>0)
    sq=np.where(recent,ratio*ratio,0).sum(axis=0)
    variance=np.divide(sq,n,out=np.zeros(len(links)),where=n>0)-mean_recent**2
    std=np.where(n>0,np.sqrt(np.maximum(variance,0)),.2)
    mean=np.divide(np.nansum(ratio,axis=0),counts,out=np.ones(len(links)),where=counts>0)
    row_counts=d.groupby('link_id').size().reindex(links).fillna(0).to_numpy()
    stats=pd.DataFrame({'current':np.where(counts>0,current,1.),'mean60':mean,
         'slope':slope,'speed_std':std,'age':np.where(counts>0,age,65.),
         'flow_per_lane':np.where(counts>0,flow,0.),'coverage':counts/np.maximum(row_counts,1),
         'queue_now':(speeds[-1]<=cut).astype(float),'cut_ratio':cut/vf},index=links)
    if allow_origin or latest_valid_queue:
        # The current public rules permit released measurements at T. For
        # missing T cells, retain the latest eligible reading in the history.
        stats['queue_now']=(stats.current.to_numpy()*vf<=cut).astype(float)
    flow_ratios=flows/lane/(net.capacity_vph.reindex(links).to_numpy(float)/lane)
    qsum=np.where(recent,flow_ratios,0).sum(axis=0)
    qxy=np.where(recent,xminutes*flow_ratios,0).sum(axis=0)
    stats['flow_ratio_slope']=np.divide(n*qxy-sx*qsum,denominator,out=np.zeros(len(links)),where=(n>1)&(denominator>0))
    stats['condition_ongoing']=float(((ratio<=cut/vf).sum(axis=0)>=2).any())
    for direction,label in [('incoming','up'),('outgoing','down')]:
        ns=neighbors(topo,links,direction)
        for field in ['current','slope','flow_per_lane']:
            values=stats[field].to_numpy()
            stats[label+'_'+field]=[float(np.mean(values[idx])) if idx else np.nan for idx in ns]
    if include_occupancy:
        # window_history publishes occupancy but not its derived density.
        # Use only the common released channel so training and inference agree.
        occupancy=d['occupancy'].where(eligible(d))
        measured=d.assign(_occupancy=occupancy)
        occ=measured.pivot_table(index='timestamp',columns='link_id',values='_occupancy',aggfunc='mean').reindex(index=history_times,columns=links).to_numpy(float)
        ov=np.isfinite(occ)
        oi=np.maximum(np.max(np.where(ov,rows,-1),axis=0),0)
        oc=ov.sum(axis=0)
        recent_occ=ov&(minutes[:,None]>=-30)
        on=recent_occ.sum(axis=0)
        osx=np.where(recent_occ,xminutes,0).sum(axis=0)
        osy=np.where(recent_occ,occ,0).sum(axis=0)
        osxx=np.where(recent_occ,xminutes*xminutes,0).sum(axis=0)
        osxy=np.where(recent_occ,xminutes*occ,0).sum(axis=0)
        od=on*osxx-osx*osx
        stats['occupancy_current']=np.where(oc>0,occ[oi,columns],np.nan)
        stats['occupancy_age']=np.where(oc>0,-minutes[oi],65.)
        stats['occupancy_mean30']=np.divide(osy,on,out=np.full(len(links),np.nan),where=on>0)
        stats['occupancy_slope']=np.divide(on*osxy-osx*osy,od,out=np.zeros(len(links)),where=(on>1)&(od>0))
        stats['occupancy_coverage']=oc/np.maximum(row_counts,1)
        stats['occupancy_speed_density_ratio']=np.where(counts>0,flow/np.maximum(current*vf,1),np.nan)/np.maximum(stats.occupancy_current,.005)
        for direction,label in [('incoming','up'),('outgoing','down')]:
            ns=neighbors(topo,links,direction)
            for field in ['occupancy_current','occupancy_slope']:
                values=stats[field].to_numpy()
                stats[label+'_'+field]=[float(np.nanmean(values[idx])) if idx and np.isfinite(values[idx]).any() else np.nan for idx in ns]
    stats=road_context(stats,links,net,topo)
    x=stats.reindex(targets.link_id).reset_index(drop=True)
    lead=(tt-origin).dt.total_seconds().to_numpy()/60
    elapsed=lead+x.age.to_numpy()
    x['lead_minutes']=lead
    x['trend_forecast']=x.current+x.slope*elapsed
    x['free_speed']=targets.link_id.map(net.free_speed_kmh).to_numpy(float)
    x['capacity_per_lane']=targets.link_id.map(net.capacity_vph/net.lanes).to_numpy(float)
    x['flow_capacity_ratio']=x.flow_per_lane/x.capacity_per_lane
    x['density_critical_ratio']=x.flow_capacity_ratio/np.maximum(x.current,.05)
    x['future_flow_ratio']=x.flow_capacity_ratio+x.flow_ratio_slope*elapsed
    x['flow_pressure']=(x.up_flow_per_lane-x.down_flow_per_lane)/x.capacity_per_lane
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
    if 'downstream_queue_arrival_min' in x:
        arrival=1/(1+np.exp(-np.clip((x.lead_minutes-x.downstream_queue_arrival_min)/5.,-30,30)))
        return np.clip(1-(1-own)*(1-.8*arrival.to_numpy()),0,1)
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
    last_by_condition={}
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
        chosen=0
        for at in range(12,len(times)-7):
            future=speed[at+1:at+7]
            if np.isfinite(speed[at-12:at]).mean()<.7 or not (future<=cutoff).any():
                continue
            # Horizon labels are public train observations; never true queue labels.
            origin=pd.Timestamp(times[at])
            # Current released competition_contract: an established visible queue
            # needs at least two observations on the same link in the history.
            established=(queued[at-12:at].sum(axis=0)>=2).any()
            cond='queue_ongoing' if established else 'queue_onset'
            if cond in last_by_condition and (origin-last_by_condition[cond]).total_seconds()/60<spacing_minutes:
                continue
            wanted=raw.timestamp.isin(times[at-12:at])
            history=raw.loc[wanted].copy()
            target=pd.MultiIndex.from_product([times[at+1:at+7],links],names=['timestamp','link_id']).to_frame(index=False)
            x=queue_features(history,target,net,topo,origin)
            y=(future<=cutoff).ravel().astype(int)
            usable=np.isfinite(future).ravel()
            wid=f'{panel}:{times[at]}'
            xs.append(x.loc[usable]);ys.extend(y[usable]);wids.extend([wid]*usable.sum())
            conditions.extend([cond]*usable.sum());dates.extend([times[at][:10]]*usable.sum())
            last_by_condition[cond]=origin;chosen+=1
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
        # Preserve feature alignment when inspecting an older local model artifact.
        names=getattr(self.model,'feature_name_',None)
        return self.model.predict_proba(x.loc[:,names] if names is not None else x)[:,1]

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
        use_origin=getattr(model,'uses_origin_timestamp',False)
        if use_origin:
            from .causal_queue import append_released_origin
            h=append_released_origin(h,release,panel,split,origin)
        x=queue_features(h,target,net,topo,origin,allow_origin=use_origin,
                         latest_valid_queue=getattr(model,'latest_valid_queue_bookkeeping',False),
                         include_occupancy=getattr(model,'uses_occupancy_features',False))
        if getattr(model,'uses_published_condition',False):
            if not hasattr(w,'condition'):raise ValueError('Published window condition is required')
            x['condition_ongoing']=np.float32(str(w.condition)=='queue_ongoing')
        if hasattr(model,'classify'):
            if not hasattr(w,'condition'):
                raise ValueError('Conditional calibration requires released window condition')
            target['queue_pred']=model.classify(x,np.full(len(x),str(w.window_id)),np.full(len(x),str(w.condition)))
        else:
            target['queue_pred']=(model.probability(x)>=model.threshold).astype(int)
        pieces.append(target[QUEUE_KEY+['queue_pred']])
    result=pd.concat(pieces,ignore_index=True)
    if len(result)!=len(template):
        raise ValueError('Queue template/window mismatch')
    return result
