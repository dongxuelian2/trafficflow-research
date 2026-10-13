"""Leakage-safe offline state reconstruction from masked observations."""
from __future__ import annotations
from dataclasses import dataclass
import pickle
import numpy as np
import pandas as pd
from .data import Release, STATE_KEY, VALUES, eligible

@dataclass
class Profile:
    slots: pd.DataFrame
    fallback: pd.DataFrame

    @classmethod
    def fit(cls, frames):
        sums, counts = None, None
        for raw in frames:
            d = raw.loc[eligible(raw), ['timestamp','link_id']+VALUES].copy()
            dt = pd.to_datetime(d.timestamp, utc=True)
            d['slot'] = dt.dt.weekday*288+dt.dt.hour*12+dt.dt.minute//5
            g = d.groupby(['link_id','slot'])[VALUES]
            s, n = g.sum(), g.count()
            sums = s if sums is None else sums.add(s, fill_value=0)
            counts = n if counts is None else counts.add(n, fill_value=0)
        if sums is None or not len(sums):
            raise ValueError('No eligible train observations for profile')
        fallback = sums.groupby(level=0).sum()/counts.groupby(level=0).sum()
        return cls(sums/counts, fallback)

    def predict(self, frame, net):
        dt = pd.to_datetime(frame.timestamp, utc=True)
        slot = dt.dt.weekday*288+dt.dt.hour*12+dt.dt.minute//5
        index = pd.MultiIndex.from_arrays([frame.link_id, slot])
        v = self.slots.reindex(index).to_numpy(float)
        fb = self.fallback.reindex(frame.link_id).to_numpy(float)
        v = np.where(np.isfinite(v), v, fb)
        default = np.c_[frame.link_id.map(net.free_speed_kmh).to_numpy(float),
                        frame.link_id.map(net.capacity_vph).to_numpy(float)*.4]
        return np.where(np.isfinite(v), v, default)

def neighbors(topo: pd.DataFrame, links: list[str], direction: str) -> list[list[int]]:
    # Traverse latent mainline links rather than pretending consecutive sensor IDs are adjacent.
    measured = {x:i for i,x in enumerate(links)}
    field = f'{direction}_link_ids'
    result = []
    for link in links:
        found, visited, pending = [], {link}, [link]
        for _ in range(32):
            nxt = []
            for node in pending:
                if node not in topo.index or field not in topo:
                    continue
                for other in str(topo.loc[node,field]).split(';'):
                    other = other.strip()
                    if not other or other in visited:
                        continue
                    visited.add(other)
                    if other in measured:
                        found.append(measured[other])
                    elif other in topo.index:
                        nxt.append(other)
            if not nxt:
                break
            pending = nxt
        result.append(sorted(set(found)))
    return result

def shifted(a, lag):
    out = np.full_like(a, np.nan)
    if lag > 0:
        out[lag:] = a[:-lag]
    elif lag < 0:
        out[:lag] = a[-lag:]
    else:
        out[:] = a
    return out

def temporal(a, max_gap=60):
    out, age = np.full_like(a,np.nan), np.full_like(a,1440.)
    t = np.arange(len(a))
    for j in range(a.shape[1]):
        good = np.flatnonzero(np.isfinite(a[:,j]))
        if not len(good):
            continue
        out[:,j] = np.interp(t,good,a[good,j])
        at = np.searchsorted(good,t)
        left = good[np.maximum(at-1,0)]
        right = good[np.minimum(at,len(good)-1)]
        age[:,j] = np.minimum(np.abs(t-left),np.abs(t-right))*5
        out[age[:,j]>max_gap,j] = np.nan
    return out,age

def kalman_residual(observed, prior, process=.025, measurement=.08):
    # Random-walk residual filter and Rauch--Tung--Striebel smoothing, offline Task 1 only.
    n,m = observed.shape
    out = np.empty_like(observed)
    for j in range(m):
        x, p = np.zeros(n),np.zeros(n)
        mean,var = 0.,1.
        for t in range(n):
            var += process
            if np.isfinite(observed[t,j]):
                gain = var/(var+measurement)
                mean += gain*(observed[t,j]-prior[t,j]-mean)
                var *= 1-gain
            x[t],p[t] = mean,var
        for t in range(n-2,-1,-1):
            gain=p[t]/(p[t]+process)
            x[t] += gain*(x[t+1]-x[t])
        out[:,j]=prior[:,j]+x
    return out

class StateFeatures:
    def __init__(self, frame: pd.DataFrame, net: pd.DataFrame, topo: pd.DataFrame, profile: Profile):
        self.frame = frame.copy()
        self.net = net
        self.links = frame.link_id.drop_duplicates().tolist()
        if not set(self.links).issubset(net.index):
            raise ValueError('State contains unknown network links')
        if 'milepost' in frame:
            miles=frame.groupby('link_id').milepost.mean()
            if miles.isna().any() or miles.duplicated().any():
                raise ValueError('Spatial interpolation requires unique finite link mileposts')
            self.links=sorted(self.links,key=lambda x:miles[x])
            positions=miles.reindex(self.links).to_numpy(float)*1.609344
        else:
            positions=np.cumsum(net.reindex(self.links).length_km.to_numpy(float))
        self.positions=positions
        self.times=sorted(frame.timestamp.unique())
        grid_index=pd.MultiIndex.from_product([self.times,self.links],names=['timestamp','link_id'])
        grid=grid_index.to_frame(index=False)
        shape=(len(self.times),len(self.links))
        lanes=net.lanes.reindex(self.links).to_numpy(float)
        prior=profile.predict(grid,net)
        prior[:,1]/=np.tile(lanes,len(self.times))
        self.prior=prior.reshape(*shape,2)
        known=frame.copy()
        known.loc[~eligible(frame),VALUES]=np.nan
        self.raw=np.stack([known.pivot_table(index='timestamp',columns='link_id',values=c,aggfunc='mean')
                               .reindex(index=self.times,columns=self.links).to_numpy(float) for c in VALUES],axis=-1)
        self.raw[:,:,1]/=lanes
        self.temp=np.empty_like(self.raw)
        self.space=np.empty_like(self.raw)
        self.smooth=np.empty_like(self.raw)
        features={}
        incoming,outgoing=neighbors(topo,self.links,'incoming'),neighbors(topo,self.links,'outgoing')
        for channel,name,scale in [(0,'v',25.),(1,'q',600.)]:
            a=self.raw[:,:,channel]
            tmp,age=temporal(a)
            spa=np.full_like(a,np.nan)
            for t in range(len(self.times)):
                good=np.flatnonzero(np.isfinite(a[t]))
                if len(good):
                    spa[t]=np.interp(positions,positions[good],a[t,good])
            p=self.prior[:,:,channel]
            self.temp[:,:,channel]=tmp
            self.space[:,:,channel]=spa
            self.smooth[:,:,channel]=kalman_residual(a/scale,p/scale)*scale
            features[name+'_prior']=p.ravel()/scale
            features[name+'_temporal']=tmp.ravel()/scale
            features[name+'_spatial']=spa.ravel()/scale
            features[name+'_age']=age.ravel()/60
            features[name+'_missing']=(~np.isfinite(a)).ravel().astype(float)
            for lag in [-3,-2,-1,1,2,3]:
                features[name+('_lead' if lag<0 else '_lag')+str(abs(lag))]=shifted(a,lag).ravel()/scale
            for ns,label in [(incoming,'up'),(outgoing,'down')]:
                neighbor=np.full_like(a,np.nan)
                for j,ids in enumerate(ns):
                    if ids:
                        subset=pd.DataFrame(a[:,ids]).mean(axis=1).to_numpy()
                        neighbor[:,j]=subset
                features[name+'_'+label]=neighbor.ravel()/scale
                features[name+'_'+label+'_lag']=shifted(neighbor,1).ravel()/scale
                features[name+'_'+label+'_lead']=shifted(neighbor,-1).ravel()/scale
        # Distance-weighted temporal/spatial estimate; missing local evidence falls back to train profile.
        age=np.stack([temporal(self.raw[:,:,c])[1] for c in range(2)],axis=-1)
        wt=1/(1+age/10)
        ts=np.where(np.isfinite(self.temp),wt,0)
        ss=np.where(np.isfinite(self.space),.35,0)
        base=(np.nan_to_num(self.temp)*ts+np.nan_to_num(self.space)*ss)/np.maximum(ts+ss,1e-9)
        self.base=np.where((ts+ss)>0,base,self.prior)
        dt=pd.to_datetime(grid.timestamp,utc=True)
        minute=dt.dt.hour*60+dt.dt.minute
        features['sin_tod']=np.sin(2*np.pi*minute/1440)
        features['cos_tod']=np.cos(2*np.pi*minute/1440)
        features['weekday']=dt.dt.weekday
        features['position']=np.tile(positions,len(self.times))
        for c in ['lanes','free_speed_kmh','capacity_vph','length_km']:
            features[c]=grid.link_id.map(net[c]).to_numpy(float)
        self.grid=grid
        self.features=pd.DataFrame(features).astype(np.float32)
        # Station targets are mapped back from a link grid without altering original keys.
        self.row_index=grid_index.get_indexer(pd.MultiIndex.from_frame(frame[['timestamp','link_id']]))

    def lookup(self, targets: pd.DataFrame):
        idx=pd.MultiIndex.from_frame(self.grid).get_indexer(pd.MultiIndex.from_frame(targets[['timestamp','link_id']]))
        if (idx<0).any():
            raise ValueError('Template target absent from observation grid')
        return idx

class StateModel:
    def __init__(self, profile: Profile, method='interpolation', seed=2026, iterations=250, threads=4):
        self.profile,self.method,self.seed=profile,method,seed
        self.iterations,self.threads=iterations,threads
        self.models=[]

    def fit(self, x: pd.DataFrame, y: np.ndarray, base: np.ndarray):
        residual=(y-base)/np.array([25.,600.])
        for c in range(2):
            if self.method=='lightgbm':
                from lightgbm import LGBMRegressor
                model=LGBMRegressor(n_estimators=self.iterations,learning_rate=.035,num_leaves=31,
                   min_child_samples=80,reg_lambda=5.,colsample_bytree=.9,random_state=self.seed,
                   n_jobs=self.threads,verbosity=-1,deterministic=True,force_col_wise=True)
                model.fit(x,residual[:,c])
            elif self.method=='catboost':
                from catboost import CatBoostRegressor
                model=CatBoostRegressor(iterations=self.iterations,depth=6,learning_rate=.04,l2_leaf_reg=8,
                       random_seed=self.seed,thread_count=self.threads,verbose=False,allow_writing_files=False)
                model.fit(x.fillna(-999),residual[:,c])
            else:
                raise ValueError('Training is supported for LightGBM and CatBoost')
            self.models.append(model)
        return self

    def predict(self, features: StateFeatures, template: pd.DataFrame):
        idx=features.lookup(template)
        if self.method=='historical':
            y=features.prior.reshape(-1,2)[idx].copy()
        elif self.method=='kalman':
            y=features.smooth.reshape(-1,2)[idx].copy()
        else:
            y=features.base.reshape(-1,2)[idx].copy()
            if self.method in ['lightgbm','catboost']:
                if len(self.models)!=2:
                    raise ValueError('Model is not trained')
                x=features.features.iloc[idx]
                if self.method=='catboost':
                    x=x.fillna(-999)
                y+=np.stack([m.predict(x) for m in self.models],axis=-1)*np.array([25.,600.])
            elif self.method!='interpolation':
                raise ValueError('Unknown state method')
        y=np.maximum(y,0)
        lane=template.link_id.map(features.net.lanes).to_numpy(float)
        y[:,1]*=lane
        if not np.isfinite(y).all():
            raise ValueError('Non-finite state predictions')
        result=template[STATE_KEY].copy()
        result[VALUES]=y
        return result

    def save(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self,f)

    @classmethod
    def load(cls,path):
        # Only load artifacts produced by this trusted local pipeline.
        with path.open('rb') as f:
            model=pickle.load(f)
        if not isinstance(model,cls):
            raise ValueError('Invalid model artifact')
        return model
