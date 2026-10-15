"""Causal onset set-size and threshold controls around frozen classifiers."""
import numpy as np
import pandas as pd
from .queue import QueueModel,queue_features
from .onset_rank_queue import onset_topk_decisions


SUMMARY_NAMES=['links','current_gap_min','current_gap_q10','current_gap_mean','current_gap_std','current_queue_fraction',
    'trend_gap_min','trend_gap_q10','trend_gap_mean','trend_gap_std','trend_queue_fraction','slope_min','slope_mean','slope_std',
    'flow_ratio_mean','flow_ratio_max','flow_ratio_missing_fraction','speed_std_mean','speed_std_max','coverage_mean','age_mean','sin_tod','cos_tod','weekday']


def onset_summaries(features,window,lead,links):
    """24 deterministic whole-window summaries, no labels/quality-future input."""
    w,t,l=map(np.asarray,[window,lead,links]);l=l.astype(str)
    if len(features)!=len(w):raise ValueError('Feature/domain mismatch')
    onset_topk_decisions(np.zeros(len(w)),w,t,l,1)
    keys=[];rows=[]
    for key,ids in pd.DataFrame({'window':w}).groupby('window',sort=False).indices.items():
        final=ids[t[ids]==30];final=final[np.argsort(l[final],kind='stable')];x=features.iloc[final]
        def vals(field):return x[field].to_numpy(float)
        def finite(v):return v[np.isfinite(v)]
        def stat(v,func):
            z=finite(v);return float(func(z)) if len(z) else 0.
        current=vals('current')-vals('cut_ratio');trend=vals('trend_forecast')-vals('cut_ratio');slope=vals('slope');flow=vals('flow_capacity_ratio');sd=vals('speed_std')
        row=[len(final),stat(current,np.min),stat(current,lambda a:np.quantile(a,.1)),stat(current,np.mean),stat(current,np.std),float((current<=0).mean()),
            stat(trend,np.min),stat(trend,lambda a:np.quantile(a,.1)),stat(trend,np.mean),stat(trend,np.std),float((trend<=0).mean()),
            stat(slope,np.min),stat(slope,np.mean),stat(slope,np.std),stat(flow,np.mean),stat(flow,np.max),float((~np.isfinite(flow)).mean()),
            stat(sd,np.mean),stat(sd,np.max),stat(vals('coverage'),np.mean),stat(vals('age'),np.mean),float(x.sin_tod.iloc[0]),float(x.cos_tod.iloc[0]),float(x.weekday.iloc[0])]
        keys.append(str(key));rows.append(row)
    result=np.asarray(rows,float)
    if result.shape!=(len(keys),24) or not np.isfinite(result).all():raise ValueError('Finite onset summaries required')
    return np.array(keys),result


def fit_count_state(x,positive,eligible,links,adjust=False,intercept_only=False,alpha=10.):
    """Fit log1p observed count, optionally an explicit MCAR coverage correction."""
    x=np.asarray(x,float);y,m,n=map(lambda a:np.asarray(a,float),[positive,eligible,links])
    if x.ndim!=2 or x.shape[1]!=24 or any(a.shape!=(len(x),) for a in [y,m,n]) or not np.isfinite(x).all() or not all(np.isfinite(a).all() for a in [y,m,n]) or (m<=0).any() or (y<0).any() or (y>m).any() or (m>n).any() or alpha<=0:raise ValueError('Valid fit-only count targets required')
    target=y*n/m if adjust else y;z=np.log1p(target);mu=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-12]=1.;design=(x-mu)/scale;intercept=float(z.mean())
    coefficient=np.zeros(x.shape[1]) if intercept_only else np.linalg.solve(design.T@design+alpha*np.eye(x.shape[1]),design.T@(z-intercept))
    return {'mean':mu.tolist(),'scale':scale.tolist(),'coefficient':coefficient.tolist(),'intercept':intercept,'alpha':alpha,'coverage_adjusted':bool(adjust),'intercept_only':bool(intercept_only),'n_fit_windows':len(x)}


def predict_set_sizes(state,x,links):
    x=np.asarray(x,float);n=np.asarray(links,int)
    if x.ndim!=2 or x.shape[1]!=24 or n.shape!=(len(x),) or (n<1).any() or not np.isfinite(x).all():raise ValueError('Finite legal summaries and public sizes required')
    z=((x-np.asarray(state['mean']))/np.asarray(state['scale']))@np.asarray(state['coefficient'])+state['intercept']
    value=np.expm1(np.clip(z,0,np.log1p(n)))
    return np.clip(np.floor(value+.5).astype(int),1,n)


def onset_threshold_decisions(probability,window,lead,links,threshold,force):
    p=np.asarray(probability,float);w,t,l=map(np.asarray,[window,lead,links])
    if not np.isfinite(threshold) or not 0<=threshold<=1 or ((p<0)|(p>1)).any():raise ValueError('Classifier probabilities/threshold required')
    top=onset_topk_decisions(p,w,t,l,1);out=((p>=threshold)&(t==30)).astype(int)
    if force:
        for ids in pd.DataFrame({'window':w}).groupby('window',sort=False).indices.values():
            if not out[ids].any():out[ids]=top[ids]
    return out


def onset_count_decisions(score,window,lead,links,sizes):
    s=np.asarray(score,float);w,t,l=map(np.asarray,[window,lead,links]);out=np.zeros(len(s),int)
    onset_topk_decisions(s,w,t,l,1)
    for key,ids in pd.DataFrame({'window':w}).groupby('window',sort=False).indices.items():out[ids]=onset_topk_decisions(s[ids],w[ids],t[ids],l[ids],int(sizes[str(key)]))
    return out


class OnsetSetDecoderModel(QueueModel):
    def __init__(self,base,rule,setting=None,count_state=None):
        super().__init__('onset_set_decoder',base.seed,600,4)
        if base.kind!='classifier' or rule not in ['threshold','count_prior','count_observed','count_adjusted']:raise ValueError('Frozen onset classifier and known decoder required')
        if (rule=='threshold')!=(setting is not None) or (rule!='threshold' and count_state is None):raise ValueError('Decoder parameters required')
        self.base=base;self.links=list(base.links);self.rule=rule;self.setting=setting;self.count_state=count_state

    def predict_window(self,history,target,net,topology,origin,condition):
        if condition=='queue_ongoing':return self.base.incumbent.predict_window(history,target,net,topology,origin,condition)
        if condition!='queue_onset':raise ValueError('Published condition required')
        score,lead,lookup,links=self.base._onset(history,target,net,topology,origin);window=np.full(len(score),'public_window')
        if self.rule=='threshold':pred=onset_threshold_decisions(score,window,lead,links,self.setting['threshold'],self.setting['force'])
        else:
            origin=pd.Timestamp(origin).tz_convert('UTC');times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
            full=pd.MultiIndex.from_product([times,self.links],names=['timestamp','link_id']).to_frame(index=False);reference=self.base.incumbent.baseline
            x=queue_features(history,full,net,topology,origin,allow_origin=True,include_occupancy=getattr(reference,'uses_occupancy_features',False))
            if getattr(reference,'uses_published_condition',False):x['condition_ongoing']=0.
            keys,summary=onset_summaries(x,window,lead,links);k=predict_set_sizes(self.count_state,summary,summary[:,0]);pred=onset_count_decisions(score,window,lead,links,dict(zip(keys,k)))
        out=target.copy();out['queue_pred']=pred[lookup];return out

    def save(self,path):
        cache=self.base._boosters;self.base._boosters=None
        try:super().save(path)
        finally:self.base._boosters=cache
