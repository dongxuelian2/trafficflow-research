"""Causal observation-only baselines for independently generated Queue scenes."""
import numpy as np


def causal_speed_queue_forecasts(history_speed,history_eligible,free_speed,condition):
    v=np.asarray(history_speed,float);eligible=np.asarray(history_eligible,bool)
    if v.ndim!=2 or v.shape[0]!=13 or eligible.shape!=v.shape:
        raise ValueError('Exactly thirteen legal history slots and matching quality required')
    if not np.isfinite(free_speed) or free_speed<=0 or condition not in ['queue_onset','queue_ongoing']:
        raise ValueError('Public free speed and condition required')
    valid=eligible&np.isfinite(v);steps=np.arange(13)[:,None]
    last=np.where(valid,steps,-1).max(axis=0);idx=np.maximum(last,0)
    current=np.where(last>=0,v[idx,np.arange(v.shape[1])],free_speed)
    recent=valid&np.broadcast_to(steps>=7,v.shape)
    x=np.broadcast_to(steps-12,v.shape);n=recent.sum(axis=0)
    sx=np.where(recent,x,0).sum(axis=0);sy=np.where(recent,v,0).sum(axis=0)
    sxx=np.where(recent,x*x,0).sum(axis=0);sxy=np.where(recent,x*v,0).sum(axis=0)
    denominator=n*sxx-sx*sx
    slope=np.divide(n*sxy-sx*sy,denominator,out=np.zeros(v.shape[1]),where=(n>1)&(denominator>0))
    mx=np.divide(sx,n,out=np.zeros(v.shape[1]),where=n>0)
    my=np.divide(sy,n,out=current.copy(),where=n>0)
    future=np.arange(1,7)[:,None]
    prediction={'persistence':np.broadcast_to(current,(6,v.shape[1])),
                'linear_trend':my[None]+slope[None]*(future-mx[None])}
    result={}
    for name,speed in prediction.items():
        queue=np.clip(speed,0,1.05*free_speed)<=.6*free_speed
        if condition=='queue_onset':queue[:5]=False
        result[name]=queue
    return result
