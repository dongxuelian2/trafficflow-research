"""Exact State supervision exposure and pure-CPU causal input contracts."""
import numpy as np
import pandas as pd
from .data import eligible


def state_coordinates(dense_origins,event_origins,fit_dates,batches):
    dates={date:i for i,date in enumerate(fit_dates)};event=[]
    for text in event_origins:
        t=pd.Timestamp(str(text))
        if t.tzinfo is None or t.date().isoformat() not in dates or t.minute%5 or t.second:raise ValueError('Legal five-minute FIT event origin required')
        event.append((dates[t.date().isoformat()],t.hour*12+t.minute//5))
    event=np.asarray(event,np.int32);warm=np.asarray(dense_origins,np.int32)[batches['warm']];fine=event[batches['event']]
    if warm.shape!=(400,8,2) or fine.shape!=(800,4,2):raise ValueError('Original matched400x8/800x4 schedule required')
    return {'warm':warm,'event':fine}


def context_exposure(schedule):
    keys=np.concatenate([a.reshape(-1,2) for a in schedule.values()]);coordinates=np.unique(keys,axis=0);lookup={tuple(c):i for i,c in enumerate(coordinates)};counts=np.zeros((len(coordinates),2),np.int64)
    for stage,key in enumerate(['warm','event']):
        for coordinate in schedule[key].reshape(-1,2):counts[lookup[tuple(coordinate)],stage]+=1
    mass=counts[:,0].astype(float)/schedule['warm'].shape[1]+counts[:,1].astype(float)/schedule['event'].shape[1]
    return coordinates,counts,mass


def nominal_teacher_weights(context_mass,domain):
    """Sum of per-update window/field-average coefficients, excluding common .1."""
    domain=np.asarray(domain,bool);count=domain.sum(axis=2).clip(min=1);weights=np.asarray(context_mass,float)[:,None,None]/count[:,:,None]*domain
    normalized=weights.copy()
    for field in range(2):
        use=domain[:,field]
        if use.any():normalized[:,field][use]/=weights[:,field][use].mean()
    return weights,normalized


def current_weighted_mse(a,prediction):
    domain=np.asarray(a['current_domain'],bool);counts=domain.sum(axis=(0,2));error=np.asarray(prediction,float)-np.asarray(a['truth_current'],float)
    return float(sum(w*np.sum(error[:,i][domain[:,i]]**2)/counts[i] for i,w in enumerate([.54,.46]))) if (counts>0).all() else None


def causal_state_input(history,links,net,origin):
    """Masked observations only; no future/current truth or provided event label."""
    origin=pd.Timestamp(origin);links=list(map(str,links))
    if origin.tzinfo is None:raise ValueError('Timezone-aware State origin required')
    origin=origin.tz_convert('UTC')
    timestamp=pd.to_datetime(history.timestamp,utc=True);slots=pd.date_range(origin-pd.Timedelta(minutes=60),origin,freq='5min')
    if not timestamp.isin(slots).all():raise ValueError('State input outside causal13slot grid')
    frame=history.copy();frame['_time']=timestamp;frame.loc[~eligible(frame),['speed_kmh','flow_vph','occupancy']]=np.nan
    def grid(column):return frame.pivot_table(index='_time',columns='link_id',values=column,aggfunc='mean').reindex(index=slots,columns=links).to_numpy(float)
    v=grid('speed_kmh')/25.;q=grid('flow_vph')/net.lanes.reindex(links).to_numpy(float)/600.;o=grid('occupancy')/.25;phase=2*np.pi*np.asarray(slots.hour*60+slots.minute)/1440.;shape=v.shape
    x=np.stack([np.nan_to_num(v),np.nan_to_num(q),np.isfinite(v),np.isfinite(q),np.nan_to_num(o),np.isfinite(o),np.broadcast_to(np.sin(phase)[:,None],shape),np.broadcast_to(np.cos(phase)[:,None],shape),np.zeros(shape)],axis=-1).astype(np.float32)
    cutoff=net.v_cut.reindex(links).to_numpy(np.float32)/25.;queue=(x[:12,:,0]<=cutoff)&(x[:12,:,2]>.5);x[:,:,8]=bool((queue.sum(axis=0)>=2).any())
    if not np.isfinite(x).all():raise ValueError('Finite causal State channels required')
    return x
