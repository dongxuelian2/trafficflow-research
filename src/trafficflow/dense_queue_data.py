"""Indexed causal Queue histories and fit-only dense future teachers."""
import numpy as np
import pandas as pd
from .data import eligible


def prepare_dense_day(raw,masked,links,net):
    times=sorted(raw.timestamp.astype(str).unique());links=list(map(str,links))
    if len(times)!=288 or len(set(t[:10] for t in times))!=1 or sorted(masked.timestamp.astype(str).unique())!=times:raise ValueError('Matching complete UTC raw/masked fit day required')
    if raw.link_id.astype(str).drop_duplicates().tolist()!=links:raise ValueError('Dense Queue link order differs')
    lane=net.lanes.reindex(links).to_numpy(float);cut=net.v_cut.reindex(links).to_numpy(float)
    def grids(frame):
        frame=frame.copy();frame.loc[~eligible(frame),['speed_kmh','flow_vph','occupancy']]=np.nan
        def grid(column):return frame.pivot_table(index='timestamp',columns='link_id',values=column).reindex(index=times,columns=links).to_numpy(float)
        v,q,o=[grid(c) for c in ['speed_kmh','flow_vph','occupancy']];channels=np.stack([np.nan_to_num(v/25),np.nan_to_num(q/lane/600),np.isfinite(v),np.isfinite(q),np.nan_to_num(o/.25),np.isfinite(o)],axis=-1).astype(np.float32)
        return channels,v,np.isfinite(v)
    r,v,available=grids(raw);m,_,_=grids(masked);minute=np.arange(288)*5;phase=2*np.pi*minute/1440;time=np.stack([np.sin(phase),np.cos(phase)],axis=-1).astype(np.float32)
    if not np.isfinite(r).all() or not np.isfinite(m).all():raise ValueError('Dense channels must be finite')
    return {'raw_channels':r,'masked_channels':m,'time_channels':time,'truth_speed':r[:,:,0].copy(),'truth_queue':(v<=cut).astype(np.float32),'available':available,'timestamps':np.array(times),'links':np.array(links)}


def dense_origins(pack):
    quality=np.asarray(pack['available'],bool);times=np.arange(12,len(quality)-6);history=times[:,None]+np.arange(-12,0)
    return times[quality[history].mean(axis=(1,2))>=.7]


def dense_window(pack,at,ongoing=None):
    if not isinstance(at,(int,np.integer)) or at<12 or at+6>=len(pack['available']):raise ValueError('One-hour history and within-day six-step target required')
    inferred=bool((pack['truth_queue'][at-12:at].sum(axis=0)>=2).any())
    if ongoing is not None and not isinstance(ongoing,(bool,np.bool_)):raise ValueError('Causal/provided condition must be boolean')
    condition=inferred if ongoing is None else bool(ongoing);raw=np.concatenate([pack['raw_channels'][at-12:at],pack['masked_channels'][at:at+1]],axis=0);shape=raw.shape[:2]
    time=np.broadcast_to(pack['time_channels'][at-12:at+1,None,:],(*shape,2));x=np.concatenate([raw,time,np.full((*shape,1),condition,np.float32)],axis=-1)
    if x.shape!=(13,raw.shape[1],9) or not np.isfinite(x).all():raise ValueError('Finite causal 13-slot input required')
    return {'x':x,'labels':pack['truth_queue'][at+1:at+7].copy(),'truth_speed':pack['truth_speed'][at+1:at+7].copy(),'available':pack['available'][at+1:at+7].copy(),'ongoing':condition,'inferred_ongoing':inferred}
