"""Observable physics diagnostics and optional conservative flow adjustment.

No value returned by this module is a formal Task 3 score.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import lsq_linear
from .data import VALUES

def reconstruct(frame, prediction):
    d=frame.copy()
    if 'pct_observed' in d:
        d.loc[pd.to_numeric(d.pct_observed,errors='coerce')<75,VALUES]=np.nan
    p=prediction[['timestamp','station_id','link_id']+VALUES]
    d=d.merge(p,on=['timestamp','station_id','link_id'],how='left',suffixes=('','_pred'),validate='one_to_one')
    target=d.speed_kmh_pred.notna()
    for c in VALUES:
        d.loc[target,c]=d.loc[target,c+'_pred']
    return d.drop(columns=[c+'_pred' for c in VALUES]),target.to_numpy()

def conservation_operator(times,links,v,q,net,topo,on,off):
    n,m=q.shape
    li={link:j for j,link in enumerate(links)}
    rr,cc,vv,rhs=[],[],[],[]
    tstamp=pd.to_datetime(times,utc=True)
    consecutive=np.diff(tstamp.as_unit('ns').asi8)==300*10**9
    offset=0
    for j,link in enumerate(links):
        if link not in topo.index:
            continue
        inc=[s.strip() for s in str(topo.loc[link].get('incoming_link_ids','')).split(';') if s.strip()]
        out=[s.strip() for s in str(topo.loc[link].get('outgoing_link_ids','')).split(';') if s.strip()]
        if not inc or not out or not set(inc+out).issubset(li):
            continue
        required=[j]+[li[s] for s in inc+out]
        valid=consecutive&np.isfinite(q[:-1,required]).all(axis=1)&np.isfinite(q[1:,j])
        valid &= np.isfinite(v[:-1,j])&np.isfinite(v[1:,j])&np.isfinite(on[:-1,j]+off[:-1,j])
        t=np.flatnonzero(valid)
        if not len(t):continue
        row=offset+np.arange(len(t));offset+=len(t)
        length=float(net.loc[link,'length_km'])
        terms=[((t+1)*m+j,length/np.maximum(v[t+1,j],1.)),(t*m+j,-length/np.maximum(v[t,j],1.))]
        terms += [(t*m+li[s],np.full(len(t),-1/12)) for s in inc]
        terms += [(t*m+li[s],np.full(len(t),1/12)) for s in out]
        for column,value in terms:
            rr.append(row);cc.append(column);vv.append(value)
        rhs.append((on[t,j]-off[t,j])/12)
    B=sparse.csr_matrix((np.concatenate(vv) if vv else [],
                       (np.concatenate(rr) if rr else [],np.concatenate(cc) if cc else [])),shape=(offset,n*m))
    return B,np.concatenate(rhs) if rhs else np.empty(0)

def grids(frame,net):
    times=sorted(frame.timestamp.unique());links=frame.link_id.drop_duplicates().tolist()
    q=frame.pivot_table(index='timestamp',columns='link_id',values='flow_vph',aggfunc='mean').reindex(index=times,columns=links).to_numpy(float)
    v=frame.pivot_table(index='timestamp',columns='link_id',values='speed_kmh',aggfunc='mean').reindex(index=times,columns=links).to_numpy(float)
    return times,links,v,q

def diagnose(frame,prediction,net,topo,ramps=None):
    full,_=reconstruct(frame,prediction)
    times,links,v,q=grids(full,net)
    lanes=net.lanes.reindex(links).to_numpy(float)
    cap=net.capacity_vph.reindex(links).to_numpy(float)
    vf=net.free_speed_kmh.reindex(links).to_numpy(float)
    jam=net.k_jam.reindex(links).to_numpy(float)
    crit=net.critical_density.reindex(links).to_numpy(float)
    k=q/np.maximum(v,1.)
    qfd=np.where(k<=crit,vf*k,(cap/np.maximum(jam-crit,1e-9))*np.maximum(jam-k,0))
    good=np.isfinite(q)&np.isfinite(v)
    fd=float(np.abs((q-qfd)/lanes)[good].mean()) if good.any() else None
    if ramps is None:
        on,off=np.full_like(q,np.nan),np.full_like(q,np.nan)
    else:
        on,off=ramps
    B,rhs=conservation_operator(times,links,v,q,net,topo,on,off)
    residual=B@np.nan_to_num(q).ravel()-rhs
    return {'kind':'public_physics_diagnostic_only','formal_S_physics':None,
       'FD_mean_absolute_residual_per_lane':fd,'observable_transitions':len(rhs),
       'possible_transitions':max(0,(len(times)-1)*len(links)),
       'conservation_RMS_vehicles':float(np.sqrt(np.mean(residual**2))) if len(rhs) else None,
       'conservation_MAE_vehicles':float(np.abs(residual).mean()) if len(rhs) else None,
       'below_50_vph_share':float((prediction.flow_vph.to_numpy(float)<50).mean()) if len(prediction) else None}

def refine(frame,prediction,net,topo,ramps,strength=.01,fd_bounds=False):
    if strength<0:
        raise ValueError('Negative physics strength')
    full,_=reconstruct(frame,prediction)
    times,links,v,q=grids(full,net)
    flat=pd.MultiIndex.from_product([times,links])
    target=pd.MultiIndex.from_frame(prediction[['timestamp','link_id']]).unique()
    indices=flat.get_indexer(target)
    if (indices<0).any():
        raise ValueError('Physics target missing')
    B,rhs=conservation_operator(times,links,v,q,net,topo,*ramps)
    result=prediction.copy()
    if len(rhs) and strength>0:
        q0=np.nan_to_num(q).ravel()
        Bj=B[:,indices]
        boundary=rhs-B@q0+Bj@q0[indices]
        scale=np.tile(net.lanes.reindex(links).to_numpy(float)*600,len(times))[indices]
        system=sparse.vstack([sparse.eye(len(indices)),np.sqrt(strength)*Bj@sparse.diags(scale)],format='csr')
        y=np.r_[q0[indices]/scale,np.sqrt(strength)*boundary]
        solved=lsq_linear(system,y,bounds=(0,np.inf),tol=1e-5,lsmr_tol=1e-6,max_iter=200,lsmr_maxiter=2000)
        if not solved.success:
            raise RuntimeError('Physics solve did not converge')
        map_q=pd.Series(solved.x*scale,index=target)
        result['flow_vph']=map_q.reindex(pd.MultiIndex.from_frame(result[['timestamp','link_id']])).to_numpy()
    if fd_bounds:
        # Ordinary physical bounds; no empty-road-floor clipping to manipulate the evaluator.
        result['speed_kmh']=np.minimum(result.speed_kmh.to_numpy(float),result.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
        result['flow_vph']=np.minimum(result.flow_vph.to_numpy(float),result.link_id.map(net.capacity_vph).to_numpy(float)*1.05)
    return result
