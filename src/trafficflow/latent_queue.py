"""Causal, identity-free features and decisions for Queue label-noise research.

Synthetic Gaussian soft targets are a training diagnostic, not hidden truth.
"""
import numpy as np
from scipy.special import ndtr


def gaussian_queue_targets(observed_speed,cutoff,sigma):
    """Flat-prior Gaussian soft teacher; known synthetic sigma only."""
    v,cut=np.broadcast_arrays(np.asarray(observed_speed,float),np.asarray(cutoff,float))
    if not np.isfinite(sigma) or sigma<0 or not np.isfinite(v).all() or not np.isfinite(cut).all():raise ValueError('Finite teacher and nonnegative noise required')
    return ((v<=cut).astype(float) if sigma==0 else ndtr((cut-v)/sigma)).astype(np.float32)


def causal_queue_features(speed,flow,valid,positions_km,free_speed,capacity,cutoff,ongoing):
    """Exactly13 legal slots, no time/identity/latent/quality-future inputs.

    Positional neighborhoods are statistical within5km, not physical edges.
    Returns horizon-major6xnodes rows. Speeds/flows share a validity mask.
    """
    v,q=np.asarray(speed,float),np.asarray(flow,float);mask=np.asarray(valid,bool);p=np.asarray(positions_km,float)
    if v.ndim!=2 or v.shape[0]!=13 or q.shape!=v.shape or mask.shape!=v.shape or p.shape!=(v.shape[1],) or not len(p):raise ValueError('Thirteen slots and aligned node geometry required')
    n=len(p);vf,cap,cut=[np.broadcast_to(np.asarray(a,float),(n,)) for a in [free_speed,capacity,cutoff]]
    if not np.isfinite(p).all() or any(not np.isfinite(a).all() or (a<=0).any() for a in [vf,cap,cut]) or not np.isfinite(v[mask]).all() or not np.isfinite(q[mask]).all():raise ValueError('Finite legal values and positive FD required')
    if not isinstance(ongoing,(bool,np.bool_)):raise ValueError('Published condition must be boolean')
    vr=np.where(mask,v,0.)/vf;qr=np.where(mask,q,0.)/cap;steps=np.arange(13)[:,None];count=mask.sum(axis=0)
    last=np.where(mask,steps,-1).max(axis=0);safe=np.maximum(last,0);columns=np.arange(n)
    current=np.where(last>=0,vr[safe,columns],1.);current_q=np.where(last>=0,qr[safe,columns],0.)
    mean=np.divide(vr.sum(axis=0),count,out=np.ones(n),where=count>0);meanq=np.divide(qr.sum(axis=0),count,out=np.zeros(n),where=count>0)
    variance=np.maximum(np.divide((vr*vr).sum(axis=0),count,out=np.ones(n),where=count>0)-mean*mean,0.)
    minimum=np.where(count>0,np.where(mask,vr,np.inf).min(axis=0),1.)
    recent=mask&(steps>=6);t=np.broadcast_to(steps-12,v.shape);nr=recent.sum(axis=0);sx=np.where(recent,t,0).sum(axis=0);sxx=np.where(recent,t*t,0).sum(axis=0);den=nr*sxx-sx*sx
    slopes=[]
    for values in [vr,qr]:
        sy=np.where(recent,values,0).sum(axis=0);sxy=np.where(recent,t*values,0).sum(axis=0)
        slopes.append(np.divide(nr*sxy-sx*sy,den,out=np.zeros(n),where=(nr>1)&(den>0)))
    stats=np.column_stack([current,mean,minimum,np.sqrt(variance),slopes[0],np.where(last>=0,12-last,13)/12,count/13,current_q,meanq,slopes[1]])
    delta=p[None,:]-p[:,None];neighbors=[]
    for allowed in [(delta<0)&(delta>=-5),(delta>0)&(delta<=5)]:
        counts=allowed.sum(axis=1)
        for values,default in [(current,1.),(slopes[0],0.),(count/13,0.)]:neighbors.append(np.divide(allowed@values,counts,out=np.full(n,default),where=counts>0))
    local=np.concatenate([vr.T,qr.T,mask.T.astype(float),stats,np.column_stack(neighbors),np.full((n,1),float(ongoing)),(cut/vf)[:,None]],axis=1)
    result=np.concatenate([np.column_stack([local,np.full(n,lead/30)]) for lead in [5,10,15,20,25,30]])
    if not np.isfinite(result).all():raise ValueError('Nonfinite causal features')
    return result.astype(np.float32)


def latent_queue_decisions(probability,ongoing,thresholds,force=False):
    """Full horizon decisions before the evaluator's future mask is known."""
    p=np.asarray(probability,float);c=np.asarray(ongoing,bool)
    if p.ndim!=3 or p.shape[1]!=6 or c.shape!=(len(p),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Finite window6node probabilities required')
    t=np.where(c,thresholds['ongoing'],thresholds['onset'])
    if not np.isfinite(t).all() or ((t<0)|(t>1)).any():raise ValueError('Probability thresholds required')
    allowed=np.ones(p.shape,bool);allowed[~c,:5]=False;prediction=(p>=t[:,None,None])&allowed
    if force:
        empty=~prediction.reshape(len(p),-1).any(axis=1);score=np.where(allowed,p,-np.inf).reshape(len(p),-1)
        prediction.reshape(len(p),-1)[np.flatnonzero(empty),score[empty].argmax(axis=1)]=True
    return prediction


def window_ious(labels,prediction,available):
    y,p,m=np.asarray(labels,bool),np.asarray(prediction,bool),np.asarray(available,bool)
    if y.ndim!=3 or y.shape!=p.shape or m.shape!=y.shape:raise ValueError('Window label/prediction/domain mismatch')
    inter=(m&y&p).sum(axis=(1,2));union=(m&(y|p)).sum(axis=(1,2));score=np.divide(inter,union,out=np.ones(len(y)),where=union>0)
    score[~m.any(axis=(1,2))]=0.
    return score
