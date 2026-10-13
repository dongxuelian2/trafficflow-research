"""Learned density-increment assimilation with exact visible observations."""
import numpy as np


def assimilate_density(prior,observed,target,increment,pair_valid,strength):
    """Minimize prior error + strength * increment error, only freeing targets.

    All arrays are time by link, except increments/pair_valid have one fewer
    time step. This is a supervised transition diagnostic, not official flux.
    """
    prior=np.asarray(prior,float)
    observed=np.asarray(observed,float)
    target=np.asarray(target,bool)
    increment=np.asarray(increment,float)
    pair_valid=np.asarray(pair_valid,bool)
    if prior.shape!=observed.shape or target.shape!=prior.shape or increment.shape!=(prior.shape[0]-1,prior.shape[1]) or pair_valid.shape!=increment.shape:
        raise ValueError('Density assimilation shape mismatch')
    if not np.isfinite(strength) or strength<0 or not np.isfinite(prior[target]).all():
        raise ValueError('Invalid density prior or strength')
    visible=np.isfinite(observed)
    valid=visible|target
    if (pair_valid&~(valid[:-1]&valid[1:])).any() or not np.isfinite(increment[pair_valid]).all():
        raise ValueError('Unavailable anchor or increment in enabled pair')
    diagonal=np.ones_like(prior)
    rhs=np.where(target,prior,np.nan_to_num(observed))
    both=pair_valid&target[:-1]&target[1:]
    left=pair_valid&target[:-1]
    right=pair_valid&target[1:]
    diagonal[:-1]+=strength*left
    diagonal[1:]+=strength*right
    delta=np.where(pair_valid,increment,0.)
    rhs[:-1]+=strength*np.where(left,-delta+np.where(target[1:],0,np.nan_to_num(observed[1:])),0.)
    rhs[1:]+=strength*np.where(right,delta+np.where(target[:-1],0,np.nan_to_num(observed[:-1])),0.)
    off=-strength*both.astype(float)
    # Independent tridiagonal systems for each link, vectorized across links.
    for t in range(1,len(prior)):
        ratio=off[t-1]/diagonal[t-1]
        diagonal[t]-=ratio*off[t-1]
        rhs[t]-=ratio*rhs[t-1]
    solution=np.empty_like(prior)
    solution[-1]=rhs[-1]/diagonal[-1]
    for t in range(len(prior)-2,-1,-1):
        solution[t]=(rhs[t]-off[t]*solution[t+1])/diagonal[t]
    return np.where(target,solution,observed)
