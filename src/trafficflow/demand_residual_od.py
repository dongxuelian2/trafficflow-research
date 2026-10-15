"""Shared log-demand center followed by flexible entropy path correction.

Counts inform both stages. This is a two-stage regularized point estimator,
not a joint posterior or independently observed prior.
"""
import numpy as np
from .entropy_od import entropy_inverse
from .log_demand_od import log_demand_inverse


GROUP_SETTINGS={'group_sd':.2,'noise_cv':.02,'prior_floor':2.,'huber_delta':None}


def residual_from_demand(A,counts,prior,demand,center_blend=.5,regularization=.05,residual_floor=2.):
    b,g=np.asarray(prior,float),np.asarray(demand,float)
    if not np.isfinite(center_blend) or not 0<=center_blend<=1:
        raise ValueError('Center blend must be finite in [0,1]')
    if not np.isfinite(residual_floor) or residual_floor<=0:raise ValueError('Positive finite residual center floor required')
    if b.ndim!=1 or g.shape!=b.shape or not np.isfinite(g).all() or (g<=0).any():
        raise ValueError('Positive finite matching demand center required')
    if not np.isfinite(b).all() or (b<0).any():raise ValueError('Finite nonnegative weak prior required')
    # alpha=0 exactly recovers the frozen entropy reference for every input.
    center=np.exp((1-center_blend)*np.log(np.maximum(b,20.))+center_blend*np.log(g)) if center_blend else np.maximum(b,20.)
    f,info=entropy_inverse(A,counts,center,regularization=regularization,prior_floor=residual_floor)
    return f,{'success':True,'solver':'demand_center_then_entropy_path_residual',
        'center_blend':center_blend,'regularization':regularization,'entropy_info':info,
        'center_floor':residual_floor,'weak_prior_floor_before_blend':20.,
        'center_relative_L1_from_weak_prior':float(np.abs(center-b).sum()/max(b.sum(),1e-9)),
        'scope':'Counts reused in two stages;regularized point estimate,not joint Bayesian posterior'}


def demand_residual_inverse(A,counts,prior,paths,center_blend=.5,regularization=.05,residual_floor=2.):
    if center_blend==0:
        f,info=residual_from_demand(A,counts,prior,np.maximum(prior,2.),center_blend,regularization,residual_floor)
        info['group_stage_skipped']=True
        return f,info
    g,group_info=log_demand_inverse(A,counts,prior,paths,**GROUP_SETTINGS)
    f,info=residual_from_demand(A,counts,prior,g,center_blend,regularization,residual_floor)
    info['group_stage']=group_info
    return f,info
