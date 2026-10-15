"""Exact independent-Bernoulli IoU plug-in decision, plus separate calibration.

Algorithm reference: Dai & Li, JMLR24(2023), Lemma13 (RankSEG).
Own suffix-convolution implementation; no external implementation copied.
Assumptions are not claims about spatial traffic labels or future eligibility.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from .queue import QueueModel
from .onset_rank_queue import onset_topk_decisions
from .onset_cardinality_queue import onset_threshold_decisions


def fit_monotone_platt(probability,labels,window,regularization=.01,min_windows=3):
    p,y,w=map(np.asarray,[probability,labels,window]);p=p.astype(float);y=y.astype(float)
    if p.ndim!=1 or y.shape!=p.shape or w.shape!=p.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all() or regularization<=0:raise ValueError('Finite calibration-only probabilities/binary labels required')
    keys,codes=np.unique(w.astype(str),return_inverse=True);counts=np.bincount(codes);state={'slope':1.,'intercept':0.,'regularization':float(regularization),'slope_bounds':[.05,5.],'intercept_bounds':[-8.,8.],
        'clip':1e-6,'n_windows':len(keys),'n_rows':len(y),'n_positive':int(y.sum()),'fitted':False,'success':True,'loss_before':None,'loss_after':None,'objective':None}
    if not len(y):state['reason']='no calibration rows';return state
    z=logit(np.clip(p,1e-6,1-1e-6));weight=1/counts[codes];weight=weight/weight.sum()
    def objective(theta):
        linear=theta[0]*z+theta[1];delta=theta-np.array([1.,0.]);loss=np.dot(weight,np.logaddexp(0,linear)-y*linear)+regularization/2*np.dot(delta,delta)
        error=weight*(expit(linear)-y);gradient=np.array([np.dot(error,z),error.sum()])+regularization*delta
        return float(loss),gradient
    state['loss_before']=objective(np.array([1.,0.]))[0]
    if len(keys)<min_windows or len(np.unique(y))<2:state['reason']='insufficient calibration windows or classes';state['loss_after']=state['loss_before'];state['objective']=state['loss_before'];return state
    result=minimize(objective,np.array([1.,0.]),jac=True,method='L-BFGS-B',bounds=[(.05,5.),(-8.,8.)],options={'ftol':1e-15,'gtol':1e-10,'maxiter':1000})
    if not result.success:raise RuntimeError('Monotone calibration failed: '+str(result.message))
    slope,intercept=map(float,result.x);linear=slope*z+intercept
    state.update(slope=slope,intercept=intercept,fitted=True,loss_after=float(np.dot(weight,np.logaddexp(0,linear)-y*linear)),objective=float(result.fun),iterations=int(result.nit),reason=str(result.message))
    return state


def apply_platt(state,probability):
    p=np.asarray(probability,float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Finite probabilities required')
    return expit(state['slope']*logit(np.clip(p,state['clip'],1-state['clip']))+state['intercept'])


def independent_iou_set(probability,links,allow_empty=True):
    """O(N^2) exact expectation over every top-k, empty/empty IoU=1.

    k>0: sum(top-k p) * E[1/(k + outside-positive-count)].
    k=0: P(all labels zero). Future scoring masks are not an argument.
    This is exact for the stated full-domain independent-label model only.
    """
    p=np.asarray(probability,float);links=np.asarray(links).astype(str);n=len(p)
    if p.ndim!=1 or links.shape!=p.shape or not n or len(set(links))!=n or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Unique links and finite full-domain probabilities required')
    order=np.lexsort((links,-p));q=p[order];prefix=np.cumsum(q);distribution=np.array([1.]);expected=np.zeros(n+1)
    for k in range(n,0,-1):
        expected[k]=prefix[k-1]*np.dot(distribution,1/(k+np.arange(len(distribution))))
        updated=np.zeros(len(distribution)+1);updated[:-1]+=distribution*(1-q[k-1]);updated[1:]+=distribution*q[k-1];distribution=updated
    expected[0]=distribution[0];allowed=np.arange(0 if allow_empty else 1,n+1);maximum=float(np.max(expected[allowed]));k=int(allowed[np.flatnonzero(expected[allowed]>=maximum-1e-12)[0]])
    result=np.zeros(n,int);result[order[:k]]=1
    return result,{'k':k,'expected_iou':float(expected[k]),'empty_probability':float(expected[0]),'expected_positive_count':float(prefix[-1]),'expected_by_size':expected}


def onset_expected_iou_decisions(probability,window,lead,links,allow_empty=True):
    p=np.asarray(probability,float);w,t,l=map(np.asarray,[window,lead,links]);onset_topk_decisions(p,w,t,l,1);out=np.zeros(len(p),int)
    for ids in pd.DataFrame({'window':w}).groupby('window',sort=False).indices.values():
        last=ids[t[ids]==30];out[last]=independent_iou_set(p[last],l[last],allow_empty)[0]
    return out


class ProbabilisticOnsetQueueModel(QueueModel):
    def __init__(self,base,calibration=None,allow_empty=True,threshold=None):
        super().__init__('probabilistic_onset',base.seed,600,4)
        if base.kind!='classifier':raise ValueError('Frozen classifier required')
        self.base=base;self.links=list(base.links);self.calibration=calibration;self.allow_empty=bool(allow_empty);self.threshold=threshold

    def predict_window(self,history,target,net,topology,origin,condition):
        if condition=='queue_ongoing':return self.base.incumbent.predict_window(history,target,net,topology,origin,condition)
        if condition!='queue_onset':raise ValueError('Published Queue condition required')
        score,lead,lookup,links=self.base._onset(history,target,net,topology,origin)
        if self.calibration is not None:score=apply_platt(self.calibration,score)
        window=np.full(len(score),'public_window')
        if self.threshold is None:pred=onset_expected_iou_decisions(score,window,lead,links,self.allow_empty)
        else:pred=onset_threshold_decisions(score,window,lead,links,self.threshold['threshold'],self.threshold['force'])
        out=target.copy();out['queue_pred']=pred[lookup];return out

    def save(self,path):
        cache=self.base._boosters;self.base._boosters=None
        try:super().save(path)
        finally:self.base._boosters=cache
