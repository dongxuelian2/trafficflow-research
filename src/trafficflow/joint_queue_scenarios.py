"""Fit-only empirical binary dependence and hypothetical eligibility scenarios.

Own randomized binary-rank construction, inspired by ensemble copula coupling
(Schefzik et al.,2013,arXiv:1302.7149). This is an empirical proxy law, not a
claim of calibrated latent traffic truth or organizer future eligibility.
"""
import numpy as np
from .queue import QueueModel
from .expected_iou_queue import apply_platt


def scenario_prefix_set(probability,links,scenario_labels,scenario_eligible,scenario_evidence,weights=None,allow_empty=True):
    """Optimize scenario-mean masked IoU over probability-ranked prefixes.

    Evidence includes all six steps. Only the final step may contain queues.
    Inputs are generated hypothetical scenarios; never a realized future mask.
    The prefix search is not a global optimum over all 2^N possible sets.
    """
    p=np.asarray(probability,float);link=np.asarray(links).astype(str);y,m,e=map(np.asarray,[scenario_labels,scenario_eligible,scenario_evidence])
    if p.ndim!=1 or not len(p) or link.shape!=p.shape or len(set(link))!=len(link) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Finite full-domain probabilities and unique links required')
    if y.ndim!=2 or y.shape[1]!=len(p) or not len(y) or m.shape!=y.shape or e.shape!=(len(y),) or any(not np.isin(a,[0,1]).all() for a in [y,m,e]):raise ValueError('Aligned binary hypothetical scenarios required')
    y,m,e=y.astype(bool),m.astype(bool),e.astype(bool)
    if (m.any(axis=1)&~e).any():raise ValueError('Last-step eligibility implies six-step evidence')
    weight=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    if weight.shape!=(len(y),) or not np.isfinite(weight).all() or (weight<0).any() or weight.sum()<=0:raise ValueError('Nonnegative scenario weights required')
    weight=weight/weight.sum();order=np.lexsort((link,-p));positive=(y&m)[:,order];coverage=m[:,order]
    intersection=np.cumsum(positive,axis=1);predicted_coverage=np.cumsum(coverage,axis=1);truth_count=positive.sum(axis=1);union=predicted_coverage+truth_count[:,None]-intersection
    scores=np.divide(intersection,union,out=np.broadcast_to(e[:,None],union.shape).astype(float).copy(),where=union>0)
    empty=float(np.sum(weight*np.where(truth_count==0,e,False)));curve=np.r_[empty,np.sum(weight[:,None]*scores,axis=0)];ids=np.arange(0 if allow_empty else 1,len(p)+1);maximum=float(curve[ids].max());k=int(ids[np.flatnonzero(curve[ids]>=maximum-1e-12)[0]])
    out=np.zeros(len(p),int);out[order[:k]]=1
    return out,{'k':k,'expected_proxy_IoU':float(curve[k]),'curve':curve,'scenario_scoreable_positive_probability':float(np.dot(weight,truth_count>0)),'n_scenarios':len(y)}


def build_scenario_bank(labels_last,eligible_six,seeds=(20261013,20261029),draws_per_seed=2048):
    """Bootstrap FIT windows, randomize binary ties, retain joint fit masks.

    Positive observations occupy U in[0,p_fit],negative in[p_fit,1]. Missing
    labels use independent U and never enter fitted rates. Conditional on a
    scoreable node the construction has uniform margins in its population law;
    finite draws,history shift and observation/latent mismatch remain risks.
    """
    y=np.asarray(labels_last);m=np.asarray(eligible_six)
    if y.ndim!=2 or not len(y) or not y.shape[1] or m.shape!=(len(y),6,y.shape[1]) or not np.isin(y,[0,1]).all() or not np.isin(m,[0,1]).all() or not isinstance(draws_per_seed,int) or draws_per_seed<1 or not seeds:raise ValueError('Nonempty fit-only binary windows required')
    y,m=y.astype(bool),m.astype(bool);last=m[:,-1];n=last.sum(axis=0);positive=(y&last).sum(axis=0);rate=np.divide(positive,n,out=np.full(y.shape[1],.5),where=n>0);independent=[];coupled=[];indices=[];members=[];masks=[];evidence=[]
    for member,seed in enumerate(seeds):
        rng=np.random.default_rng(seed);idx=rng.integers(0,len(y),draws_per_seed);u=rng.random((draws_per_seed,y.shape[1]));known=last[idx]
        transformed=np.where(known,np.where(y[idx],u*rate,rate+u*(1-rate)),u)
        independent.append(u);coupled.append(transformed);indices.append(idx);members.append(np.full(draws_per_seed,member));masks.append(last[idx]);evidence.append(m[idx].any(axis=(1,2)))
    return {'independent_uniform':np.concatenate(independent),'copula_uniform':np.concatenate(coupled),'last_eligible':np.concatenate(masks),'evidence':np.concatenate(evidence),
        'template_indices':np.concatenate(indices),'member':np.concatenate(members),'fitted_rates':rate,'fit_positive_counts':positive,'fit_eligible_counts':n,'seeds':np.array(seeds),'draws_per_seed':np.array(draws_per_seed)}


def bank_prefix_set(probability,links,bank,dependence,availability,member=None):
    if dependence not in ['independent','copula'] or availability not in ['full','empirical']:raise ValueError('Known scenario controls required')
    use=np.ones(len(bank['member']),bool) if member is None else bank['member']==member
    if not use.any():raise ValueError('Scenario member missing')
    y=bank[dependence+'_uniform'][use]<np.asarray(probability)[None,:]
    m=bank['last_eligible'][use] if availability=='empirical' else np.ones(y.shape,bool);e=bank['evidence'][use] if availability=='empirical' else np.ones(len(y),bool)
    return scenario_prefix_set(probability,links,y,m,e)


class ScenarioOnsetQueueModel(QueueModel):
    def __init__(self,base,calibration,bank,dependence,availability):
        super().__init__('joint_scenario_onset',base.seed,600,4)
        if base.kind!='classifier' or dependence not in ['independent','copula'] or availability not in ['full','empirical']:raise ValueError('Frozen classifier and scenario law required')
        self.base=base;self.links=list(base.links);self.calibration=calibration;self.bank=bank;self.dependence=dependence;self.availability=availability

    def predict_window(self,history,target,net,topology,origin,condition):
        if condition=='queue_ongoing':return self.base.incumbent.predict_window(history,target,net,topology,origin,condition)
        if condition!='queue_onset':raise ValueError('Published Queue condition required')
        score,lead,lookup,links=self.base._onset(history,target,net,topology,origin);score=apply_platt(self.calibration,score);last=np.flatnonzero(lead==30)
        if not np.array_equal(links[last].astype(str),np.array(self.links).astype(str)):raise ValueError('Scenario bank node order differs')
        pred=np.zeros(len(score),int);pred[last]=bank_prefix_set(score[last],links[last],self.bank,self.dependence,self.availability)[0];out=target.copy();out['queue_pred']=pred[lookup];return out

    def save(self,path):
        cache=self.base._boosters;self.base._boosters=None
        try:super().save(path)
        finally:self.base._boosters=cache
