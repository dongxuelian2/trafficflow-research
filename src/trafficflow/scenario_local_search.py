"""Bounded non-prefix search with separate conditional-Monte-Carlo validation."""
import numpy as np
from .joint_queue_scenarios import scenario_prefix_set,ScenarioOnsetQueueModel
from .expected_iou_queue import apply_platt


def scenario_set_scores(prediction,labels,eligible,evidence):
    pred=np.asarray(prediction,bool);y,m,e=np.asarray(labels,bool),np.asarray(eligible,bool),np.asarray(evidence,bool)
    intersection=(y&m&pred).sum(axis=1);union=((y|pred)&m).sum(axis=1)
    return np.divide(intersection,union,out=e.astype(float).copy(),where=union>0)


def bounded_scenario_search(probability,links,labels,eligible,evidence,weights=None,max_moves=8,min_step=1e-6):
    prefix,initial=scenario_prefix_set(probability,links,labels,eligible,evidence,weights)
    if not isinstance(max_moves,int) or max_moves<0 or not np.isfinite(min_step) or min_step<0:raise ValueError('Finite bounded search budget required')
    y,m,e=np.asarray(labels,bool),np.asarray(eligible,bool),np.asarray(evidence,bool);link=np.asarray(links).astype(str);weight=np.ones(len(y)) if weights is None else np.asarray(weights,float);weight=weight/weight.sum();a=(y&m).astype(int);b=m.astype(int)-a;truth=a.sum(axis=1)
    pred=prefix.copy();current=float(np.sum(weight*scenario_set_scores(pred,y,m,e)));trace=[];evaluations=0;stopped='move_budget'
    for _ in range(max_moves):
        intersection=(a*pred).sum(axis=1);union=truth+(b*pred).sum(axis=1);sign=1-2*pred
        candidate_i=intersection[:,None]+a*sign;candidate_u=union[:,None]+b*sign
        score=np.divide(candidate_i,candidate_u,out=np.broadcast_to(e[:,None],candidate_u.shape).astype(float).copy(),where=candidate_u>0);values=np.sum(weight[:,None]*score,axis=0);evaluations+=len(pred);best=float(values.max())
        if best<=current+min_step:stopped='single_flip_stationary';break
        tied=np.flatnonzero(values>=best-1e-12);j=min(tied,key=lambda j:(not bool(pred[j]),str(link[j])))
        old=current;operation='remove' if pred[j] else 'add';pred[j]=1-pred[j];current=float(values[j]);trace.append({'link_id':str(link[j]),'operation':operation,'before':old,'after':current,'gain':current-old})
    return pred,{'prefix_prediction':prefix,'initial_expected_proxy_IoU':initial['expected_proxy_IoU'],'expected_proxy_IoU':current,'moves':trace,'n_neighbor_evaluations':evaluations,'max_moves':max_moves,'min_step':min_step,'stop_reason':stopped,'global_optimum_claimed':False}


def accept_on_validation(prefix,proposal,labels,eligible,evidence,min_gain=.001):
    """Approximate paired MC interval, conditional on a hypothetical fit law.

    This is a numerical screen, not training uncertainty, a multiple-testing
    guarantee, or confidence in real traffic/official performance.
    """
    left=scenario_set_scores(prefix,labels,eligible,evidence);right=scenario_set_scores(proposal,labels,eligible,evidence)
    if len(left)<2:raise ValueError('At least two validation scenarios required')
    delta=right-left;gain=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(delta)));interval=[gain-1.96*se,gain+1.96*se];accepted=bool(gain>=min_gain and interval[0]>0)
    return (np.asarray(proposal).copy() if accepted else np.asarray(prefix).copy()),{'accepted':accepted,'validation_expected_gain':gain,'validation_paired_MC_ci95':interval,'minimum_gain':min_gain,'n_validation_scenarios':len(delta),'scope':'Conditional simulation check,not a true traffic/model/Kaggle interval.'}


def bank_search(probability,links,bank,validation_bank,max_moves=8):
    p=np.asarray(probability);y=bank['copula_uniform']<p[None,:];m=bank['last_eligible'];e=bank['evidence'];proposal,detail=bounded_scenario_search(p,links,y,m,e,max_moves=max_moves)
    vy=validation_bank['copula_uniform']<p[None,:];vm=validation_bank['last_eligible'];ve=validation_bank['evidence'];pred,acceptance=accept_on_validation(detail['prefix_prediction'],proposal,vy,vm,ve)
    return pred,proposal,{**detail,**acceptance}


class ValidatedLocalScenarioQueueModel(ScenarioOnsetQueueModel):
    def __init__(self,prefix_model,validation_bank,max_moves=8):
        super().__init__(prefix_model.base,prefix_model.calibration,prefix_model.bank,'copula','empirical');self.validation_bank=validation_bank;self.max_moves=max_moves

    def predict_window(self,history,target,net,topology,origin,condition):
        if condition=='queue_ongoing':return self.base.incumbent.predict_window(history,target,net,topology,origin,condition)
        if condition!='queue_onset':raise ValueError('Published Queue condition required')
        score,lead,lookup,links=self.base._onset(history,target,net,topology,origin);p=apply_platt(self.calibration,score);last=np.flatnonzero(lead==30)
        if not np.array_equal(links[last].astype(str),np.asarray(self.links).astype(str)):raise ValueError('Scenario node order differs')
        pred=np.zeros(len(score),int);pred[last]=bank_search(p[last],links[last],self.bank,self.validation_bank,self.max_moves)[0];out=target.copy();out['queue_pred']=pred[lookup];return out
