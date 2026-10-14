"""Calibrate complete forecast probabilities; eligibility is scorer-only."""
import numpy as np
from .queue_calibration import decisions,public_onset_timing


def full_queue_decisions(probability,window,condition,lead,thresholds,require_nonempty=False,timing='before'):
    p=np.asarray(probability,float)
    window,condition,lead=map(np.asarray,[window,condition,lead])
    if p.ndim!=1 or any(a.shape!=p.shape for a in [window,condition,lead]) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Full Queue probability/metadata arrays disagree')
    if not np.isin(condition,['queue_onset','queue_ongoing']).all() or not np.isin(lead,[5,10,15,20,25,30]).all():
        raise ValueError('Invalid public Queue condition/horizon')
    if timing=='after':
        return public_onset_timing(decisions(p,window,condition,thresholds,require_nonempty),lead,condition)
    if timing!='before':raise ValueError('Unknown onset timing order')
    allowed=(condition!='queue_onset')|(lead==30)
    thresholds_array=np.array([thresholds.get(str(c),thresholds['global']) for c in condition])
    pred=(p>=thresholds_array)&allowed
    if require_nonempty:
        for key in np.unique(window):
            indices=np.flatnonzero(window==key)
            if len(np.unique(condition[indices]))!=1:raise ValueError('Window condition must be constant')
            candidates=indices[allowed[indices]]
            if not len(candidates):raise ValueError('No public allowed candidate in forecast window')
            if not pred[indices].any():pred[candidates[np.argmax(p[candidates])]]=True
    return pred.astype(int)


class FullForecastScorer:
    """Fast calibration score matching score_full_forecast, including empty evidence."""
    def __init__(self,labels,window,condition,score_mask):
        self.y,self.mask=np.asarray(labels,bool),np.asarray(score_mask,bool)
        window,condition=map(np.asarray,[window,condition])
        if self.y.ndim!=1 or any(a.shape!=self.y.shape for a in [window,condition,self.mask]):raise ValueError('Scorer arrays disagree')
        self.windows,self.groups=np.unique(window,return_inverse=True)
        self.conditions=[]
        for i in range(len(self.windows)):
            c=np.unique(condition[self.groups==i])
            if len(c)!=1:raise ValueError('Window condition must be constant')
            self.conditions.append(c[0])
        self.conditions=np.array(self.conditions)
        self.counts=np.bincount(self.groups,weights=self.mask,minlength=len(self.windows))

    def score(self,prediction):
        p=np.asarray(prediction,bool)
        if p.shape!=self.y.shape:raise ValueError('Forecast prediction shape disagrees with scorer')
        inter=np.bincount(self.groups,weights=self.mask&self.y&p,minlength=len(self.windows))
        union=np.bincount(self.groups,weights=self.mask&(self.y|p),minlength=len(self.windows))
        iou=np.divide(inter,union,out=np.ones(len(union)),where=union>0);iou[self.counts==0]=0.
        return float(np.mean([iou[self.conditions==c].mean() for c in np.unique(self.conditions)]))


def tune_full_probabilities(probability,window,condition,lead,labels,score_mask,timing='before'):
    """Only calibration scores use labels/mask; every decision uses full rows."""
    p=np.asarray(probability,float);window,condition,lead=map(np.asarray,[window,condition,lead])
    scorer=FullForecastScorer(labels,window,condition,score_mask)
    grid=np.linspace(.05,.80,31);settings=[]
    for force in [False,True]:
        best_condition={}
        for c in np.unique(condition):
            use=condition==c
            local=FullForecastScorer(np.asarray(labels)[use],window[use],condition[use],np.asarray(score_mask)[use])
            values=[(local.score(full_queue_decisions(p[use],window[use],condition[use],lead[use],{'global':float(t)},force,timing)),float(t)) for t in grid]
            best_condition[str(c)]=max(values,key=lambda r:(r[0],-abs(r[1]-.5)))[1]
        for mode in ['global','condition']:
            candidates=[{'global':float(t)} for t in grid] if mode=='global' else [{'global':.5,**best_condition}]
            for thresholds in candidates:
                prediction=full_queue_decisions(p,window,condition,lead,thresholds,force,timing)
                settings.append({'mode':mode,'thresholds':thresholds,'require_nonempty':force,
                    'timing':timing,'calibration_proxy_IoU':scorer.score(prediction)})
    selected=max(settings,key=lambda r:(r['calibration_proxy_IoU'],r['mode']=='global',not r['require_nonempty']))
    return {'selected':selected,'settings':settings,'scope':'Complete public forecast domain. Future score mask used solely to evaluate calibration predictions.'}
