"""Conditional probability decisions using public window sampling constraints."""
import numpy as np
import pandas as pd
from .queue import QueueModel
from .metrics import grouped_iou

def public_onset_timing(prediction,lead_minutes,condition):
    """Host announcement 742350: first five published onset steps are empty."""
    pred=np.asarray(prediction).copy()
    lead=np.asarray(lead_minutes)
    condition=np.asarray(condition)
    if pred.ndim!=1 or pred.shape!=lead.shape or pred.shape!=condition.shape:
        raise ValueError('Onset timing shape mismatch')
    if not np.isin(lead,[5,10,15,20,25,30]).all():raise ValueError('Unexpected Queue lead')
    pred[(condition=='queue_onset')&(lead<30)]=0
    return pred

def decisions(probability,window,condition,thresholds,require_nonempty=False):
    p=np.asarray(probability,float);window=np.asarray(window);condition=np.asarray(condition)
    threshold=np.array([thresholds.get(str(c),thresholds['global']) for c in condition])
    pred=p>=threshold
    if require_nonempty:
        # This is an optional latent-nonempty modeling assumption, not a safe
        # guarantee after score-time eligibility removes cells. The caller must
        # supply the full PUBLIC forecast template: future-eligible subsets
        # change this set decision and leak evaluator information in validation.
        order=pd.DataFrame({'w':window,'p':p,'pred':pred}).groupby('w',sort=False)
        for indices in order.indices.values():
            if not pred[indices].any():pred[indices[np.argmax(p[indices])]]=True
    return pred.astype(int)

class RefinedQueueModel(QueueModel):
    def __init__(self,iterations=600,num_leaves=63,min_child_samples=60,seed=2026):
        super().__init__('lightgbm',seed,iterations,4)
        self.num_leaves=num_leaves;self.min_child_samples=min_child_samples
        self.thresholds={'global':.5};self.require_nonempty=False

    def fit(self,data):
        from lightgbm import LGBMClassifier
        self.model=LGBMClassifier(n_estimators=self.iterations,learning_rate=.035,num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,reg_lambda=8.,random_state=self.seed,n_jobs=self.threads,
            verbosity=-1,deterministic=True,force_col_wise=True)
        self.model.fit(data.x,data.y);return self

    def classify(self,x,window,condition):
        pred=decisions(self.probability(x),window,condition,self.thresholds,self.require_nonempty)
        if getattr(self,'uses_public_onset_timing',False):
            pred=public_onset_timing(pred,x.lead_minutes,condition)
        return pred

    def tune(self,data):
        p=self.probability(data.x);grid=np.linspace(.05,.80,31);settings=[]
        for force in [False,True]:
            best_by_condition={}
            for cond in np.unique(data.condition):
                sel=data.condition==cond;rows=[]
                for threshold in grid:
                    pred=decisions(p[sel],data.window[sel],data.condition[sel],{'global':float(threshold)},force)
                    if getattr(self,'uses_public_onset_timing',False):pred=public_onset_timing(pred,data.x.loc[sel].lead_minutes,data.condition[sel])
                    rows.append((grouped_iou(data.y[sel],pred,data.window[sel],data.condition[sel])['proxy_IoU'],float(threshold)))
                best_by_condition[str(cond)]=max(rows,key=lambda pair:(pair[0],-abs(pair[1]-.5)))[1]
            for mode in ['global','condition']:
                choices=[{'global':float(t)} for t in grid] if mode=='global' else [{'global':.5,**best_by_condition}]
                for thresholds in choices:
                    pred=decisions(p,data.window,data.condition,thresholds,force)
                    if getattr(self,'uses_public_onset_timing',False):pred=public_onset_timing(pred,data.x.lead_minutes,data.condition)
                    metric=grouped_iou(data.y,pred,data.window,data.condition)['proxy_IoU']
                    settings.append({'mode':mode,'require_nonempty':force,'thresholds':thresholds,'calibration_proxy_IoU':metric})
        best=max(settings,key=lambda s:(s['calibration_proxy_IoU'],s['mode']=='global',not s['require_nonempty']))
        self.thresholds=best['thresholds'];self.threshold=self.thresholds['global'];self.require_nonempty=best['require_nonempty']
        return {'selected':best,'settings':settings,'proxy_Brier':float(np.mean((p-data.y)**2)),
                'formal_S_queue':None}
