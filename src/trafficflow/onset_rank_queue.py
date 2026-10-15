"""Query-relative onset location scores; ongoing stays with a frozen model."""
import numpy as np
import pandas as pd
from .queue import QueueModel,queue_features


def onset_topk_decisions(score,window,lead,links,k):
    """Rank only T+30 on each complete six-step public forecast domain.

    Scores need not be probabilities. Missing future scoring cells are not an
    input. Link-id tie breaking is invariant to target row permutations.
    """
    s=np.asarray(score,float);w,t,l=map(np.asarray,[window,lead,links]);l=l.astype(str)
    if s.ndim!=1 or any(a.shape!=s.shape for a in [w,t,l]) or not np.isfinite(s).all() or not isinstance(k,(int,np.integer)) or k<1:raise ValueError('Finite scores,matching metadata and positive integer k required')
    if not np.isin(t,[5,10,15,20,25,30]).all():raise ValueError('Unexpected public lead')
    out=np.zeros(len(s),int)
    for ids in pd.DataFrame({'window':w}).groupby('window',sort=False).indices.values():
        final=ids[t[ids]==30];keys=l[final]
        if not len(keys) or len(set(keys))!=len(keys) or len(ids)!=6*len(keys):raise ValueError('Complete public six-step domain required')
        for tau in [5,10,15,20,25,30]:
            values=l[ids[t[ids]==tau]]
            if len(values)!=len(keys) or set(values)!=set(keys):raise ValueError('Future eligibility must not remove forecast candidates')
        order=np.lexsort((keys,-s[final]));out[final[order[:min(k,len(final))]]]=1
    return out


class OnsetTopKQueueModel(QueueModel):
    def __init__(self,incumbent,model_strings,feature_names,k,kind):
        super().__init__('onset_topk',incumbent.seed,600,4)
        if getattr(incumbent,'alpha',None)!=0 or kind not in ['classifier','ranker'] or not model_strings:raise ValueError('Frozen full-domain tree reference and new score models required')
        if not isinstance(k,int) or k<1:raise ValueError('Positive onset set size required')
        self.incumbent=incumbent;self.links=list(incumbent.links);self.model_strings=list(model_strings);self.feature_names=list(feature_names);self.k=k;self.kind=kind;self._boosters=None

    def _onset(self,history,target,net,topology,origin):
        origin=pd.Timestamp(origin)
        if origin.tzinfo is None:raise ValueError('Timezone-aware forecast origin required')
        origin=origin.tz_convert('UTC');times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
        canonical=pd.MultiIndex.from_product([times,self.links],names=['timestamp','link_id']);index=pd.MultiIndex.from_frame(target[['timestamp','link_id']]);lookup=canonical.get_indexer(index)
        if len(index)!=len(canonical) or (lookup<0).any() or index.has_duplicates:raise ValueError('Complete public Queue target required')
        full=canonical.to_frame(index=False);base=self.incumbent.baseline
        x=queue_features(history,full,net,topology,origin,allow_origin=True,include_occupancy=getattr(base,'uses_occupancy_features',False))
        if getattr(base,'uses_published_condition',False):x['condition_ongoing']=0.
        if list(x.columns)!=self.feature_names:raise ValueError('Fitted onset feature schema differs')
        if self._boosters is None:
            from lightgbm import Booster
            self._boosters=[Booster(model_str=s) for s in self.model_strings]
        score=np.mean([model.predict(x,num_threads=4) for model in self._boosters],axis=0)
        if not np.isfinite(score).all():raise ValueError('Nonfinite onset ranking score')
        return score,x.lead_minutes.to_numpy(),lookup,full.link_id.to_numpy()

    def predict_window(self,history,target,net,topology,origin,condition):
        if condition=='queue_ongoing':return self.incumbent.predict_window(history,target,net,topology,origin,condition)
        if condition!='queue_onset':raise ValueError('Published Queue condition required')
        score,lead,lookup,links=self._onset(history,target,net,topology,origin)
        pred=onset_topk_decisions(score,np.full(len(score),'public_window'),lead,links,self.k)
        out=target.copy();out['queue_pred']=pred[lookup];return out

    def save(self,path):
        boosters=self._boosters;self._boosters=None
        try:super().save(path)
        finally:self._boosters=boosters
