"""Deployable full-template causal tree/graph Queue probability mixture."""
import numpy as np
import pandas as pd
from .queue import QueueModel,queue_features
from .full_queue_calibration import full_queue_decisions


class FullDomainQueueModel(QueueModel):
    def __init__(self,baseline,links,setting,alpha=0.,state=None,geometry=None):
        super().__init__('full_domain',baseline.seed,baseline.iterations,baseline.threads)
        if not 0<=alpha<=1:raise ValueError('Invalid causal Queue mixture')
        if alpha and (state is None or geometry is None):raise ValueError('Graph Queue needs weights and public geometry')
        self.baseline=baseline;self.links=list(links);self.setting=setting;self.alpha=float(alpha)
        self.state={k:v.detach().cpu().clone() for k,v in state.items()} if state is not None else None
        self.geometry={k:np.asarray(v).copy() for k,v in geometry.items()} if geometry is not None else None
        self._network=None

    def _full_probability(self,history,target,net,topology,origin,condition):
        if condition not in ['queue_onset','queue_ongoing']:raise ValueError('Public Queue condition required')
        origin=pd.Timestamp(origin)
        if origin.tzinfo is None:raise ValueError('Queue origin needs timezone')
        origin=origin.tz_convert('UTC')
        if pd.to_datetime(history.timestamp,utc=True).max()>origin:raise ValueError('Future Queue measurement input rejected')
        times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
        canonical=pd.MultiIndex.from_product([times,self.links],names=['timestamp','link_id'])
        index=pd.MultiIndex.from_frame(target[['timestamp','link_id']])
        lookup=canonical.get_indexer(index)
        if len(target)!=len(canonical) or (lookup<0).any() or index.has_duplicates:
            raise ValueError('Queue model requires the complete public panel forecast template')
        full=canonical.to_frame(index=False)
        features=queue_features(history,full,net,topology,origin,allow_origin=True,
            include_occupancy=getattr(self.baseline,'uses_occupancy_features',False))
        if getattr(self.baseline,'uses_published_condition',False):
            features['condition_ongoing']=float(condition=='queue_ongoing')
        probability=self.baseline.probability(features)
        if self.alpha:
            neural=self._graph_probability(history,net,origin,condition).ravel()
            probability=(1-self.alpha)*probability+self.alpha*neural
        return probability,features.lead_minutes.to_numpy(),lookup

    def _graph_probability(self,history,net,origin,condition):
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        from .graph_queue import CausalGraphQueue,queue_history_input
        from .probability_calibration import weighted_bce_probability
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self._network is None:
            torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            self._network=CausalGraphQueue(hidden=32,use_graph=True).to(device)
            self._network.load_state_dict(self.state);self._network.eval()
        x=queue_history_input(history,self.links,net,origin,condition)
        graph=[torch.tensor(self.geometry[k],device=device) for k in ['incoming','outgoing','geometry']]
        with torch.no_grad():
            logits,_=self._network(torch.from_numpy(x.transpose(2,0,1)[None].copy()).to(device),*graph)
            return weighted_bce_probability(logits.cpu().numpy()[0],4.)

    def forecast_probability(self,history,target,net,topology,origin,condition):
        probability,_,lookup=self._full_probability(history,target,net,topology,origin,condition)
        result=target.copy();result['queue_score']=probability[lookup]
        return result

    def predict_window(self,history,target,net,topology,origin,condition):
        probability,lead,lookup=self._full_probability(history,target,net,topology,origin,condition)
        pred=full_queue_decisions(probability,np.full(len(probability),'public_window'),np.full(len(probability),condition),
            lead,self.setting['thresholds'],self.setting['require_nonempty'],self.setting['timing'])
        result=target.copy();result['queue_pred']=pred[lookup]
        return result

    def save(self,path):
        cache=self._network;self._network=None
        try:super().save(path)
        finally:self._network=cache
