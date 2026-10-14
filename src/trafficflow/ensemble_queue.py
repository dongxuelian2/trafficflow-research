"""Mean corrected scores from frozen causal networks on a complete template."""
import numpy as np
from .full_domain_queue import FullDomainQueueModel


class EnsembleFullDomainQueueModel(FullDomainQueueModel):
    def __init__(self,baseline,links,setting,alpha,states,geometry):
        if not len(states):raise ValueError('At least one frozen network required')
        super().__init__(baseline,links,setting,alpha,states[0],geometry)
        self.states=[{k:v.detach().cpu().clone() for k,v in state.items()} for state in states]
        self._networks=None

    def _graph_probability(self,history,net,origin,condition):
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        from .graph_queue import CausalGraphQueue,queue_history_input
        from .probability_calibration import weighted_bce_probability
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self._networks is None:
            torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            self._networks=[]
            for state in self.states:
                model=CausalGraphQueue(hidden=32,use_graph=True).to(device)
                model.load_state_dict(state);model.eval();self._networks.append(model)
        x=queue_history_input(history,self.links,net,origin,condition)
        graph=[torch.tensor(self.geometry[k],device=device) for k in ['incoming','outgoing','geometry']]
        probabilities=[]
        with torch.no_grad():
            tensor=torch.from_numpy(x.transpose(2,0,1)[None].copy()).to(device)
            for model in self._networks:
                logits,_=model(tensor,*graph)
                probabilities.append(weighted_bce_probability(logits.cpu().numpy()[0],4.))
        return np.mean(probabilities,axis=0)

    def save(self,path):
        cache=self._networks;self._networks=None
        try:super().save(path)
        finally:self._networks=cache
