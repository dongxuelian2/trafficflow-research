"""Portable joint model with history-only forecast and reconstruction APIs."""
import numpy as np
from .ensemble_queue import EnsembleFullDomainQueueModel


class JointDynamicsQueueModel(EnsembleFullDomainQueueModel):
    def __init__(self,baseline,links,setting,alpha,states,geometry,rollout):
        super().__init__(baseline,links,setting,alpha,states,geometry);self.rollout=bool(rollout)

    def predict_dynamics(self,history,net,origin,condition):
        import torch
        from .graph_queue import queue_history_input
        from .joint_graph_queue import JointGraphQueue
        from .probability_calibration import weighted_bce_probability
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self._networks is None:
            torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;self._networks=[]
            for state in self.states:
                model=JointGraphQueue(hidden=32,rollout=self.rollout).to(device);model.load_state_dict(state);model.eval();self._networks.append(model)
        x=queue_history_input(history,self.links,net,origin,condition);graph=[torch.tensor(self.geometry[k],device=device) for k in ['incoming','outgoing','geometry']];rows=[]
        with torch.no_grad():
            tensor=torch.tensor(x.transpose(2,0,1)[None].copy(),device=device)
            for model in self._networks:
                outputs=[v.cpu().numpy()[0] for v in model(tensor,*graph)];outputs[0]=weighted_bce_probability(outputs[0],4.);rows.append(outputs)
        return {k:np.mean([r[i] for r in rows],axis=0) for i,k in enumerate(['probability','speed','flow','analysis'])}

    def _graph_probability(self,history,net,origin,condition):
        return self.predict_dynamics(history,net,origin,condition)['probability']

    def reconstruct_current(self,history,net,origin):
        from .graph_queue import queue_history_input
        x=queue_history_input(history,self.links,net,origin,'queue_onset');cut=net.v_cut.reindex(self.links).to_numpy(np.float32)/25.;queue=(x[:12,:,0]<=cut)&(x[:12,:,2]>.5);condition='queue_ongoing' if (queue.sum(axis=0)>=2).any() else 'queue_onset'
        return self.predict_dynamics(history,net,origin,condition)['analysis']
