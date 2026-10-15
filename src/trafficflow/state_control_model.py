"""State-only objective and portable masked-history predictor."""
import pickle
from pathlib import Path
import numpy as np
from .state_control_data import causal_state_input
from .joint_queue_data import reconstruction_features


def state_only_loss(prediction,truth,domain):
    count=domain.sum(dim=2).clamp_min(1);loss=((prediction-truth).abs()*domain).sum(dim=2)/count
    return .1*loss[:,0].mean()+.1*loss[:,1].mean()


class StateControlModel:
    def __init__(self,links,geometry,states=None,trees=None):
        if bool(states)==bool(trees):raise ValueError('Exactly one State estimator family required')
        self.links=list(links);self.geometry=geometry;self.states=states;self.trees=trees;self._networks=None;self._boosters=None

    def predict_tensor(self,x):
        x=np.asarray(x,np.float32)
        if x.ndim!=4 or x.shape[1:]!=(13,len(self.links),9):raise ValueError('State batch shape differs')
        if self.states is not None:
            import os
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            import torch
            from .joint_graph_queue import JointGraphQueue
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            if self._networks is None:
                self._networks=[]
                for state in self.states:
                    model=JointGraphQueue().to(device);model.load_state_dict(state);model.eval();self._networks.append(model)
            graph=[torch.tensor(self.geometry[k],device=device) for k in ['incoming','outgoing','geometry']]
            with torch.no_grad(),torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,allow_tf32=False):
                tensor=torch.tensor(x.transpose(0,3,1,2).copy(),device=device);values=[model.analyze(tensor,*graph).cpu().numpy() for model in self._networks]
            return np.mean(values,axis=0)
        from lightgbm import Booster
        if self._boosters is None:self._boosters=[[Booster(model_str=text) for text in pair] for pair in self.trees]
        features=reconstruction_features(x,self.geometry);members=[]
        for pair in self._boosters:
            p=np.stack([model.predict(features,num_threads=4) for model in pair],axis=-1).reshape(len(x),len(self.links),2).transpose(0,2,1).astype(np.float32).clip(min=0);visible=x[:,-1,:,2:4].transpose(0,2,1)>.5;members.append(np.where(visible,x[:,-1,:,:2].transpose(0,2,1),p))
        return np.mean(members,axis=0)

    def predict_current(self,history,net,origin):
        x=causal_state_input(history,self.links,net,origin);return self.predict_tensor(x[None])[0]

    def save(self,path):
        cache=(self._networks,self._boosters);self._networks=self._boosters=None
        try:
            with Path(path).open('xb') as f:pickle.dump(self,f,protocol=5)
        finally:self._networks,self._boosters=cache

    @staticmethod
    def load(path):
        with Path(path).open('rb') as f:return pickle.load(f)
