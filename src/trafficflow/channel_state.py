"""Fixed per-channel State ensemble;all choices must be sealed externally."""
import numpy as np
from .state import StateModel
from .data import STATE_KEY,VALUES


class ChannelStateEnsemble(StateModel):
    def __init__(self,models,weights):
        self.models=list(models);self.weights=np.asarray(weights,float)
        if not self.models or self.weights.shape!=(len(self.models),2) or not np.isfinite(self.weights).all() or (self.weights<0).any() or (self.weights.sum(axis=0)<=0).any():
            raise ValueError('Nonnegative model-by-two-channel weights required')
        if not all(isinstance(m,StateModel) for m in self.models):raise ValueError('Trusted State models required')
        self.weights=self.weights/self.weights.sum(axis=0)[None,:]
        self.profile=self.models[0].profile;self.feature_version=self.models[0].feature_version
        for m in self.models[1:]:
            if m.feature_version!=self.feature_version or not m.profile.slots.equals(self.profile.slots) or not m.profile.fallback.equals(self.profile.fallback):
                raise ValueError('State ensemble profiles/features differ')

    def predict(self,features,template):
        values=np.zeros((len(template),2),float);keys=template[STATE_KEY].reset_index(drop=True)
        for model,weight in zip(self.models,self.weights):
            if not weight.any():continue
            p=model.predict(features,template)
            if not p[STATE_KEY].reset_index(drop=True).equals(keys):raise ValueError('Component changed prediction keys/order')
            v=p[VALUES].to_numpy(float)
            if not np.isfinite(v).all() or (v<0).any():raise ValueError('Invalid component State values')
            values+=v*weight[None,:]
        result=template[STATE_KEY].copy();result[VALUES]=values;return result

    def save(self,path):
        caches=[];seen=set()
        def visit(model):
            if id(model) in seen:return
            seen.add(id(model))
            if hasattr(model,'_network'):caches.append((model,model._network));model._network=None
            for name in ['baseline','model','control','expert']:
                child=getattr(model,name,None)
                if isinstance(child,StateModel):visit(child)
            for child in getattr(model,'models',[]):
                if isinstance(child,StateModel):visit(child)
        visit(self)
        try:super().save(path)
        finally:
            for model,network in caches:model._network=network
