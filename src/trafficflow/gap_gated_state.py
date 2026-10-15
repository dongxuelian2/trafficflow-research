"""Offline State expert selection from public measurement availability only."""
import numpy as np
import pandas as pd
from .state import StateModel
from .data import STATE_KEY,VALUES


def missing_run_steps(raw,times):
    values=np.asarray(raw)
    if values.ndim!=3 or values.shape[-1]!=2 or len(times)!=len(values) or not len(times):
        raise ValueError('State raw/time grid required')
    dt=pd.to_datetime(times,utc=True)
    if len(times)>1 and not (np.diff(dt.as_unit('s').asi8)==300).all():raise ValueError('Consecutive five-minute grid required')
    missing=~np.isfinite(values).all(axis=-1)
    forward=np.zeros(missing.shape,np.int32);backward=np.zeros_like(forward)
    for t in range(len(values)):forward[t]=missing[t]*(1+(forward[t-1] if t else 0))
    for t in range(len(values)-1,-1,-1):backward[t]=missing[t]*(1+(backward[t+1] if t+1<len(values) else 0))
    return np.where(missing,forward+backward-1,0)


class GapGatedState(StateModel):
    """Use a fixed expert for target cells in an observed >=180min gap.

    Uses both sides of the released offline State day. It is not a causal
    Queue model;no teacher,scoring mask,regime label or outage metadata is read.
    """
    def __init__(self,control,expert,threshold_steps=36):
        if not isinstance(control,StateModel) or not isinstance(expert,StateModel):raise ValueError('Trusted State models required')
        if not isinstance(threshold_steps,int) or threshold_steps<1:raise ValueError('Positive integer gap threshold required')
        if control.feature_version!=expert.feature_version or not control.profile.slots.equals(expert.profile.slots) or not control.profile.fallback.equals(expert.profile.fallback):
            raise ValueError('Experts require identical profiles and feature versions')
        self.control=control;self.expert=expert;self.threshold_steps=threshold_steps
        self.profile=control.profile;self.feature_version=control.feature_version

    def gate(self,features,template):
        runs=missing_run_steps(features.raw,features.times)
        return runs.ravel()[features.lookup(template)]>=self.threshold_steps

    def predict(self,features,template):
        if not len(template):
            result=template[STATE_KEY].copy();result[VALUES]=np.empty((0,2));return result
        gate=self.gate(features,template);keys=template[STATE_KEY].reset_index(drop=True)
        def checked(model,target):
            prediction=model.predict(features,target)
            if not prediction[STATE_KEY].reset_index(drop=True).equals(target[STATE_KEY].reset_index(drop=True)):
                raise ValueError('Expert changed prediction keys/order')
            values=prediction[VALUES].to_numpy(float)
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid expert prediction')
            return prediction
        if gate.all():return checked(self.expert,template)
        result=checked(self.control,template)
        if gate.any():result.loc[gate,VALUES]=checked(self.expert,template.loc[gate])[VALUES].to_numpy(float)
        if not result[STATE_KEY].reset_index(drop=True).equals(keys):raise ValueError('Gate changed prediction keys')
        return result

    def save(self,path):
        caches=[];seen=set()
        def visit(model):
            if id(model) in seen:return
            seen.add(id(model))
            if hasattr(model,'_network'):caches.append((model,model._network));model._network=None
            for key in ['control','expert','baseline']:
                member=getattr(model,key,None)
                if isinstance(member,StateModel):visit(member)
            for member in getattr(model,'models',[]):
                if isinstance(member,StateModel):visit(member)
        visit(self)
        try:super().save(path)
        finally:
            for model,network in caches:model._network=network
