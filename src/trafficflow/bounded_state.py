"""Apply the same published speed/flow FD caps to a trusted State model."""
import numpy as np
from .state import StateModel
from .data import VALUES


class BoundedState(StateModel):
    def __init__(self,model):
        if not isinstance(model,StateModel):raise ValueError('Trusted State model required')
        self.model=model;self.profile=model.profile;self.feature_version=model.feature_version

    def predict(self,features,template):
        result=self.model.predict(features,template)
        result['speed_kmh']=np.clip(result.speed_kmh.to_numpy(float),0,template.link_id.map(features.net.free_speed_kmh).to_numpy(float)*1.05)
        result['flow_vph']=np.clip(result.flow_vph.to_numpy(float),0,template.link_id.map(features.net.capacity_vph).to_numpy(float)*1.05)
        if not np.isfinite(result[VALUES].to_numpy(float)).all():raise ValueError('Nonfinite bounded State')
        return result
