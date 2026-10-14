"""Observation-density expectation with a weighted speed/flow projection."""
import numpy as np
from .data import VALUES,STATE_KEY
from .state import StateModel


class DensityStateModel(StateModel):
    def __init__(self,baseline,density_model,blend=0.):
        self.baseline=baseline
        self.density_model=density_model
        self.profile=baseline.profile
        self.feature_version=baseline.feature_version
        self.blend=blend

    def predict(self,features,template):
        return self.project(features,template,self.baseline.predict(features,template))

    def project(self,features,template,baseline_prediction):
        # Reuse a frozen baseline prediction when evaluating projection grids.
        # Exact key alignment prevents cached predictions being misassigned.
        if not baseline_prediction[STATE_KEY].reset_index(drop=True).equals(template[STATE_KEY].reset_index(drop=True)):
            raise ValueError('Density projection baseline keys are not aligned')
        result=baseline_prediction.copy()
        net=features.net
        lanes=template.link_id.map(net.lanes).to_numpy(float)
        vf=template.link_id.map(net.free_speed_kmh).to_numpy(float)
        capacity=template.link_id.map(net.capacity_vph).to_numpy(float)
        y=result[VALUES].to_numpy(float,copy=True)
        y[:,0]=np.minimum(y[:,0],vf*1.05)
        y[:,1]=np.minimum(y[:,1],capacity*1.05)/lanes
        if self.blend:
            x=features.features.iloc[features.lookup(template)]
            density=x.density_base.to_numpy(float)*25+self.density_model.predict(x)*25
            jam=template.link_id.map(net.k_jam).to_numpy(float)/lanes
            density=np.clip(density,1e-3,jam*1.05)
            wv,wq=.54/25**2,.46/600**2
            speed=(wv*y[:,0]+wq*density*y[:,1])/(wv+wq*density**2)
            projected=np.column_stack([speed,density*speed])
            y=(1-self.blend)*y+self.blend*projected
            y[:,0]=np.minimum(y[:,0],vf*1.05)
            y[:,1]=np.minimum(y[:,1],capacity/lanes*1.05)
        y=np.maximum(y,0)
        y[:,1]*=lanes
        if not np.isfinite(y).all():raise ValueError('Nonfinite density-consistent prediction')
        result[VALUES]=y
        return result


class InterpolatedDensityResidual:
    """Use masked temporal density or the existing temporal/spatial mixture."""
    def __init__(self,mode='temporal'):
        if mode not in ('temporal','base'):raise ValueError('Unknown density interpolation')
        self.mode=mode

    def predict(self,x):
        base=x.density_base.to_numpy(float)
        density=x.density_temporal.to_numpy(float) if self.mode=='temporal' else base
        return np.where(np.isfinite(density),density,base)-base
