"""Public-observation density-transition learning and anchored State estimates."""
import numpy as np
import pandas as pd
from .data import VALUES
from .state import StateModel
from .density_assimilation import assimilate_density

PAIR_COLUMNS=['density_base','density_temporal','density_spatial','v_temporal','q_temporal',
    'v_spatial','q_spatial','v_age','q_age','v_prior','q_prior','v_lag1','v_lead1','q_lag1','q_lead1',
    'v_up','v_down','q_up','q_down','q_up_flux','q_down_flux','corridor_flow_scale',
    'corridor_queue_fraction','sin_tod','cos_tod','position','lanes','free_speed_kmh','capacity_vph','length_km']


def pair_features(features,target):
    shape=target.shape
    values=features.features[PAIR_COLUMNS].to_numpy(np.float32).reshape(*shape,-1)
    obs=features.raw[...,1]/np.maximum(features.raw[...,0],1.)
    available=np.isfinite(obs)
    obs=np.where(available,obs,np.nan)
    left,right=values[:-1],values[1:]
    extra=np.stack([obs[:-1]/25,obs[1:]/25,available[:-1],available[1:],target[:-1],target[1:]],axis=-1)
    array=np.concatenate([left,right,right-left,extra],axis=-1).reshape(-1,3*len(PAIR_COLUMNS)+6)
    columns=[f'{side}_{c}' for side in ('left','right','difference') for c in PAIR_COLUMNS]+[
        'observed_density_left','observed_density_right','available_left','available_right','target_left','target_right']
    valid=available|target
    times=pd.to_datetime(features.times,utc=True)
    contiguous=np.asarray((times[1:]-times[:-1]).total_seconds()==300)
    enabled=(valid[:-1]&valid[1:]&(target[:-1]|target[1:]))&contiguous[:,None]
    return pd.DataFrame(array,columns=columns),enabled,obs


class IncrementStateModel(StateModel):
    def __init__(self,baseline,transition_model,strength=0.):
        self.baseline=baseline
        self.transition_model=transition_model
        self.strength=strength
        self.profile=baseline.profile
        self.feature_version=baseline.feature_version

    def predict(self,features,template):
        result=self.baseline.predict(features,template)
        net=features.net
        idx=features.lookup(template)
        lanes=template.link_id.map(net.lanes).to_numpy(float)
        vf=template.link_id.map(net.free_speed_kmh).to_numpy(float)
        capacity=template.link_id.map(net.capacity_vph).to_numpy(float)
        y=result[VALUES].to_numpy(float)
        y[:,0]=np.minimum(y[:,0],vf*1.05)
        y[:,1]=np.minimum(y[:,1],capacity*1.05)/lanes
        if self.strength:
            shape=features.raw.shape[:2]
            count=np.bincount(idx,minlength=np.prod(shape))
            target=(count>0).reshape(shape)
            sums=np.zeros((np.prod(shape),2))
            np.add.at(sums,idx,y)
            means=sums/np.maximum(count[:,None],1)
            prior=(means[:,1]/np.maximum(means[:,0],1.)).reshape(shape)
            x,enabled,obs=pair_features(features,target)
            delta=np.zeros(enabled.shape)
            delta[enabled]=self.transition_model.predict(x.loc[enabled.ravel()])*25
            adjusted=assimilate_density(prior,obs,target,delta,enabled,self.strength).ravel()[idx]
            jam=template.link_id.map(net.k_jam).to_numpy(float)/lanes
            density=np.clip(adjusted,1e-3,jam*1.05)
            wv,wq=.54/25**2,.46/600**2
            speed=(wv*y[:,0]+wq*density*y[:,1])/(wv+wq*density*density)
            y=np.column_stack([speed,density*speed])
            y[:,0]=np.minimum(y[:,0],vf*1.05)
            y[:,1]=np.minimum(y[:,1],capacity/lanes*1.05)
        y=np.maximum(y,0)
        y[:,1]*=lanes
        if not np.isfinite(y).all():raise ValueError('Nonfinite anchored increment prediction')
        result[VALUES]=y
        return result
