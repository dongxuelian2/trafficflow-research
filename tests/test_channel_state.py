import numpy as np
import pandas as pd
import pytest
from trafficflow.data import STATE_KEY,VALUES
from trafficflow.state import StateModel,Profile
from trafficflow.channel_state import ChannelStateEnsemble


class FixedState(StateModel):
    def __init__(self,values):
        self.values=np.asarray(values,float);self.calls=0;self.profile=Profile(pd.DataFrame(),pd.DataFrame());self.feature_version=2
        self._network=None
    def predict(self,features,template):
        self.calls+=1;out=template[STATE_KEY].copy();out[VALUES]=np.broadcast_to(self.values,(len(template),2));return out


def test_channel_mixture_preserves_units_identity_keys_and_cached_save(tmp_path):
    target=pd.DataFrame({'panel':['p']*2,'timestamp':['t']*2,'station_id':['1','2'],'link_id':['x']*2,'mask_regime':['R3']*2})
    a,b=FixedState([90.,3000.]),FixedState([40.,2000.]);model=ChannelStateEnsemble([a,b],[[1.,.25],[0.,.75]])
    p=model.predict(None,target);np.testing.assert_array_equal(p[VALUES],[[90.,2250.],[90.,2250.]])
    pd.testing.assert_frame_equal(p[STATE_KEY],target[STATE_KEY])
    identity=ChannelStateEnsemble([a,b],[[1,1],[0,0]]);before=b.calls
    np.testing.assert_array_equal(identity.predict(None,target)[VALUES],np.broadcast_to(a.values,(2,2)));assert b.calls==before
    sentinel=object();a._network=sentinel;path=tmp_path/'channels.pkl';model.save(path)
    assert a._network is sentinel
    loaded=StateModel.load(path);assert loaded.models[0]._network is None
    pd.testing.assert_frame_equal(loaded.predict(None,target),p)
    with pytest.raises(ValueError,match='weights'):ChannelStateEnsemble([a,b],[[1,1],[-1,0]])
    with pytest.raises(ValueError,match='weights'):ChannelStateEnsemble([a,b],[[1,0],[0,0]])
