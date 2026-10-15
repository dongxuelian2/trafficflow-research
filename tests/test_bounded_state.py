from types import SimpleNamespace
import numpy as np
import pandas as pd
from trafficflow.state import StateModel,Profile
from trafficflow.bounded_state import BoundedState
from trafficflow.data import STATE_KEY,VALUES


class ExtremeState(StateModel):
    def __init__(self):
        self.profile=Profile(pd.DataFrame(),pd.DataFrame());self.feature_version=2
    def predict(self,features,template):
        out=template[STATE_KEY].copy();out[VALUES]=np.array([[-1.,-4.],[300.,12000.],[80.,2500.]])
        return out


def test_published_fd_caps_preserve_total_flow_units_and_keys(tmp_path):
    template=pd.DataFrame({'panel':['p']*3,'timestamp':['t']*3,'station_id':['a','b','c'],
        'link_id':['x','y','z'],'mask_regime':['R1']*3})
    features=SimpleNamespace(net=pd.DataFrame({'free_speed_kmh':[100.,90.,80.],'capacity_vph':[6000.,4000.,3000.]},index=['x','y','z']))
    model=BoundedState(ExtremeState());out=model.predict(features,template)
    np.testing.assert_array_equal(out.speed_kmh,[0.,94.5,80.]);np.testing.assert_array_equal(out.flow_vph,[0.,4200.,2500.])
    pd.testing.assert_frame_equal(out[STATE_KEY],template[STATE_KEY])
    path=tmp_path/'tree.pkl';model.save(path)
    pd.testing.assert_frame_equal(StateModel.load(path).predict(features,template),out)
