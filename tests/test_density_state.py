from types import SimpleNamespace
import numpy as np
import pandas as pd
from trafficflow.density_state import DensityStateModel
from trafficflow.data import VALUES


class Baseline:
    profile=None
    feature_version=2
    def predict(self,features,template):
        result=template.copy()
        result[VALUES]=[[60.,3000.],[200.,10000.]]
        return result


class Density:
    def predict(self,x):return np.zeros(len(x))


def test_density_projection_preserves_zero_blend_and_bounds():
    template=pd.DataFrame({'link_id':['a','b']})
    net=pd.DataFrame({'lanes':[3.,3.],'free_speed_kmh':[100.,100.],
                      'capacity_vph':[6000.,6000.],'k_jam':[300.,300.]},index=['a','b'])
    features=SimpleNamespace(net=net,features=pd.DataFrame({'density_base':[.8,.8]}),lookup=lambda t:np.arange(len(t)))
    m=DensityStateModel(Baseline(),Density(),0.)
    zero=m.predict(features,template)
    np.testing.assert_allclose(zero[VALUES],[[60,3000],[105,6300]])
    m.blend=1.
    projected=m.predict(features,template)
    np.testing.assert_allclose(projected.flow_vph/projected.speed_kmh,[60.,60.])
    assert (projected.speed_kmh<=105).all() and (projected.flow_vph<=6300).all()
    assert np.isfinite(projected[VALUES]).all().all()
