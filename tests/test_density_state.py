from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from trafficflow.density_state import DensityStateModel,InterpolatedDensityResidual
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
    template=pd.DataFrame({'panel':['p','p'],'timestamp':['t','t'],'station_id':['a','b'],
                          'link_id':['a','b'],'mask_regime':['R1','R1']})
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
    cached=Baseline().predict(features,template)
    pd.testing.assert_frame_equal(projected,m.project(features,template,cached))
    np.testing.assert_allclose(cached[VALUES],[[60.,3000.],[200.,10000.]])
    with pytest.raises(ValueError,match='not aligned'):
        m.project(features,template,cached.iloc[::-1])


def test_temporal_density_falls_back_when_masked_evidence_is_unavailable():
    x=pd.DataFrame({'density_base':[.4,.8],'density_temporal':[.6,np.nan]})
    np.testing.assert_allclose(InterpolatedDensityResidual('temporal').predict(x),[.2,0.])
    np.testing.assert_allclose(InterpolatedDensityResidual('base').predict(x),[0.,0.])
