import numpy as np
import pandas as pd
import pytest
from trafficflow.queue_graph_geometry import travel_time_neighborhoods


def test_public_wave_kernel_direction_and_multi_link_reach():
    # wave = 6000/(360-60) = 20 km/h; each 1 km edge takes 3 minutes.
    net = pd.DataFrame({'length_km': [1., 1., 1.], 'capacity_vph': [6000.]*3,
        'k_jam': [360.]*3, 'critical_density': [60.]*3}, index=['a', 'b', 'c'])
    topo = pd.DataFrame({'outgoing_link_ids': ['b', 'c', '']}, index=net.index)
    up, down = travel_time_neighborhoods(['a', 'b', 'c'], net, topo, 4.)
    np.testing.assert_array_equal(up, [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    np.testing.assert_array_equal(down, up.T)
    longer_up, longer_down = travel_time_neighborhoods(['a', 'b', 'c'], net, topo, 7.)
    expected = np.exp(-np.array([3., 6.])/3.5)
    expected /= expected.sum()
    np.testing.assert_allclose(longer_down[0, 1:], expected, rtol=1e-6)
    np.testing.assert_allclose(longer_up[2, [1, 0]], expected, rtol=1e-6)
    assert longer_down[2].sum() == 0
    assert not np.diag(longer_down).any()
    with pytest.raises(ValueError, match='positive'):
        travel_time_neighborhoods(['a'], net, topo, 0.)
