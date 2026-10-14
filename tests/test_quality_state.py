from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from trafficflow.quality_state import quality_channels, quality_reconstruction_input


def test_quality_channels_preserve_legal_low_quality_values_and_missing_targets():
    frame = pd.DataFrame({'timestamp': ['a', 'a', 'b', 'b'], 'link_id': ['x', 'y', 'x', 'y'],
        'speed_kmh': [50., np.nan, np.nan, 75.], 'flow_vph': [2400., np.nan, 1200., np.inf],
        'pct_observed': [25., 100., np.nan, 120.]})
    features = SimpleNamespace(frame=frame, times=['a', 'b'], links=['x', 'y'],
        net=pd.DataFrame({'lanes': [4., 2.]}, index=['x', 'y']))
    x = quality_channels(features)
    np.testing.assert_array_equal(x[0, 0], [2., 1., 1., 1., .25, 1.])
    np.testing.assert_array_equal(x[0, 1], [0., 0., 0., 0., 1., 1.])
    np.testing.assert_array_equal(x[1, 0], [0., .5, 0., 1., 0., 0.])
    np.testing.assert_array_equal(x[1, 1], [3., 0., 1., 0., 1., 1.])
    strict = np.arange(48, dtype=np.float32).reshape(2, 2, 12)
    extended = quality_reconstruction_input(features, strict)
    np.testing.assert_array_equal(extended[..., :12], strict)
    assert extended.shape == (2, 2, 18)
    assert np.isfinite(extended).all()
    with pytest.raises(ValueError, match='twelve'):
        quality_reconstruction_input(features, extended)
