import numpy as np
import pytest
from trafficflow.ctm_state_adapter import synthetic_state_input


def adapted(v, q, eligible):
    return synthetic_state_input(v, q, eligible, lanes=4., free_speed=100., capacity=6000., jam_density=400.)


def test_ineligible_observation_values_cannot_enter_inputs_or_interpolation():
    v = np.array([[50., 90.], [np.nan, 80.], [70., 60.]])
    q = np.array([[4000., 6000.], [np.nan, 3000.], [2000., 1000.]])
    eligible = np.array([[True, False], [True, False], [True, True]])
    first = adapted(v, q, eligible)
    v[~eligible], q[~eligible] = 999., 888888.
    second = adapted(v, q, eligible)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    x, base, incoming, outgoing, fd = first
    # Missing target interpolates from visible observations only.
    np.testing.assert_allclose(base[1, 0], [60., 750.])
    np.testing.assert_array_equal(x[..., 8:10], 0.)
    np.testing.assert_array_equal(incoming, [[0., 0.], [1., 0.]])
    np.testing.assert_array_equal(outgoing, incoming.T)
    np.testing.assert_array_equal(fd[:, 0], [100., 1500., 100.])


def test_fully_missing_grid_uses_geometry_without_synthetic_truth():
    x, base, *_ = adapted(np.full((3, 2), np.nan), np.full((3, 2), np.nan), np.ones((3, 2), bool))
    np.testing.assert_allclose(base, np.broadcast_to([100., 600.], base.shape))
    np.testing.assert_array_equal(x[..., :4], 0.)
    assert np.isfinite(x).all()
    with pytest.raises(ValueError, match='grids'):
        adapted(np.ones((2, 3)), np.ones((3, 2)), np.ones((2, 3), bool))
