import numpy as np
import pytest
from trafficflow.ctm_queue_diagnostics import causal_speed_queue_forecasts


def test_causal_trend_extrapolates_missing_current_slots_from_legal_past():
    v=(110-3*np.arange(13))[:,None].astype(float);quality=np.ones_like(v,bool)
    quality[-2:]=False;v[-2:]=np.nan
    result=causal_speed_queue_forecasts(v,quality,110.,'queue_ongoing')
    np.testing.assert_array_equal(result['linear_trend'].ravel(),[0,0,1,1,1,1])
    assert not result['persistence'].any()
    onset=causal_speed_queue_forecasts(v,quality,110.,'queue_onset')
    np.testing.assert_array_equal(onset['linear_trend'].ravel(),[0,0,0,0,0,1])


def test_history_only_forecaster_rejects_future_slots_and_has_missing_fallback():
    with pytest.raises(ValueError,match='thirteen'):
        causal_speed_queue_forecasts(np.ones((19,2)),np.ones((19,2),bool),110.,'queue_onset')
    result=causal_speed_queue_forecasts(np.full((13,2),np.nan),np.zeros((13,2),bool),110.,'queue_ongoing')
    assert not result['persistence'].any() and not result['linear_trend'].any()
