import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from trafficflow.probability_calibration import weighted_bce_probability


def test_weight_correction_recovers_independently_optimized_bce_risk():
    for truth_probability in (.01, .13, .5, .87, .99):
        for weight in (.25, 1., 4., 20.):
            def risk(predicted):
                return -weight*truth_probability*np.log(predicted)-(1-truth_probability)*np.log1p(-predicted)
            optimum = minimize_scalar(risk, bounds=(1e-10, 1-1e-10), method='bounded', options={'xatol': 1e-13})
            logit = np.log(optimum.x)-np.log1p(-optimum.x)
            assert weighted_bce_probability(logit, weight) == pytest.approx(truth_probability, abs=3e-7)
    values = weighted_bce_probability([-1000., 0., 1000.], 4.)
    np.testing.assert_allclose(values, [0., .2, 1.])


def test_weight_correction_rejects_invalid_predictions_and_weights():
    for weight in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match='Finite logits'):
            weighted_bce_probability([0.], weight)
    for values in ([np.nan], [np.inf]):
        with pytest.raises(ValueError, match='Finite logits'):
            weighted_bce_probability(values, 4.)
