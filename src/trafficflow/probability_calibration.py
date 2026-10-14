"""Analytical class-weight correction, separate from fitted calibration."""
import numpy as np
from scipy.special import expit


def weighted_bce_probability(logits, positive_weight):
    """Invert ideal positive-weight BCE odds: logit(p)=logit(r)-log(w).

    A finite neural model, auxiliary loss and event sampling can still produce
    miscalibration. This transformation is a testable candidate, not a claim
    that its output equals a traffic-event posterior.
    """
    value = np.asarray(logits, dtype=float)
    weight = float(positive_weight)
    if not np.isfinite(weight) or weight <= 0 or not np.isfinite(value).all():
        raise ValueError('Finite logits and a positive class weight are required')
    return expit(value-np.log(weight))
