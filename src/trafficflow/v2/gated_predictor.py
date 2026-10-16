"""Explicit research gate. Frozen predictor remains the deployment fallback."""
def state_promotion(gain, interval, panel_gains):
    return bool(gain >= .002 and interval is not None and interval[0] > 0 and min(panel_gains.values()) >= -.001)


class GatedPredictor:
    def __init__(self, baseline, challenger=None, promoted=False):
        if promoted and challenger is None:
            raise ValueError('Promotion needs an independently validated challenger')
        self.baseline, self.challenger, self.promoted = baseline, challenger, promoted

    def predict(self, *args, **kwargs):
        return (self.challenger if self.promoted else self.baseline).predict(*args, **kwargs)
