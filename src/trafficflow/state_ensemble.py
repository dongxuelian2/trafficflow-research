"""Fixed-weight State ensembles with strict shared-feature and key contracts."""
import numpy as np
from .data import STATE_KEY, VALUES
from .state import StateModel


class StateEnsemble(StateModel):
    def __init__(self, models, weights):
        self.models = list(models)
        self.weights = np.asarray(weights, float)
        if not self.models or self.weights.shape != (len(self.models),) or not np.isfinite(self.weights).all() or (self.weights < 0).any() or self.weights.sum() <= 0:
            raise ValueError('Finite nonnegative ensemble weights with positive sum required')
        if not all(isinstance(model, StateModel) for model in self.models):
            raise ValueError('Only trusted State model artifacts may be combined')
        self.weights = self.weights/self.weights.sum()
        self.profile = self.models[0].profile
        self.feature_version = self.models[0].feature_version
        for model in self.models[1:]:
            if model.feature_version != self.feature_version or not model.profile.slots.equals(self.profile.slots) or not model.profile.fallback.equals(self.profile.fallback):
                raise ValueError('Ensemble members require the same profile and feature version')

    def predict(self, features, template):
        keys = template[STATE_KEY].reset_index(drop=True)
        values = np.zeros((len(template), 2), float)
        for weight, model in zip(self.weights, self.models):
            prediction = model.predict(features, template)
            if not prediction[STATE_KEY].reset_index(drop=True).equals(keys):
                raise ValueError('Ensemble member changed prediction key order')
            value = prediction[VALUES].to_numpy(float)
            if not np.isfinite(value).all() or (value < 0).any():
                raise ValueError('Invalid ensemble member prediction')
            values += weight*value
        result = template[STATE_KEY].copy()
        result[VALUES] = values
        return result

    def save(self, path):
        caches = [(model, model._network) for model in self.models if hasattr(model, '_network')]
        for model, _ in caches:
            model._network = None
        try:
            super().save(path)
        finally:
            for model, network in caches:
                model._network = network
