from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from trafficflow.state import StateModel
from trafficflow.state_ensemble import StateEnsemble
from trafficflow.data import STATE_KEY, VALUES


class Member(StateModel):
    def __init__(self, offset=0., reverse=False):
        self.profile = SimpleNamespace(slots=pd.DataFrame({'v': [1.]}), fallback=pd.DataFrame({'q': [1.]}))
        self.feature_version = 2
        self.offset = offset
        self.reverse = reverse
        self._network = object()

    def predict(self, features, template):
        result = template[STATE_KEY].copy()
        result[VALUES] = np.array([[60., 3000.], [90., 2700.]])+self.offset
        return result.iloc[::-1] if self.reverse else result


def targets():
    return pd.DataFrame({'panel': ['p', 'p'], 'timestamp': ['t', 't'], 'station_id': ['a', 'b'],
        'link_id': ['a', 'b'], 'mask_regime': ['R1', 'R1']})


def test_state_ensemble_key_contract_and_portable_save(tmp_path):
    members = [Member(), Member(20.)]
    model = StateEnsemble(members, [1., 3.])
    output = model.predict(None, targets())
    np.testing.assert_allclose(output[VALUES], [[75., 3015.], [105., 2715.]])
    path = tmp_path/'ensemble.pkl'
    original_caches = [m._network for m in members]
    model.save(path)
    assert [m._network for m in members] == original_caches
    loaded = StateModel.load(path)
    assert all(m._network is None for m in loaded.models)
    pd.testing.assert_frame_equal(output, loaded.predict(None, targets()))
    with pytest.raises(ValueError, match='key order'):
        StateEnsemble([Member(), Member(reverse=True)], [1., 1.]).predict(None, targets())


def test_state_ensemble_rejects_incompatible_profiles_and_weights():
    incompatible = Member()
    incompatible.profile.slots.iloc[0, 0] = 2.
    with pytest.raises(ValueError, match='same profile'):
        StateEnsemble([Member(), incompatible], [1., 1.])
    for weights in ([0., 0.], [1., -1.], [np.nan, 1.], [1.]):
        with pytest.raises(ValueError, match='weights'):
            StateEnsemble([Member(), Member()], weights)
