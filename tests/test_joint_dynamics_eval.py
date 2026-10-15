import numpy as np
import pytest
from trafficflow.joint_dynamics_eval import state_sufficient,combine_state


def test_state_subset_formula_units_and_missing_regimes_remain_unknown():
    a={'truth_current':np.zeros((1,2,3)),'current_domain':np.ones((1,2,3),bool),'current_regime':np.array([['R1','R2','R3']])};p=np.ones((1,2,3))*.5
    stats=state_sufficient(a,p);out=combine_state([stats]);assert out['public_subset_regime_macro']==pytest.approx(.5)
    assert out['details'][0]['rmse_speed_kmh']==12.5 and out['details'][0]['rmse_flow_per_lane']==300.
    a['current_domain'][:,:,2]=False;out=combine_state([state_sufficient(a,p)]);assert not out['all_regimes_present'] and out['public_subset_regime_macro'] is None
