import numpy as np
from trafficflow.density_label_metrics import public_increment_stats, aggregate_public_increments


def test_public_increment_metric_matches_hand_calculated_all_pair_denominator():
    truth = np.array([[[4.,1.]],[[4.,2.]],[[4.,6.]],[[4.,8.]]])
    day = {'y_full':truth,'target_indices':np.array([1]),'regime':'R1',
           'lwr_pair_valid':np.ones((3,1),bool)}
    measured = public_increment_stats(day,np.array([[100.,1800.]]),np.array([1.]))
    assert measured['error_L1'] == 12.
    assert measured['observation_increment_L1'] == 42.
    # First and final pairs only: target error6, denominator6+12.
    day['lwr_pair_valid'][1] = False
    reduced = public_increment_stats(day,np.array([[100.,1800.]]),np.array([1.]))
    assert reduced['error_L1'] == 6. and reduced['observation_increment_L1'] == 18.


def test_public_increment_aggregation_uses_regime_mean_not_pooled_ratio():
    rows = [{'regime':r,'error_L1':e,'observation_increment_L1':d,'n_public_valid_pairs':1}
            for r,e,d in [('R1',1.,2.),('R2',3.,6.),('R3',0.,3.)]]
    result = aggregate_public_increments(rows)
    np.testing.assert_allclose(result['public_coverage_increment_relative_L1'],1./3.)
    assert not np.isclose(result['public_coverage_increment_relative_L1'],4./11.)
