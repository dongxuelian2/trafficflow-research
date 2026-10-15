import itertools
import numpy as np
import pytest
from trafficflow.scenario_local_search import bounded_scenario_search,scenario_set_scores,accept_on_validation


def counterexample():
    y=np.array([[0,1,0,0,0],[1,0,1,1,0],[1,0,1,0,1],[1,0,0,1,1]],bool);w=np.array([.41,.59/3,.59/3,.59/3]);return y,w@y,np.ones(y.shape,bool),np.ones(4,bool),w


def test_local_search_closes_known_nonprefix_gap_with_global_exhaustion():
    y,p,m,e,w=counterexample();pred,record=bounded_scenario_search(p,list('ABCDE'),y,m,e,w)
    maximum=max(np.dot(w,scenario_set_scores(candidate,y,m,e)) for candidate in itertools.product([0,1],repeat=5))
    assert record['initial_expected_proxy_IoU']==pytest.approx(.436)
    assert record['expected_proxy_IoU']==pytest.approx(maximum)==pytest.approx(.4425)
    assert record['moves'][0]['link_id']=='B' and record['moves'][0]['operation']=='remove'
    assert record['stop_reason']=='single_flip_stationary' and not record['global_optimum_claimed']


def test_budget_monotonicity_stationarity_and_permutation():
    y,p,m,e,w=counterexample();pred,record=bounded_scenario_search(p,list('ABCDE'),y,m,e,w,max_moves=0)
    assert not record['moves'] and record['n_neighbor_evaluations']==0
    order=np.array([4,2,0,3,1]);a,d=bounded_scenario_search(p,list('ABCDE'),y,m,e,w);b,f=bounded_scenario_search(p[order],np.array(list('ABCDE'))[order],y[:,order],m[:,order],e,w)
    np.testing.assert_array_equal(b,a[order]);assert [r['link_id'] for r in d['moves']]==[r['link_id'] for r in f['moves']]
    for j in range(len(a)):
        changed=a.copy();changed[j]=1-changed[j]
        assert np.dot(w,scenario_set_scores(changed,y,m,e))<=d['expected_proxy_IoU']+1e-6


def test_separate_scenario_validation_rejects_overfit_and_accepts_supported_move():
    y,p,m,e,w=counterexample();proposal,d=bounded_scenario_search(p,list('ABCDE'),y,m,e,w);prefix=d['prefix_prediction']
    rejected,r=accept_on_validation(prefix,proposal,np.ones((8,5),bool),np.ones((8,5),bool),np.ones(8,bool));assert not r['accepted'];np.testing.assert_array_equal(rejected,prefix)
    supported,s=accept_on_validation(prefix,proposal,np.tile(y[1],(8,1)),np.ones((8,5),bool),np.ones(8,bool));assert s['accepted'];np.testing.assert_array_equal(supported,proposal)
