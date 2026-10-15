import numpy as np
import pytest
from trafficflow.joint_queue_scenarios import scenario_prefix_set,build_scenario_bank,bank_prefix_set


def hand_score(pred,y,m,e,w):
    scores=[]
    for a,b,has_evidence in zip(y,m,e):
        left=set(np.flatnonzero(a&b));right=set(np.flatnonzero(pred&b));union=left|right
        scores.append(len(left&right)/len(union) if union else float(has_evidence))
    return float(np.dot(np.array(w)/np.sum(w),scores))


def test_every_prefix_matches_scalar_sets_with_evidence_weights_and_permutation():
    p=np.array([.6,.6,.2]);links=np.array(['b','a','c']);y=np.array([[1,0,1],[0,1,0],[0,0,0]],bool);m=np.array([[1,1,0],[0,1,1],[0,0,0]],bool);e=np.array([1,1,0],bool);w=np.array([2.,3.,1.])
    pred,d=scenario_prefix_set(p,links,y,m,e,w);order=np.lexsort((links,-p))
    for k,value in enumerate(d['curve']):
        candidate=np.zeros(3,bool);candidate[order[:k]]=True
        assert value==pytest.approx(hand_score(candidate,y,m,e,w),abs=1e-12)
    assert d['expected_proxy_IoU']==pytest.approx(max(d['curve']))
    z=np.array([2,0,1]);other=scenario_prefix_set(p[z],links[z],y[:,z],m[:,z],e,w)[0];np.testing.assert_array_equal(other,pred[z])


def test_latent_nonempty_does_not_imply_scored_nonempty():
    y=np.eye(4,dtype=bool);m=~y;e=np.ones(4,bool)
    empty,d=scenario_prefix_set(np.full(4,.25),np.arange(4),y,m,e)
    forced,f=scenario_prefix_set(np.full(4,.25),np.arange(4),y,m,e,allow_empty=False)
    assert empty.sum()==0 and d['expected_proxy_IoU']==1
    assert forced.sum()==1 and f['expected_proxy_IoU']==.25
    zero,z=scenario_prefix_set([.9],['a'],[[1]],[[0]],[0]);assert z['expected_proxy_IoU']==0 and zero.sum()==0


def test_fit_bank_replay_ignores_unknown_labels_and_uses_common_draws():
    y=np.array([[1,0,0],[0,1,1],[1,0,1]],bool);m=np.ones((3,6,3),bool);m[1,-1,0]=False
    bank=build_scenario_bank(y,m,seeds=(31,37),draws_per_seed=16);changed=y.copy();changed[1,0]=~changed[1,0]
    other=build_scenario_bank(changed,m,seeds=(31,37),draws_per_seed=16)
    for key in bank:np.testing.assert_array_equal(bank[key],other[key])
    rng=np.random.default_rng(31);idx=rng.integers(0,3,16);uniform=rng.random((16,3));np.testing.assert_array_equal(bank['template_indices'][:16],idx);np.testing.assert_array_equal(bank['independent_uniform'][:16],uniform)
    a,d=bank_prefix_set([.4,.5,.7],['a','b','c'],bank,'copula','empirical');b,f=bank_prefix_set([.4,.5,.7],['a','b','c'],bank,'copula','empirical');np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(d['curve'],f['curve'])
    with pytest.raises(ValueError):scenario_prefix_set([.5],['a'],[[1]],[[1]],[0])
