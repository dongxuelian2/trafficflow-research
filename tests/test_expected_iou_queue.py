import itertools
import numpy as np
import pytest
from scipy.special import expit
from trafficflow.expected_iou_queue import independent_iou_set,fit_monotone_platt,apply_platt,onset_expected_iou_decisions


def brute(p,pred):
    result=0.
    for truth in itertools.product([0,1],repeat=len(p)):
        y=np.array(truth,bool);weight=np.prod(np.where(y,p,1-p));union=np.sum(y|pred)
        result+=weight*(np.sum(y&pred)/union if union else 1.)
    return result


@pytest.mark.parametrize('allow_empty',[False,True])
def test_exact_expectations_and_global_best_match_exhaustive_sets(allow_empty):
    for p in [np.array([0.,1.,.5]),np.array([.1,.1,.1]),np.array([.8,.4,.3]),np.random.default_rng(79).uniform(size=6)]:
        links=np.array([str(i) for i in range(len(p))]);pred,detail=independent_iou_set(p,links,allow_empty)
        possible=[np.array(x,bool) for x in itertools.product([0,1],repeat=len(p)) if allow_empty or sum(x)>0]
        optimum=max(brute(p,x) for x in possible)
        assert brute(p,pred.astype(bool))==pytest.approx(optimum,abs=1e-12)
        order=np.lexsort((links,-p))
        for k,expected in enumerate(detail['expected_by_size']):
            candidate=np.zeros(len(p),bool);candidate[order[:k]]=True
            assert expected==pytest.approx(brute(p,candidate),abs=1e-12)


def test_empty_ties_permutation_and_complete_onset_domain():
    assert independent_iou_set([0,0],['b','a'])[0].sum()==0
    assert independent_iou_set([.5],['a'])[0].sum()==0
    p=np.array([.7,.7,.1]);links=np.array(['c','a','b']);a=independent_iou_set(p,links)[0];order=np.array([2,0,1])
    np.testing.assert_array_equal(independent_iou_set(p[order],links[order])[0],a[order])
    window=np.full(18,'w');lead=np.repeat([5,10,15,20,25,30],3);ids=np.tile(links,6);score=np.tile(p,6)
    out=onset_expected_iou_decisions(score,window,lead,ids)
    assert not out[lead<30].any();np.testing.assert_array_equal(out[-3:],a)
    with pytest.raises(ValueError):onset_expected_iou_decisions(score[:-1],window[:-1],lead[:-1],ids[:-1])
    with pytest.raises(TypeError):onset_expected_iou_decisions(score,window,lead,ids,future_eligible=np.ones(18,bool))


def test_calibration_window_weighting_gradient_and_monotonicity():
    p=np.full(12,.5);y=np.tile([0,0,0,1],3);w=np.repeat(['a','b','c'],4);state=fit_monotone_platt(p,y,w)
    assert state['fitted'] and state['slope']==pytest.approx(1.) and state['intercept']<0
    assert expit(state['intercept'])-.25+.01*state['intercept']==pytest.approx(0,abs=1e-8)
    repeated=np.r_[np.arange(12),np.arange(4)]
    other=fit_monotone_platt(p[repeated],y[repeated],w[repeated])
    assert other['intercept']==pytest.approx(state['intercept'],abs=1e-8)
    assert np.all(np.diff(apply_platt(state,[0,.1,.5,.9,1]))>0)
    fallback=fit_monotone_platt(p[:4],y[:4],w[:4]);assert not fallback['fitted'] and fallback['slope']==1 and fallback['intercept']==0
