import numpy as np
import pytest
from trafficflow.onset_rank_queue import onset_topk_decisions


def domain():
    return np.repeat(['a','b'],18),np.tile(np.repeat([5,10,15,20,25,30],3),2),np.tile(['z','a','m'],12)


def test_rank_scores_and_deterministic_full_domain_ties():
    w,t,l=domain();s=np.full(36,999.);s[t==30]=[-3,-1,-1,8,2,4]
    p=onset_topk_decisions(s,w,t,l,1)
    assert p.sum()==2 and not p[t<30].any()
    assert l[(w=='a')&(p==1)].tolist()==['a'] and l[(w=='b')&(p==1)].tolist()==['z']
    order=np.random.default_rng(82).permutation(36);q=onset_topk_decisions(s[order],w[order],t[order],l[order],1)
    np.testing.assert_array_equal(q,p[order])


def test_forecast_domain_cannot_be_future_eligible_subset():
    w,t,l=domain();s=np.arange(36,dtype=float)
    with pytest.raises(ValueError):onset_topk_decisions(s[:-1],w[:-1],t[:-1],l[:-1],1)
    l=l.copy();l[-1]=l[-2]
    with pytest.raises(ValueError):onset_topk_decisions(s,w,t,l,1)
    for k in [0,-1,1.5]:
        with pytest.raises(ValueError):onset_topk_decisions(s,w,t,l,k)


def test_set_size_cap_and_no_label_or_quality_argument():
    w,t,l=domain();s=np.arange(36,dtype=float);p=onset_topk_decisions(s,w,t,l,99)
    assert p.sum()==6 and not p[t<30].any()
    with pytest.raises(TypeError):onset_topk_decisions(s,w,t,l,1,future_eligible=np.ones(36,bool))
    s[0]=np.nan
    with pytest.raises(ValueError):onset_topk_decisions(s,w,t,l,1)
