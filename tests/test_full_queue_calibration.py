import numpy as np
from trafficflow.full_queue_calibration import full_queue_decisions,FullForecastScorer,tune_full_probabilities
from trafficflow.queue_domain import score_full_forecast


def test_onset_time_constraint_precedes_nonempty_and_does_not_need_score_mask():
    p=np.array([.9,.1]);window=np.array(['w','w']);condition=np.array(['queue_onset']*2);lead=np.array([5,30])
    old=full_queue_decisions(p,window,condition,lead,{'global':.5},True,'after')
    new=full_queue_decisions(p,window,condition,lead,{'global':.5},True,'before')
    np.testing.assert_array_equal(old,[0,0]);np.testing.assert_array_equal(new,[0,1])


def test_fast_calibration_score_matches_window_scorer_with_missing_and_empty_evidence():
    y=np.array([1,0,0,0,1,0]);pred=np.array([1,1,0,0,0,1])
    window=np.array(['a','a','b','b','c','c']);condition=np.array(['queue_onset']*4+['queue_ongoing']*2)
    mask=np.array([1,1,0,0,1,0],bool)
    assert FullForecastScorer(y,window,condition,mask).score(pred)==score_full_forecast(y,pred,window,condition,mask)['proxy_IoU']


def test_full_calibration_retains_unscored_high_probability_candidate():
    p=np.array([.9,.4]);window=np.array(['w','w']);condition=np.array(['queue_onset']*2);lead=np.array([30,30])
    y=np.array([0,1]);mask=np.array([False,True])
    result=tune_full_probabilities(p,window,condition,lead,y,mask)
    selected=result['selected']
    pred=full_queue_decisions(p,window,condition,lead,selected['thresholds'],selected['require_nonempty'],selected['timing'])
    assert FullForecastScorer(y,window,condition,mask).score(pred)==selected['calibration_proxy_IoU']
    # Future mask did not shrink the returned prediction domain.
    assert len(pred)==2


def test_deployable_queue_rejects_future_measurements_and_incomplete_templates():
    from types import SimpleNamespace
    import pandas as pd
    import pytest
    from trafficflow.full_domain_queue import FullDomainQueueModel
    base=SimpleNamespace(seed=1,iterations=1,threads=1)
    setting={'thresholds':{'global':.5},'require_nonempty':False,'timing':'before'}
    model=FullDomainQueueModel(base,['a'],setting)
    origin=pd.Timestamp('2030-11-01T00:00:00Z')
    target=pd.DataFrame({'timestamp':[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]],'link_id':'a'})
    with pytest.raises(ValueError,match='Future Queue'):
        model.predict_window(pd.DataFrame({'timestamp':['2030-11-01T00:05:00Z']}),target,None,None,origin,'queue_onset')
    with pytest.raises(ValueError,match='complete public'):
        model.predict_window(pd.DataFrame({'timestamp':['2030-11-01T00:00:00Z']}),target.iloc[:-1],None,None,origin,'queue_onset')
