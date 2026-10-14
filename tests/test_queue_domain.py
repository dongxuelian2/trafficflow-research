import numpy as np
import pandas as pd
from trafficflow.queue_specialist import event_prefix_decisions
from trafficflow.queue_domain import predict_full_forecast,score_full_forecast


class Prefix:
    def classify(self,x,window,condition):
        return event_prefix_decisions(x.probability.to_numpy(),window,condition,x.lead_minutes.to_numpy())


def test_future_eligibility_only_changes_scoring_not_full_set_predictions():
    x=pd.DataFrame({'probability':[.9,.4],'lead_minutes':[30,30]})
    window=np.array(['w','w']);condition=np.array(['queue_onset']*2)
    full=predict_full_forecast(Prefix(),x,window,condition)
    np.testing.assert_array_equal(full,[1,0])
    mask=np.array([False,True]);labels=np.array([0,1])
    assert score_full_forecast(labels,full,window,condition,mask)['proxy_IoU']==0.
    # The old filtered-domain procedure uses future availability to move its
    # forced prefix to another cell, spuriously scoring1 on the same target.
    filtered=Prefix().classify(x.loc[mask],window[mask],condition[mask])
    np.testing.assert_array_equal(filtered,[1])
    np.testing.assert_array_equal(predict_full_forecast(Prefix(),x,window,condition),full)


def test_window_with_no_eligible_cells_is_zero_not_empty_empty_perfect_mark():
    result=score_full_forecast(np.array([0,0]),np.array([0,0]),np.array(['w','w']),
        np.array(['queue_ongoing']*2),np.array([False,False]))
    assert result['proxy_IoU']==0.
