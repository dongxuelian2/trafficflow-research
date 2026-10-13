import numpy as np
import pandas as pd
from trafficflow.queue import ProxyData
from trafficflow.queue_event_loss import event_weights


def test_event_condition_and_class_mass_do_not_depend_on_cell_count():
    # One condition has twice the events and very unequal positive set sizes.
    w=np.array(['a']*4+['b']*3+['c']*5)
    c=np.array(['onset']*7+['ongoing']*5)
    y=np.array([1,0,0,0,1,1,0,1,0,0,0,0])
    d=ProxyData(pd.DataFrame(),y,w,c,np.array(['2030-01-01']*12))
    for balanced in (False,True):
        weights=event_weights(d,balanced)
        assert np.all(weights>0) and np.isclose(weights.mean(),1.)
        assert np.isclose(weights[c=='onset'].sum(),weights[c=='ongoing'].sum())
        assert np.isclose(weights[w=='a'].sum(),weights[w=='b'].sum())
        if balanced:
            for event in ('a','b','c'):
                assert np.isclose(weights[(w==event)&(y==1)].sum(),weights[(w==event)&(y==0)].sum())
