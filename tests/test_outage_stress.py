import numpy as np
import pandas as pd
import pytest
from trafficflow.data import eligible
from trafficflow.outage_stress import outage_geometry,remove_measurements,MEASUREMENTS


def frame():
    dates=pd.date_range('2030-11-13',periods=288,freq='5min',tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    f=pd.DataFrame([(t,link,float(pos)) for t in dates for link,pos in [('z',1),('b',2),('a',3),('x',4),('c',5)]],
        columns=['timestamp','link_id','milepost'])
    for key in MEASUREMENTS:f[key]=80. if key!='pct_observed' else 100.
    for key in ['is_observed','is_score_eligible']:f[key]=True
    for key in ['is_imputed','is_missing']:f[key]=False
    return f


def test_outage_selection_is_label_independent_and_nested():
    f=frame();small,g=outage_geometry(f,'p','2030-11-13',31,36,.3)
    big,_=outage_geometry(f,'p','2030-11-13',31,72,.7)
    assert np.all(~small|big) and small.sum()==36*2 and big.sum()==72*4
    changed=f.copy();changed[MEASUREMENTS]=np.nan
    again,metadata=outage_geometry(changed,'p','2030-11-13',31,36,.3)
    np.testing.assert_array_equal(small,again);assert g==metadata
    selected_positions=f.loc[small].groupby('link_id').milepost.first().sort_values().to_numpy()
    assert np.all(np.diff(selected_positions)==1.)


@pytest.mark.parametrize('integer_flags',[False,True])
def test_outage_removes_every_measurement_and_preserves_original_input(integer_flags):
    f=frame()
    if integer_flags:
        for column in ['is_observed','is_imputed','is_score_eligible','is_missing']:f[column]=f[column].astype('int64')
    old=f.copy(deep=True);selected,_=outage_geometry(f,'p','2030-11-13',31,72,.7)
    hidden=remove_measurements(f,selected)
    pd.testing.assert_frame_equal(f,old)
    assert hidden.loc[selected,MEASUREMENTS].isna().all().all() and not eligible(hidden)[selected].any()
    assert hidden.loc[selected,'is_missing'].all() and not hidden.loc[selected,'is_observed'].any()
    pd.testing.assert_frame_equal(hidden.loc[~selected],f.loc[~selected])
    with pytest.raises(ValueError,match='shape'):remove_measurements(f,np.zeros(4,bool))


def test_outage_rejects_partial_days_and_unregistered_sizes():
    with pytest.raises(ValueError,match='Complete'):outage_geometry(frame().iloc[:20],'p','2030-11-13',31,36,.3)
    with pytest.raises(ValueError,match='sizes'):outage_geometry(frame(),'p','2030-11-13',31,35,.3)
