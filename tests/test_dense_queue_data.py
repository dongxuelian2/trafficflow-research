import copy
import numpy as np
import pandas as pd
import pytest
from trafficflow.dense_queue_data import prepare_dense_day,dense_window,dense_origins


def example():
    times=pd.date_range('2030-08-01',periods=288,freq='5min',tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ');frame=pd.MultiIndex.from_product([times,['a','b']],names=['timestamp','link_id']).to_frame(index=False)
    frame['station_id']=frame.link_id;frame['speed_kmh']=75.;frame['flow_vph']=1200.;frame['occupancy']=.1;frame['pct_observed']=100.;masked=frame.copy();masked.loc[masked.link_id=='a',['speed_kmh','flow_vph']]=np.nan
    net=pd.DataFrame({'lanes':[2,4],'v_cut':[60.,60.]},index=['a','b']);return prepare_dense_day(frame,masked,['a','b'],net)


def test_dense_channels_legal_current_mask_and_targets_are_separate():
    pack=example();window=dense_window(pack,30)
    assert window['x'].shape==(13,2,9);np.testing.assert_array_equal(window['x'][:12,:,0],3.)
    assert window['x'][-1,0,0]==0 and window['x'][-1,0,2]==0
    assert window['x'][-1,1,1]==.5 and window['x'][-1,1,2]==1
    np.testing.assert_array_equal(window['truth_speed'],3.);assert not window['labels'].any()


def test_future_and_raw_T_poison_cannot_change_current_input():
    pack=example();before=dense_window(pack,30)['x'];bad=copy.deepcopy(pack)
    bad['raw_channels'][30:]=999;bad['masked_channels'][31:]=999;bad['truth_queue'][30:]=1;bad['truth_speed'][31:]=-999;bad['available'][31:]=False
    np.testing.assert_array_equal(dense_window(bad,30)['x'],before)


def test_all_five_minute_origins_and_history_only_filter():
    pack=example();np.testing.assert_array_equal(dense_origins(pack),np.arange(12,282))
    assert len(dense_origins(pack))==270
    with pytest.raises(ValueError):dense_window(pack,282)
    with pytest.raises(ValueError):dense_window(pack,11)
