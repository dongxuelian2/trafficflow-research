import pandas as pd
import pytest
from trafficflow.data import Release
from trafficflow.causal_queue import released_origin_rows


def test_origin_predicate_never_returns_horizon_values(tmp_path):
    folder=tmp_path/'corridors'/'P'
    (folder/'network').mkdir(parents=True)
    states=folder/'validation'/'mainline_states_masked'/'mask_regime=R1'
    states.mkdir(parents=True)
    source=pd.DataFrame({'timestamp':['2030-06-01T01:00:00Z','2030-06-01T01:05:00Z'],
        'station_id':['s','s'],'link_id':['l','l'],'speed_kmh':[20.,999.],
        'flow_vph':[1000.,999999.],'pct_observed':[100.,100.]})
    path=states/'synthetic_mainline_2030_06_01.parquet'
    source.to_parquet(path,index=False)
    release=Release(tmp_path)
    first=released_origin_rows(release,'P','validation','2030-06-01T01:00:00Z')
    assert len(first)==1 and first.speed_kmh.iloc[0]==20.
    source.loc[1,['speed_kmh','flow_vph']]=[1.,0.]
    source.to_parquet(path,index=False)
    second=released_origin_rows(release,'P','validation','2030-06-01T01:00:00Z')
    pd.testing.assert_frame_equal(first,second)
    with pytest.raises(ValueError,match='timezone'):
        released_origin_rows(release,'P','validation','2030-06-01T01:00:00')
