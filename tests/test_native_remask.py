import numpy as np
import pandas as pd
import pytest
from trafficflow.data import eligible
from trafficflow.native_remask import native_remask


def source():
    times=pd.date_range('2030-08-01',periods=288,freq='5min',tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    f=pd.DataFrame([(t,str(j),float(j)) for t in times for j in range(8)],columns=['timestamp','link_id','milepost'])
    f['speed_kmh']=70.;f['flow_vph']=1400.;f['occupancy']=.12;f['density_occ_linear_vehpkm']=40.;f['pct_observed']=100.
    f['is_observed']=1;f['is_score_eligible']=1;f['is_missing']=0;f['is_imputed']=0
    return f


def test_native_coordinate_mask_does_not_depend_on_measurements_or_row_order():
    raw=source();frame,target,meta=native_remask(raw,'p','2030-08-01',17,'R3')
    changed=raw.assign(speed_kmh=999.,flow_vph=999999.)
    _,other,again=native_remask(changed,'p','2030-08-01',17,'R3')
    np.testing.assert_array_equal(target,other);np.testing.assert_array_equal(meta['point_rows'],again['point_rows'])
    shuffled=raw.sample(frac=1,random_state=31)
    _,_,reordered=native_remask(shuffled,'p','2030-08-01',17,'R3')
    np.testing.assert_array_equal(pd.Series(reordered['point_rows'],index=shuffled.index).sort_index(),meta['point_rows'])
    assert .45<target.mean()<.55 and not eligible(frame)[target].any()
    assert frame.loc[target,['speed_kmh','flow_vph','occupancy','density_occ_linear_vehpkm','pct_observed']].isna().all().all()
    pd.testing.assert_frame_equal(raw,source())


def test_queue_history_is_visible_guard_is_hidden_and_neither_is_supervised():
    raw=source();origin='2030-08-01T23:30:00Z'
    point,point_target,p=native_remask(raw,'p','2030-08-01',19,'R2')
    queued,target,q=native_remask(raw,'p','2030-08-01',19,'R2',[origin],True)
    np.testing.assert_array_equal(p['point_rows'],q['point_rows'])
    assert not target[q['history_rows']|q['guard_rows']].any()
    assert not eligible(queued)[q['guard_rows']].any()
    assert eligible(queued)[q['history_rows']&~q['guard_rows']].all()
    np.testing.assert_array_equal(target,q['point_rows']&~q['history_rows']&~q['guard_rows'])
    assert q['mock_origin'] is not None and target.sum()<point_target.sum()
    # Poisoning every loss target cannot leak through removed channels.
    poison=raw.copy();poison.loc[target,['speed_kmh','flow_vph','occupancy','density_occ_linear_vehpkm']]=1e8
    rebuilt,mask,_=native_remask(poison,'p','2030-08-01',19,'R2',[origin],True)
    pd.testing.assert_frame_equal(queued,rebuilt);np.testing.assert_array_equal(mask,target)
    previous='2030-07-31T23:30:00Z'
    clipped,loss,cross=native_remask(raw,'p','2030-08-01',19,'R2',[previous])
    assert cross['guard_rows'][raw.timestamp=='2030-08-01T00:00:00Z'].all()
    assert not eligible(clipped)[cross['guard_rows']].any() and not loss[cross['guard_rows']].any()
    with pytest.raises(ValueError,match='Complete'):native_remask(raw.iloc[:20],'p','2030-08-01',9,'R1')
