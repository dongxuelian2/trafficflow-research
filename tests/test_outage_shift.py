import numpy as np
import pandas as pd
import pytest
from trafficflow.outage_shift import SHIFT_CASES,shifted_outage_masks


def day(date):
    times=pd.date_range(date,periods=288,freq='5min',tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    f=pd.DataFrame([(t,str(j),float(j)) for t in times for j in range(10)],columns=['timestamp','link_id','milepost'])
    f['speed_kmh']=80.;f['flow_vph']=2000.;f['pct_observed']=100.
    return f


@pytest.mark.parametrize('case',SHIFT_CASES,ids=lambda c:c['name'])
def test_shift_geometry_uses_only_metadata_and_handles_day_edges(case):
    frames=[day('2030-12-11')]
    if case['placement']=='cross_midnight':frames.append(day('2030-12-12'))
    masks,metadata=shifted_outage_masks(frames,'panel','2030-12-11',99,case)
    poison=[f.assign(speed_kmh=-9999.,flow_vph=np.nan,pct_observed=0.) for f in frames]
    repeated,other=shifted_outage_masks(poison,'panel','2030-12-11',99,case)
    assert metadata==other
    for a,b in zip(masks,repeated):np.testing.assert_array_equal(a,b)
    if case['placement']!='scattered':
        assert sum(m.sum() for m in masks)==case['steps']*int(np.ceil(10*case['fraction']))
    else:assert 0<sum(masks[0])<len(masks[0])
    if case['placement']=='start':assert metadata['segments'][0]['start_index']==0
    if case['placement']=='end':assert metadata['segments'][0]['start_index']+case['steps']==288
    if case['placement']=='cross_midnight':assert [r['steps'] for r in metadata['segments']]==[12,36]


def test_shift_geometry_rejects_nonconsecutive_cross_midnight_and_partial_days():
    with pytest.raises(ValueError,match='consecutive'):shifted_outage_masks([day('2030-12-11'),day('2030-12-13')],'p','2030-12-11',9,SHIFT_CASES[-1])
    with pytest.raises(ValueError,match='Complete'):shifted_outage_masks([day('2030-12-11').iloc[:20]],'p','2030-12-11',9,SHIFT_CASES[0])
