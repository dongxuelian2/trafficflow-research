import pandas as pd
import numpy as np
from trafficflow.queue_events import select_event_candidates


def test_balanced_conditions_preserve_spacing_and_prevent_dense_class_starvation():
    candidates=[]
    for day in pd.date_range('2030-12-01','2030-12-15',tz='UTC'):
        candidates.append({'condition':'queue_onset','origin':day+pd.Timedelta(hours=9)})
        for minute in range(9*60+5,20*60,5):
            candidates.append({'condition':'queue_ongoing','origin':day+pd.Timedelta(minutes=minute)})
    selected,counts=select_event_candidates(candidates,4129,30,360,True)
    assert counts['queue_onset']>=8 and counts['queue_ongoing']>0
    assert all(abs((a['origin']-b['origin']).total_seconds())>=360*60 for i,a in enumerate(selected) for b in selected[i+1:])


def test_default_selector_preserves_legacy_permutation_and_does_not_mutate_rows():
    rows=[{'condition':'queue_onset' if i%2 else 'queue_ongoing','origin':pd.Timestamp('2030-12-01T00:00:00Z')+pd.Timedelta(hours=i*7)} for i in range(6)]
    expected=[rows[i] for i in np.random.default_rng(411).permutation(6)]
    selected,counts=select_event_candidates(rows,411,10,360,False)
    assert selected==expected and counts=={'queue_onset':3,'queue_ongoing':3}
    assert rows[0]['origin']==pd.Timestamp('2030-12-01T00:00:00Z')
