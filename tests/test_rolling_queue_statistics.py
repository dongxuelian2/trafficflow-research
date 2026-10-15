"""Scientific checks for the rolling-study audit and dependent-date intervals."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_rolling_onset_decoders import synchronized_comparison
from verify_rolling_onset_decoders import verify_baseline_grid
from trafficflow.full_queue_calibration import tune_full_probabilities


def example_case(panel,sign=1,start='2031-01-01'):
    dates=[d.date().isoformat() for d in pd.date_range(start,periods=30)];runs=[]
    for seed in [17,29]:
        left=[];right=[]
        for i,date in enumerate(dates):
            for condition,hour in [('queue_onset',12),('queue_ongoing',18)]:
                window=f'{panel}:{date}T{hour}:00:00Z:{seed}';row={'window':window,'condition':condition}
                left.append({**row,'IoU':.4});right.append({**row,'IoU':.4+sign*(.15 if i%2 else -.15)})
        runs.append({'seed':seed,'models':{'incumbent':{'windows':left},'threshold':{'windows':right}}})
    return {'name':panel,'dates':{'assessment':dates},'assessment':runs}


def test_calendar_synchronization_preserves_cancelling_panel_errors():
    result=synchronized_comparison([example_case('A'),example_case('B',-1)],'incumbent','threshold')
    # Perfect opposite errors must cancel in EVERY shared-date draw. Treating
    # the two panels as independent would incorrectly produce a wide interval.
    np.testing.assert_allclose(result['ci95'],[0,0],atol=1e-12)
    assert result['valid_resamples']==4000 and result['paired_macro_gain']==pytest.approx(0,abs=1e-12)
    assert [r['unique_origins'] for r in result['cases']]==[60,60]


def test_partially_overlapping_different_calendars_are_rejected():
    with pytest.raises(ValueError,match='Overlapping calendars'):
        synchronized_comparison([example_case('A'),example_case('B',start='2031-01-02')],'incumbent','threshold')


def test_independent_calibration_handles_missing_and_empty_scoring_windows():
    w=np.repeat(['a','b','c'],18);condition=np.repeat(['queue_onset','queue_ongoing','queue_ongoing'],18);lead=np.tile(np.repeat([5,10,15,20,25,30],3),3)
    p=np.random.default_rng(97).uniform(size=54);y=np.zeros(54);y[[15,17,20,29,32]]=1;mask=np.ones(54,bool);mask[36:]=False;mask[[1,15,20,22]]=False
    a={'window':w,'condition':condition,'lead_minutes':lead,'labels':y,'available':mask}
    calibration=tune_full_probabilities(p,w,condition,lead,y,mask)
    assert verify_baseline_grid(a,p,calibration)==188
