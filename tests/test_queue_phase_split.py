import pandas as pd
import pytest
from trafficflow.queue_phase_split import queue_phase_dates,validate_phase_origins


def test_five_phases_are_disjoint_complete_and_chronological():
    days=[d.date().isoformat() for d in pd.date_range('2030-06-01','2031-02-28')]
    result=queue_phase_dates(days,'2030-07-16','2030-11-30')
    assert result['checkpoint']==[d.date().isoformat() for d in pd.date_range('2030-12-01','2030-12-15')]
    assert result['decision'][-1]=='2031-01-14' and result['selection'][0]=='2031-01-15'
    assert result['assessment'][0]=='2031-01-30' and result['assessment'][-1]=='2031-02-28'
    assert len(set(d for phase in result.values() for d in phase))==sum(map(len,result.values()))


def test_phase_validator_rejects_wrong_period_and_missing_public_calendar():
    days=[d.date().isoformat() for d in pd.date_range('2030-07-16','2031-02-28')]
    result=queue_phase_dates(days,'2030-07-16','2030-11-30')
    with pytest.raises(ValueError,match='outside'):
        validate_phase_origins('decision',['2031-01-15T01:00:00Z'],result)
    with pytest.raises(ValueError,match='incomplete'):
        queue_phase_dates(days[:-1],'2030-07-16','2030-11-30')
