"""Timestamp-filtered public Task 2 inputs: never read values after origin T."""
import re
import pandas as pd
from .data import Release, eligible


def released_origin_rows(release:Release,panel,split,origin):
    origin=pd.Timestamp(origin)
    if origin.tzinfo is None:raise ValueError('Origin must be timezone aware')
    origin=origin.tz_convert('UTC')
    token=origin.strftime('%Y-%m-%dT%H:%M:%SZ')
    day=token[:10]
    chosen=[]
    for path in release.files(panel,split,masked=True):
        match=re.search(r'(20\d\d)[_-](\d\d)[_-](\d\d)',path.name)
        if match and '-'.join(match.groups())==day:chosen.append(path)
    if len(chosen)!=1:raise ValueError('Expected exactly one published masked day partition')
    # Predicate pushdown selects T only. No future-state values enter memory.
    frame=pd.read_parquet(chosen[0],filters=[('timestamp','==',token)])
    if frame.empty:raise ValueError('Forecast origin missing from public day')
    for key in ('timestamp','station_id','link_id'):
        frame[key]=frame[key].astype(str)
    if not frame.timestamp.eq(token).all():raise ValueError('Causal predicate failed')
    return frame


def append_released_origin(history,release,panel,split,origin):
    ts=pd.to_datetime(history.timestamp,utc=True)
    if (ts>=pd.Timestamp(origin)).any():raise ValueError('Published short history must end before T')
    origin_rows=released_origin_rows(release,panel,split,origin)
    return pd.concat([history,origin_rows],ignore_index=True)
