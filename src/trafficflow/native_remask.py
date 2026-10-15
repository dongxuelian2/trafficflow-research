"""Public-fit-only remasking at the official point-removal rates.

Random coordinates and optional mock Queue origins are metadata-only. These
are own training views,not the official stable hash or Queue event selector.
"""
import zlib
import numpy as np
import pandas as pd
from .data import eligible
from .outage_stress import remove_measurements


RATES={'R1':.2,'R2':.3,'R3':.5}


def native_remask(raw,panel,date,seed,regime,public_origins=(),mock_queue=False):
    if regime not in RATES:raise ValueError('Official R1/R2/R3 rate required')
    times=sorted(raw.timestamp.unique());links=sorted(raw.link_id.unique())
    dt=pd.to_datetime(times,utc=True)
    if len(times)!=288 or not (np.diff(dt.as_unit('s').asi8)==300).all() or str(times[0])[:10]!=date:
        raise ValueError('Complete released five-minute day required')
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(date.replace('-','')),zlib.crc32(panel.encode())]))
    point=rng.random((len(times),len(links)))<RATES[regime]
    ti={t:i for i,t in enumerate(times)};li={link:i for i,link in enumerate(links)}
    row_time=raw.timestamp.map(ti).to_numpy(int);row_link=raw.link_id.map(li).to_numpy(int)
    selected=point[row_time,row_link];history=np.zeros(len(raw),bool);guard=np.zeros(len(raw),bool)
    origins=list(public_origins);mock_origin=None
    if mock_queue:
        queue_rng=np.random.default_rng(np.random.SeedSequence([seed,int(date.replace('-','')),zlib.crc32(panel.encode()),991]))
        origin_index=int(queue_rng.integers(12,270));mock_origin=times[origin_index];origins.append(mock_origin)
    timestamps=pd.to_datetime(raw.timestamp,utc=True)
    for value in origins:
        origin=pd.Timestamp(value)
        history|=timestamps.between(origin-pd.Timedelta(minutes=60),origin).to_numpy()
        guard|=timestamps.between(origin+pd.Timedelta(minutes=5),origin+pd.Timedelta(minutes=90)).to_numpy()
    hidden=(selected&~history)|guard
    target=selected&~history&~guard&eligible(raw)
    frame=remove_measurements(raw,hidden)
    return frame,target,{'point_rows':selected,'history_rows':history,'guard_rows':guard,'hidden_rows':hidden,
        'mock_origin':mock_origin,'public_origins':list(public_origins),'nominal_point_rate':RATES[regime]}
