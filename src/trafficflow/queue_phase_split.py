"""Chronological, nonoverlapping Queue checkpoint/decision/selection phases."""
import pandas as pd


PHASES=['fit','checkpoint','decision','selection','assessment']


def queue_phase_dates(dates,fit_start,fit_end):
    start,end=pd.Timestamp(fit_start),pd.Timestamp(fit_end)
    if start>end:raise ValueError('Fit start must precede fit end')
    bounds={'fit':(start,end)};cursor=end
    for name,days in [('checkpoint',15),('decision',30),('selection',15),('assessment',30)]:
        bounds[name]=(cursor+pd.Timedelta(days=1),cursor+pd.Timedelta(days=days));cursor+=pd.Timedelta(days=days)
    supplied=set(dates);result={}
    for name,(left,right) in bounds.items():
        wanted=[d.date().isoformat() for d in pd.date_range(left,right,freq='D')]
        if not set(wanted).issubset(supplied):raise ValueError('Public TRAIN calendar incomplete for '+name)
        result[name]=wanted
    flat=[d for name in PHASES for d in result[name]]
    if flat!=sorted(set(flat)):raise ValueError('Queue phases overlap or are out of order')
    return result


def validate_phase_origins(part,origins,dates):
    if part not in PHASES:raise ValueError('Unknown Queue phase')
    if not origins or len(set(origins))!=len(origins):raise ValueError('Unique nonempty event origins required')
    if any(pd.Timestamp(str(origin)).date().isoformat() not in dates[part] for origin in origins):
        raise ValueError('Event origin outside registered phase')
