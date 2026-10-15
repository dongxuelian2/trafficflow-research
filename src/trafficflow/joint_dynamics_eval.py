"""Public diagnostic sufficient statistics; never official scores or model inputs."""
import numpy as np


def state_sufficient(a,prediction):
    result={}
    for regime in ['R1','R2','R3']:
        mask=a['current_domain'].all(axis=1)&(a['current_regime']==regime);error=prediction.astype(float)-a['truth_current'].astype(float)
        result[regime]={'n':int(mask.sum()),'speed_normalized_SSE':float((error[:,0][mask]**2).sum()),'flow_normalized_SSE':float((error[:,1][mask]**2).sum())}
    return result


def combine_state(rows):
    details=[]
    for regime in ['R1','R2','R3']:
        n=sum(r[regime]['n'] for r in rows);v=sum(r[regime]['speed_normalized_SSE'] for r in rows);q=sum(r[regime]['flow_normalized_SSE'] for r in rows)
        rv=np.sqrt(v/n) if n else None;rq=np.sqrt(q/n) if n else None
        details.append({'regime':regime,'n':n,'rmse_speed_kmh':float(rv*25.) if n else None,'rmse_flow_per_lane':float(rq*600.) if n else None,'public_subset_state_formula':float(.54*max(0.,1-rv)+.46*max(0.,1-rq)) if n else None})
    return {'details':details,'all_regimes_present':all(r['n'] for r in details),'public_subset_regime_macro':float(np.mean([r['public_subset_state_formula'] for r in details])) if all(r['n'] for r in details) else None,
        'scope':'Eligible both-fields-missing currentT on public causal masked-history subset only;same point score formula,not official State or complete task template.'}


def forecast_sufficient(a,v,q):
    delta=a['truth_speed']-a['truth_current'][:,0,None];valid=a['available'];origin_valid=a['current_available'][:,None];masks={'all':valid,'drop20':valid&origin_valid&(delta<=-.8),'recovery20':valid&origin_valid&(delta>=.8),'queue_teacher':valid&(a['labels']>0)};rows={}
    for name,mask in masks.items():
        n=int(mask.sum());ev=(v.astype(float)-a['truth_speed'].astype(float))[mask];eq=(q.astype(float)-a['truth_flow'].astype(float))[mask]
        rows[name]={'n':n,'speed_normalized_SSE':float((ev**2).sum()),'flow_normalized_SSE':float((eq**2).sum())}
    return rows
