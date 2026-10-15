"""Paired calendar-block uncertainty for public causal State controls."""
import numpy as np
from .joint_dynamics_eval import combine_state


def paired_state_bootstrap(cases,baseline,candidate,seed=20261015):
    dates=cases[0]['dates']
    if any(c['dates']!=dates or [r['date'] for r in c['records']]!=dates for c in cases):raise ValueError('Matched complete State calendars required')
    regimes=['R1','R2','R3'];counts=np.array([[[r['models'][baseline][g]['n'] for g in regimes] for r in c['records']] for c in cases],float)
    if any(r['models'][baseline][g]['n']!=r['models'][candidate][g]['n'] for c in cases for r in c['records'] for g in regimes):raise ValueError('Matched State target populations required')
    points={role:[combine_state([r['models'][role] for r in c['records']])['public_subset_regime_macro'] for c in cases] for role in [baseline,candidate]}
    if any(v is None for values in points.values() for v in values):return {'gain':None,'ci95':None,'scope':'A required public regime is absent;unknown,notzero.'}
    errors=np.array([[[[[r['models'][role][g][key] for g in regimes] for r in c['records']] for c in cases] for role in [baseline,candidate]] for key in ['speed_normalized_SSE','flow_normalized_SSE']],float)
    n=len(dates);rng=np.random.default_rng(seed);starts=rng.integers(0,n,(4000,int(np.ceil(n/7))));ix=((starts[:,:,None]+np.arange(7))%n).reshape(4000,-1)[:,:n];count=counts[:,ix].sum(axis=2);valid=(count>0).all(axis=(0,2));e=errors[:,:,:,ix].sum(axis=4)
    with np.errstate(invalid='ignore',divide='ignore'):score=.54*np.maximum(0.,1-np.sqrt(e[0]/count[None]))+.46*np.maximum(0.,1-np.sqrt(e[1]/count[None]))
    values=score.mean(axis=(1,3))[:,valid];delta=values[1]-values[0];point={k:float(np.mean(v)) for k,v in points.items()}
    return {'baseline':baseline,'candidate':candidate,'point_macro':point,'gain':float(np.mean(points[candidate])-np.mean(points[baseline])),'case_gains':[float(b-a) for a,b in zip(points[baseline],points[candidate])],
        'ci95':np.quantile(delta,[.025,.975]).tolist() if len(delta) else None,'probability_positive':float((delta>0).mean()) if len(delta) else None,'block_days':7,'resamples':4000,'valid_resamples':int(valid.sum()),'seed':seed,'n_cases':len(cases),'n_days':n,
        'scope':'Conditional fitted-model publiccausalState subset,matched targets,calendar draws synchronized across panels,perregimeformula/equalcase mean. Not optimizerseed uncertainty,latent/officialState/Physics/Total or a blind final test.'}
