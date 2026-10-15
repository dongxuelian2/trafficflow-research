"""Independent OD stress laws; no released/reference path truth is accessed."""
import numpy as np


NEW_REGIMES=['distance_gravity','gamma_poisson_counts','correlated_count_bias',
    'sparse_zone_shutdown','heavy_tail_combined','dense_weak_prior_independent_truth']


def shifted_synthetic(A,public_prior,paths,seed,regime):
    if regime not in NEW_REGIMES:raise ValueError(regime)
    rng=np.random.default_rng(seed+2000+100*NEW_REGIMES.index(regime))
    b0=np.asarray(public_prior,float);n=len(b0);scale=max(float(np.maximum(b0,2.).mean()),10.)
    origins=paths.origin_zone.astype(str).to_numpy();destinations=paths.destination_zone.astype(str).to_numpy()
    demand=rng.gamma(1.7,scale/1.7,n);truth=demand.copy()
    if regime=='distance_gravity':
        mass=np.ones(n)
        for labels in [origins,destinations]:
            factors={label:rng.gamma(2.,.5) for label in sorted(set(labels))}
            mass*=np.array([factors[label] for label in labels])
        # Public link count along each path is a synthetic distance proxy only.
        length=paths.link_seq.astype(str).str.count(';').to_numpy()+1
        truth=mass/(.5+length/np.median(length))**1.1
        truth*=n*scale/truth.sum()
        prior=np.maximum(0,truth+rng.normal(0,.5*scale,n))
    elif regime=='gamma_poisson_counts':
        prior=truth*rng.lognormal(-.5*.5**2,.5,n)
    elif regime=='correlated_count_bias':
        prior=truth*rng.lognormal(-.5*.3**2,.3,n)
    elif regime=='sparse_zone_shutdown':
        labels=sorted(set(origins));closed=set(rng.choice(labels,max(1,len(labels)//4),replace=False))
        truth[np.isin(origins,list(closed))]=0.
        prior=demand*rng.lognormal(-.5*.2**2,.2,n)
    elif regime=='heavy_tail_combined':
        prior=truth*np.exp(np.clip(.5*rng.standard_t(3,n),-3.,3.))
        prior[rng.random(n)<.2]=0.
    else:
        # Prior and truth share only an overall scale,not path-level signals.
        prior=rng.gamma(2.,scale/2.,n)
    clean=np.asarray(A@truth)
    if regime=='gamma_poisson_counts':counts=rng.poisson(clean).astype(float)
    elif regime=='correlated_count_bias':
        group=np.arange(len(clean))%4;bias=rng.normal(0,.06,4)
        counts=np.maximum(0,clean*(1+bias[group])+rng.normal(0,.005*np.maximum(clean,10.)))
    elif regime=='heavy_tail_combined':counts=np.maximum(0,clean+.05*np.maximum(clean,10.)*rng.standard_t(4,len(clean)))
    else:counts=np.maximum(0,clean+rng.normal(0,.02*np.maximum(clean,10.)))
    if not truth.sum()>0:raise ValueError('Synthetic demand must have positive total')
    return truth,counts,prior
