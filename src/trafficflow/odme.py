"""Sparse OD inverse solvers; never use locally solved flows as ground truth."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize, lsq_linear
from .data import Release,csv,OD_KEY,check_keys
from .metrics import od_link_score

def operator(release: Release,panel: str,split: str):
    folder=release.root/'corridors'/panel/'network'
    paths=csv(folder/'path_set.csv')
    check_keys(paths,['path_id'],'path set')
    incidence=csv(folder/'path_link_incidence.csv').drop_duplicates(['link_id','path_id'])
    counts=csv(release.root/'task4'/panel/split/'synthetic_link_counts.csv')
    check_keys(counts,['link_id'],'OD counts')
    if not set(incidence.path_id).issubset(paths.path_id):
        raise ValueError('Unknown path in incidence')
    missing=set(counts.link_id)-set(incidence.link_id)
    if missing:
        raise ValueError(f'Counted links absent from incidence: {sorted(missing)[:5]}')
    # Unobserved connectors contribute no fabricated zero measurements.
    path_index={p:i for i,p in enumerate(paths.path_id)}
    link_index={l:i for i,l in enumerate(counts.link_id)}
    inc=incidence[incidence.link_id.isin(link_index)]
    A=sparse.csr_matrix((np.ones(len(inc)),(inc.link_id.map(link_index),inc.path_id.map(path_index))),shape=(len(counts),len(paths)))
    c=pd.to_numeric(counts['count'],errors='raise').to_numpy(float)
    prior=csv(release.root/'task4'/panel/split/'synthetic_weak_prior.csv')
    check_keys(prior,['path_id'],'OD prior')
    if set(prior.path_id)!=set(paths.path_id):
        raise ValueError('Weak prior must cover released paths exactly')
    b=pd.to_numeric(prior.set_index('path_id').reindex(paths.path_id).path_flow,errors='raise').to_numpy(float)
    if not np.isfinite(c).all() or not np.isfinite(b).all() or (c<0).any() or (b<0).any():
        raise ValueError('Invalid OD counts or prior')
    return A,c,b,paths

def solve(A,c,b,method='ridge',regularization=.05,maxiter=1000):
    if regularization<0:
        raise ValueError('Negative OD regularization')
    if method=='prior':
        return b.copy(),{'success':True,'iterations':0}
    if method in ['ridge','relative']:
        weight=np.ones_like(b) if method=='ridge' else 1/np.maximum(b,10.)
        aa=sparse.vstack([A,np.sqrt(regularization)*sparse.diags(weight)],format='csr')
        yy=np.r_[c,np.sqrt(regularization)*b*weight]
        result=lsq_linear(aa,yy,bounds=(0,np.inf),tol=1e-8,lsmr_tol=1e-9,max_iter=maxiter)
        f=result.x
    elif method=='poisson':
        weights=1/np.maximum(c,1.)
        scale=np.maximum(b,10.)
        def objective(z):
            f=z*scale
            residual=A@f-c
            baseline=np.maximum(b,1.)
            ratio=(f+1e-6)/baseline
            penalty=regularization*np.sum((f+1e-6)*np.log(ratio)-f+baseline)
            grad=regularization*np.log(ratio)
            return .5*np.dot(weights*residual,residual)+penalty,(np.asarray(A.T@(weights*residual))+grad)*scale
        result=minimize(objective,np.maximum(b,1e-6)/scale,jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(b),
                        options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-6,'maxls':40})
        f=result.x*scale
    else:
        raise ValueError('Unknown OD method')
    if not result.success:
        raise RuntimeError(f'OD optimizer failed: {result.message}')
    return np.maximum(f,0),{'success':bool(result.success),'iterations':int(result.nit),
                          'S_link':od_link_score(A,f,c),'prior_L1_change':float(np.abs(f-b).sum())}

def predict(release,panel,split,method='ridge',regularization=.05):
    A,c,b,paths=operator(release,panel,split)
    f,info=solve(A,c,b,method,regularization)
    template=release.template(4,panel,split)
    ids=paths.set_index('path_id')
    if set(template.path_id)!=set(paths.path_id):
        raise ValueError('OD template path mismatch')
    for zone in ['origin_zone','destination_zone']:
        if not np.array_equal(template[zone].to_numpy(),template.path_id.map(ids[zone]).to_numpy()):
            raise ValueError('OD template zone mismatch')
    result=template[OD_KEY].copy()
    result['path_flow']=result.path_id.map(dict(zip(paths.path_id,f)))
    info.update({'S_ODME':None,'S_od':None,'S_dev':None,'S_attr':None,'S_link':od_link_score(A,f,c),
                 'matrix_shape':list(A.shape),'matrix_nnz':A.nnz})
    return result,info

def count_holdout(A,c,b,regularization=.05,seed=2026):
    rng=np.random.default_rng(seed)
    order=rng.permutation(A.shape[0]);cut=max(1,int(.8*len(order)))
    if cut==len(order):
        return {'heldout_S_link':None}
    f,info=solve(A[order[:cut]],c[order[:cut]],b,'ridge',regularization)
    return {'heldout_S_link':od_link_score(A[order[cut:]],f,c[order[cut:]]),'fit':info,
            'note':'Sensor holdout diagnostic; not hidden OD recovery accuracy.'}
