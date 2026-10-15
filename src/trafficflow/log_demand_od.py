"""Positive origin/destination demand effects,not identifiable path truth.

f_p=max(b_p,floor)*exp(u_origin[p]+v_destination[p]). Gaussian log effects
regularize a relative-Gaussian or Huber count likelihood. The objective is
nonlinear; stationarity certifies a local constrained solution only.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve


def group_design(paths):
    n=len(paths);columns=[];offset=0;codes=[]
    for key in ['origin_zone','destination_zone']:
        values=paths[key].astype(str).to_numpy();labels=sorted(set(values))
        index={label:i+offset for i,label in enumerate(labels)}
        codes.append(np.array([index[label] for label in values]));columns.extend([(key,label) for label in labels]);offset+=len(labels)
    H=sparse.csr_matrix((np.ones(2*n),(np.tile(np.arange(n),2),np.concatenate(codes))),shape=(n,offset))
    return H,columns


def demand_objective(z,A,c,b,H,group_sd,noise_cv,huber_delta=None):
    f=b*np.exp(np.asarray(H@z));count_sd=noise_cv*np.maximum(c,10.)
    r=(np.asarray(A@f)-c)/count_sd
    if huber_delta is None:
        loss=.5*r*r;derivative=r
    else:
        a=np.abs(r);loss=np.where(a<=huber_delta,.5*r*r,huber_delta*(a-.5*huber_delta))
        derivative=np.clip(r,-huber_delta,huber_delta)
    gradient=np.asarray(H.T@(f*np.asarray(A.T@(derivative/count_sd))))+z/group_sd**2
    return float(loss.sum()+.5*np.dot(z,z)/group_sd**2),gradient,f


def projected_gradient(z,gradient,bound):
    projected=gradient.copy()
    projected[(z<=-bound+1e-8)&(gradient>0)]=0.
    projected[(z>=bound-1e-8)&(gradient<0)]=0.
    return projected


def polish_local_solution(z,A,c,b,H,group_sd,noise_cv,huber_delta,bound):
    """Exact-Hessian descent or roundoff-budget residual reduction,unchanged KKT."""
    z=z.copy();sd=noise_cv*np.maximum(c,10.);accepted=0
    for iteration in range(20):
        value,gradient,f=demand_objective(z,A,c,b,H,group_sd,noise_cv,huber_delta)
        projected=projected_gradient(z,gradient,bound);certificate=float(np.abs(projected).max())
        if certificate<=1e-7:break
        r=(np.asarray(A@f)-c)/sd
        derivative=r if huber_delta is None else np.clip(r,-huber_delta,huber_delta)
        curvature=np.ones(len(c)) if huber_delta is None else (np.abs(r)<huber_delta).astype(float)
        J=(A.multiply(f)@H).toarray()/sd[:,None]
        second=f*np.asarray(A.T@(derivative/sd))
        hessian=J.T@(curvature[:,None]*J)+(H.T@H.multiply(second[:,None])).toarray()
        hessian[np.diag_indices_from(hessian)]+=1/group_sd**2
        active=((z<=-bound+1e-8)&(gradient>0))|((z>=bound-1e-8)&(gradient<0))
        free=np.flatnonzero(~active)
        if not len(free):break
        small=hessian[np.ix_(free,free)];shift=0.;step=np.zeros_like(z)
        for retry in range(12):
            try:
                matrix=small.copy();matrix[np.diag_indices_from(matrix)]+=shift
                step[free]=-cho_solve(cho_factor(matrix,check_finite=False),gradient[free],check_finite=False);break
            except np.linalg.LinAlgError:shift=1e-8 if shift==0 else shift*10
        else:break
        length=1.;updated=False
        for backtrack in range(40):
            trial=np.clip(z+length*step,-bound,bound);direction=float(np.dot(gradient,trial-z))
            if direction>=0:break
            target,tg,_=demand_objective(trial,A,c,b,H,group_sd,noise_cv,huber_delta)
            next_certificate=float(np.abs(projected_gradient(trial,tg,bound)).max())
            budget=64*np.finfo(float).eps*(abs(value)+abs(target)+1.)
            if target<=value+1e-4*direction or (target<=value+budget and next_certificate<.5*certificate):
                z=trial;accepted+=1;updated=True;break
            length*=.5
        if not updated:break
    return z,accepted


def log_demand_inverse(A,counts,prior,paths,group_sd=.6,noise_cv=.02,prior_floor=2.,
                       huber_delta=None,max_log_effect=3.,maxiter=1000,return_effects=False):
    A=sparse.csr_matrix(A,dtype=float);c,b=np.asarray(counts,float),np.asarray(prior,float)
    if c.ndim!=1 or b.ndim!=1 or A.shape!=(len(c),len(b)) or len(paths)!=len(b) or min(A.shape)<1:
        raise ValueError('Matching nonempty incidence,counts,prior and path metadata required')
    if not all(np.isfinite(a).all() for a in [A.data,c,b]) or any((a<0).any() for a in [A.data,c,b]):
        raise ValueError('Finite nonnegative observations and incidence required')
    if not np.isfinite([group_sd,noise_cv,prior_floor,max_log_effect]).all() or min(group_sd,noise_cv,prior_floor,max_log_effect)<=0 or maxiter<1:
        raise ValueError('Positive finite scales,bounds and iteration budget required')
    if huber_delta is not None and (not np.isfinite(huber_delta) or huber_delta<=0):raise ValueError('Invalid Huber threshold')
    H,columns=group_design(paths);reference=np.maximum(b,prior_floor)
    total_ratio=np.median(np.maximum(c,1.)/np.maximum(np.asarray(A@reference),1.))
    starts=[np.zeros(H.shape[1]),np.full(H.shape[1],np.clip(.5*np.log(total_ratio),-max_log_effect,max_log_effect))]
    records=[];solutions=[]
    for initial in starts:
        def objective(z):
            value,gradient,_=demand_objective(z,A,c,reference,H,group_sd,noise_cv,huber_delta)
            return value,gradient
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(-max_log_effect,max_log_effect)]*len(initial),
            options={'maxiter':maxiter,'ftol':1e-14,'gtol':1e-7,'maxls':50,'maxcor':30})
        z,polish_steps=polish_local_solution(result.x,A,c,reference,H,group_sd,noise_cv,huber_delta,max_log_effect)
        value,gradient,f=demand_objective(z,A,c,reference,H,group_sd,noise_cv,huber_delta)
        projected=projected_gradient(z,gradient,max_log_effect)
        certificate=float(np.max(np.abs(projected)))
        valid=np.isfinite(f).all() and certificate<=2e-4
        records.append({'optimizer_reported_success':bool(result.success),'message':str(result.message),
            'iterations':int(result.nit),'Newton_polish_steps':polish_steps,'objective':value,'projected_gradient_max_abs':certificate,'certificate_passed':bool(valid)})
        if valid:solutions.append((value,f,z))
    if not solutions:raise RuntimeError('Log-demand local stationarity failed: '+str([r['projected_gradient_max_abs'] for r in records]))
    value,f,z=min(solutions,key=lambda row:row[0])
    info={'success':True,'solver':'origin_destination_log_demand_two_start_LBFGSB',
        'local_objective':value,'start_records':records,'n_group_parameters':H.shape[1],'n_paths':A.shape[1],
        'log_effect_rms':float(np.sqrt(np.mean(z*z))),'max_abs_log_effect':float(np.abs(z).max()),
        'settings':{'group_sd':group_sd,'noise_cv':noise_cv,'prior_floor':prior_floor,'huber_delta':huber_delta},
        'optimization_scope':'Locally stationary bounded MAP,not global optimum or path identifiability',
        'uncertainty':'Point estimate only; no calibrated posterior interval'}
    if return_effects:info['log_effects']=z.tolist()
    return f,info
