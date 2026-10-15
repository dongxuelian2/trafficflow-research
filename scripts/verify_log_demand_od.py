"""Independent trust-region oracle and latent-group rank audit,not hidden truth."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from trafficflow.data import Release
from trafficflow.odme import operator
from trafficflow.log_demand_od import log_demand_inverse,group_design,demand_objective
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from zero_aware_od_experiment import synthetic
from log_demand_od_experiment import GRID


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--settings-index',type=int,nargs='+',default=[1,3]);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite independent numerical proof')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Log_demand_OD_independent_trust_region_and_group_rank_audit',
        'panels':['D12_I5_N','D7_I210_W'],'regimes':['hierarchical_demand','group_biased_prior','count_outliers'],
        'seed':1201,'settings_indices':args.settings_index,'oracle':'Bounded scipy least_squares,jac3-point,Gaussian group penalties retained under count Huber',
        'scope':'Development numeric and structural diagnostic only;no model selection/hidden path truth',
        'formal_ODME':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');rows=[];rank=[]
    with threadpool_limits(limits=4):
        for panel in ['D12_I5_N','D7_I210_W']:
            A,c0,b0,paths=operator(release,panel,'train');H,columns=group_design(paths)
            J=(A.multiply(np.maximum(b0,2.))@H).toarray()
            singular=np.linalg.svd(J,compute_uv=False);tol=max(J.shape)*np.finfo(float).eps*max(singular[0],1.)
            actual=int((singular>tol).sum())
            rank.append({'panel':panel,'n_paths':A.shape[1],'n_counted_links':A.shape[0],'n_group_parameters':H.shape[1],
                'local_group_design_rank':actual,'group_nullity':H.shape[1]-actual,
                'scope':'At weakprior log-effects zero;regularization can select coordinates but does not identify them from data'})
            for regime in ['hierarchical_demand','group_biased_prior','count_outliers']:
                truth,c,b=synthetic(A,b0,paths,1201,regime)
                for index in args.settings_index:
                    config=GRID[index];f,info=log_demand_inverse(A,c,b,paths,return_effects=True,**config)
                    reference=np.maximum(b,config['prior_floor']);sd=config['noise_cv']*np.maximum(c,10.)
                    def residual(z):return np.r_[(np.asarray(A@(reference*np.exp(H@z)))-c)/sd,z/config['group_sd']]
                    delta=config['huber_delta'];m=len(c)
                    def loss(z2):
                        rho=np.array([z2.copy(),np.ones(len(z2)),np.zeros(len(z2))])
                        if delta is not None:
                            outside=(np.arange(len(z2))<m)&(z2>delta**2);root=np.sqrt(z2[outside])
                            rho[0,outside]=2*delta*root-delta**2;rho[1,outside]=delta/root;rho[2,outside]=-delta/(2*root**3)
                        return rho
                    oracle=least_squares(residual,np.zeros(H.shape[1]),jac='3-point',bounds=(-3.,3.),loss=loss,
                        ftol=1e-11,xtol=1e-11,gtol=1e-10,max_nfev=400)
                    other=reference*np.exp(H@oracle.x)
                    objective,gradient,_=demand_objective(np.array(info['log_effects']),A,c,reference,H,config['group_sd'],config['noise_cv'],delta)
                    max_relative=float(np.max(np.abs(f-other)/np.maximum(1.,np.abs(other))))
                    rows.append({'panel':panel,'regime':regime,'setting_index':index,'solver_local_objective':objective,
                        'oracle_local_objective':float(oracle.cost),'oracle_reported_success':bool(oracle.success),
                        'objective_difference':float(objective-oracle.cost),'max_relative_flow_difference':max_relative,
                        'independent_solutions_agree':bool(oracle.success and np.isclose(objective,oracle.cost,rtol=1e-7,atol=1e-7) and max_relative<=2e-4),
                        'scope':'Two local optimizers agreeing is numerical evidence,not global optimum/hidden recovery'})
                    np.savez_compressed(folder/(panel+'_'+regime+'_'+str(index)+'.npz'),solver=f,oracle=other,
                        solver_log_effects=np.array(info['log_effects']),oracle_log_effects=oracle.x)
            print(panel,'independent oracle complete',flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'rows':rows,'group_rank':rank,
        'all_independent_solutions_agree':all(r['independent_solutions_agree'] for r in rows),
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_ODME':None,'formal_Total':None,
        'limitations':'Local rank and local optimizer agreement only;positive factor model excludes true zero support and all path-level residuals.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('agreements',sum(r['independent_solutions_agree'] for r in rows),'of',len(rows),'rank',rank,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
