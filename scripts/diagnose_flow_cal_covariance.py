"""Cal-only scalar flow-error complementarity;no assessment/forward reads."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot
from flow_transfer_common import context


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite cal diagnostic')
    study,cfg,source,views,original=context(args.study);cache=Path(original['registered']['protocol']['cache'])/'selection'
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Cal_only_flow_error_covariance_after_fixed_transfer_rejection','source_study_sha256':sha256_file(args.study),
        'data_scope':'Only selection NPZ and original selection teacher,never assessment/forward',
        'fit':'One bounded scalar alpha in[0,1],minimizing exact equal-regime cal flow RMSE;NN speed fixed',
        'gate':'Require cal State gain>=.0005 and no regime loss>.001 before considering another frozen forward policy',
        'interpretation':'In-sample development diagnosis,not predictive validation or independent gain',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    terms={r:np.zeros(4,float) for r in ['R1','R2','R3']};files=[]
    for record in study['prediction_files']:
        if record['part']!='selection':continue
        path=cache/(record['date']+'.npz')
        if sha256_file(path)!=study['cache_sha256'][path.as_posix()] or sha256_file(record['path'])!=record['sha256']:raise ValueError('Cal sources changed')
        with np.load(path) as d, np.load(record['path']) as p:
            ratio=d['target_link_lanes']/d['station_lanes'];error=(p['nn_primary'][:,1]-d['station_truth'][:,1])*ratio
            delta=(p['flow_seed_mean'][:,1]-p['nn_primary'][:,1])*ratio
            terms[str(d['regime'])]+=np.array([len(error),float(error@error),float(error@delta),float(delta@delta)])
        files.append({'date':record['date'],'teacher_path':path.as_posix(),'teacher_sha256':sha256_file(path),'predictions_sha256':record['sha256']})
    def rmses(alpha):return np.array([np.sqrt(max(0,(c+2*alpha*b+alpha*alpha*a)/n)) for n,c,b,a in terms.values()])
    fit=minimize_scalar(lambda a:float(rmses(a).mean()),bounds=(0,1),method='bounded',options={'xatol':1e-12})
    candidates=[(0.,float(rmses(0).mean())),(1.,float(rmses(1).mean())),(float(fit.x),float(fit.fun))];alpha=min(candidates,key=lambda r:(r[1],r[0]))[0]
    gains=.46/600*(rmses(0)-rmses(alpha));gain=float(gains.mean());minimum=float(gains.min());eligible=gain>=.0005 and minimum>=-.001
    result={'registered':registered,'source_snapshot':snapshot,'cal_sources':files,'n_cal_days':len(files),'terms':{k:v.tolist() for k,v in terms.items()},
        'optimal_bounded_cal_flow_tree_alpha':alpha,'in_sample_cal_State_gain':gain,'minimum_regime_cal_State_gain':minimum,'passes_predefined_cal_gate':eligible,
        'cal_rmse_NN':rmses(0).tolist(),'cal_rmse_tree':rmses(1).tolist(),'cal_rmse_optimal_mix':rmses(alpha).tolist(),
        'new_model_exported':False,'new_future_results_read':False,'seconds':time.perf_counter()-start,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'scope':'Cal-only sensitivity after fixed transfer failure;no validation claim.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'diagnostic.json').write_text(text,encoding='utf-8')
    print('flow_cal_covariance','alpha',alpha,'cal_gain',gain,'eligible',eligible,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
