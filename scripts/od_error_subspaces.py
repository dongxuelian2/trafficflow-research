"""Known-truth L2 errors in counted row space and its orthogonal null space."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
from trafficflow.data import Release
from trafficflow.odme import operator
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from od_shift_generators import NEW_REGIMES
from zero_aware_od_experiment import NINE


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite subspace diagnostics')
    study=json.loads(args.study.read_text())
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'OD_known_truth_rowspace_nullspace_L2_error_decomposition','source_study_sha256':sha256_file(args.study),
        'operator':'Unweighted counted public A;full thin SVD with standard singular-value rank tolerance',
        'metric':'Squared Euclidean path error / squared true flow norm;orthogonal components add exactly',
        'scope':'Own simulated truth only. L2 diagnostic is not official L1 S_od or actual hidden recovery.',
        'formal_ODME':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');operators={};bases={};ranks=[]
    rows=[];max_orthogonality=0.;max_null_residual=0.
    with threadpool_limits(limits=4):
        for panel in release.panels():
            A,_,_,_=operator(release,panel,'train');_,singular,Vt=svd(A.toarray(),full_matrices=False,check_finite=True)
            tolerance=max(A.shape)*np.finfo(float).eps*singular[0];rank=int((singular>tolerance).sum())
            basis=Vt[:rank];operators[panel]=A;bases[panel]=basis
            orthogonal=float(np.max(np.abs(basis@basis.T-np.eye(rank))));max_orthogonality=max(max_orthogonality,orthogonal)
            if orthogonal>1e-10:raise ValueError('SVD rowspace basis is not orthonormal')
            path=folder/(panel+'_counted_rowspace.npz');np.savez_compressed(path,basis=basis,singular_values=singular)
            ranks.append({'panel':panel,'n_counted_links':A.shape[0],'n_paths':A.shape[1],'rank':rank,
                'nullity':A.shape[1]-rank,'rank_tolerance':float(tolerance),'basis_path':path.as_posix(),'basis_sha256':sha256_file(path)})
        for case in study['case_files']:
            if sha256_file(case['path'])!=case['sha256']:raise ValueError('Case changed')
            A,basis=operators[case['panel']],bases[case['panel']]
            with np.load(case['path']) as arrays:
                truth=arrays['truth'];denominator=max(float(truth@truth),1e-12)
                methods=[m for m in arrays.files if m not in ['truth','counts','weak_prior']]
                errors=np.column_stack([arrays[m]-truth for m in methods]);coordinates=basis@errors
                projected=basis.T@coordinates;null=errors-projected
                total=np.sum(errors**2,axis=0)/denominator;observable=np.sum(coordinates**2,axis=0)/denominator
                null_error=np.sum(null**2,axis=0)/denominator
                if not np.allclose(total,observable+null_error,rtol=1e-10,atol=1e-12):raise ValueError('L2 Pythagorean decomposition failed')
                residual=float(np.max(np.abs(A@null))/max(1.,float(np.max(np.abs(A@errors)))))
                max_null_residual=max(max_null_residual,residual)
                if residual>1e-9:raise ValueError('Null component remains observable in counts')
                for index,method in enumerate(methods):
                    rows.append({'phase':case['phase'],'panel':case['panel'],'seed':case['seed'],'regime':case['regime'],
                        'method':method,'normalized_squared_path_error':float(total[index]),
                        'normalized_squared_rowspace_error':float(observable[index]),
                        'normalized_squared_nullspace_error':float(null_error[index])})
    frame=pd.DataFrame(rows);frame['family']=frame.panel.str.rsplit('_',n=1).str[0];summary={}
    for population,regimes in [('same_laws',NINE),('new_shift_laws',NEW_REGIMES)]:
        data=frame[(frame.phase=='assessment')&frame.regime.isin(regimes)];summary[population]={}
        for method in sorted(data.method.unique()):
            sub=data[data.method==method]
            summary[population][method]={metric:float(sub.groupby(['family','seed','regime'])[metric].mean().groupby('family').mean().mean())
                for metric in ['normalized_squared_path_error','normalized_squared_rowspace_error','normalized_squared_nullspace_error']}
    result={'registered':registered,'source_snapshot':snapshot,'ranks':ranks,'rows':rows,'summary':summary,
        'n_unique_case_prediction_records':len(rows),'max_basis_orthonormality_residual':max_orthogonality,
        'max_relative_count_residual_of_null_component':max_null_residual,
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_ODME':None,'formal_Total':None,
        'limitations':'No path is declared individually identifiable from rank. Counts are noisy and priors choose nullspace directions. '
        'Squared L2 geometry does not decompose official L1 score gains. Deployment aliases are excluded. '
        'Failure cases have no prediction and remain failures in original study;subspace averages are successful-case diagnostics only.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'diagnostics.json').write_text(text,encoding='utf-8')
    print('records',len(rows),'null_count_residual',max_null_residual,'summary',summary,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
