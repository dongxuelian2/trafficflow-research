"""Reconstruct all weighted tree rows and replay saved tabular policies."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,VALUES
from trafficflow.state import StateFeatures,StateModel
from trafficflow.native_remask import native_remask
from trafficflow.experiments import partition_date,day_template
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from trafficflow.runtime import sha256_file,capture_source_snapshot
from remask_tree_roles import context,load_tree_roles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite tree verification')
    study,cfg,source,views,original=context(args.study);old=original['registered']['protocol'];cache=Path(old['cache'])
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Independent_remask_tree_feature_label_weight_and_role_replay','source_study_sha256':sha256_file(args.study),
        'scope':'All920date/view table groups reconstructed;all6tabularroles x28cal/assessmentdays;4direct selected-policy calls',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    release=Release('data/kaggle_public');panel=cfg['panel'];net,topology=release.network(panel),release.topology(panel)
    baseline=StateModel.load(Path(old['baseline_file']));regime_days={r:sum(v['regime']==r for v in views['rows']) for r in ['R1','R2','R3']}
    original_count=sum(v['original_target_grid_cells'] for v in views['rows']);group_checks=[];maximum_weight=0.
    for variant,record in study['datasets'].items():
        if sha256_file(record['features_path'])!=record['features_sha256'] or sha256_file(record['arrays_path'])!=record['arrays_sha256']:
            raise ValueError('Training table changed')
        x=pd.read_parquet(record['features_path']);arrays=dict(np.load(record['arrays_path']))
        if len(x)!=record['rows'] or x.columns.tolist()!=baseline.models[0].feature_name_:raise ValueError('Training table columns/count differ')
        for group in record['groups']:
            day_index=next(i for i,row in enumerate(views['rows']) if row['date']==group['date']);row=views['rows'][day_index];view=group['view']
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Native view pool changed')
            raw=release.read(row['raw_file']);masked=release.read(row['masked_file'])
            if 'mask_regime' not in raw:raw['mask_regime']=row['regime']
            if view<0:
                input_frame=masked
                with np.load(row['original_cache_path']) as a:mask=a['mask'];teacher=a['y']
                mass=.5
            else:
                input_frame,_,_=native_remask(raw,panel,row['date'],views['registered']['protocol']['view_seeds'][view],row['regime'],row['public_origins'],variant=='queue')
                with np.load(row['path']) as a:mask=a['mask_'+variant][view];teacher=a['y']
                mass=.125
            f=StateFeatures(input_frame,net,topology,baseline.profile,baseline.feature_version);indices=np.flatnonzero(mask.ravel())
            first,last=group['start'],group['end'];expected_keys=np.c_[np.full(len(indices),day_index),np.full(len(indices),view),indices].astype(np.int32)
            if last-first!=len(indices) or not np.array_equal(arrays['row_keys'][first:last],expected_keys):raise ValueError('Training row identities differ')
            if not np.array_equal(x.iloc[first:last].to_numpy(),f.features.iloc[indices].to_numpy(),equal_nan=True):raise ValueError('Training input table differs')
            if not np.array_equal(arrays['y'][first:last],teacher.reshape(-1,2)[indices]*np.array([25.,600.])) or not np.array_equal(arrays['base'][first:last],f.base.reshape(-1,2)[indices]):
                raise ValueError('Training labels/interpolation base differ')
            expected=mass/(3*regime_days[row['regime']]*len(indices))*original_count
            error=float(np.abs(arrays['weight'][first:last]-expected).max());maximum_weight=max(maximum_weight,error)
            if error>1e-12:raise ValueError('Effective row weights differ')
            group_checks.append({'variant':variant,'date':group['date'],'view':view,'n_rows':len(indices),'features_labels_base_keys_exact':True})
        if not np.isclose(arrays['weight'].sum(),original_count,rtol=1e-12):raise ValueError('Effective total weight differs')
        del x,arrays
        print('tree_dataset_verified',variant,len([r for r in group_checks if r['variant']==variant]),'groups',flush=True)
    paths,models=load_tree_roles(args.study,folder/'exports');primary=models.pop('primary_export');models.pop('nn_primary');models.pop('old_lgb')
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train');metrics=[];calls=[];selected=[];maximum_prediction=0.
    nn=json.loads(Path('reports/research/A97_I210_W_native_remask_ablation.json').read_text());nn_folder=Path('artifacts/research/A97_I210_W_native_remask_ablation')
    for phase in ['selection','assessment']:
        dates=sorted((cache/phase).glob('*.npz'))
        for path in dates:
            frame=release.read(files[path.stem]);f=StateFeatures(frame,net,topology,baseline.profile,baseline.feature_version);target=day_template(release,panel,'train',frame,template)
            with np.load(path) as a:teacher=a['station_truth'];station=a['station_lanes'];lanes=a['target_link_lanes']
            with np.load(source/'predictions'/phase/(path.stem+'.npz')) as saved,np.load(nn_folder/'predictions'/phase/(path.stem+'.npz')) as old_predictions:
                if not np.array_equal(saved['nn_primary'],old_predictions[nn['selection']['primary']]):raise ValueError('Frozen NN anchor arrays differ')
                values={}
                for role,model in models.items():
                    p=model.predict(f,target)[VALUES].to_numpy(float);p[:,1]/=lanes;error=float(np.abs(p-saved[role]).max());maximum_prediction=max(maximum_prediction,error)
                    if error>1e-8:raise ValueError('Direct tabular predictor differs')
                    values[role]=p;calls.append({'phase':phase,'date':path.stem,'role':role,'maximum_lane_unit_error':error})
                for variant in ['point','queue']:
                    for alpha in [.25,.5,.75]:
                        key=f'{variant}_hybrid_{alpha}';p=alpha*values[variant+'_seed_mean']+(1-alpha)*saved['nn_primary']
                        if not np.array_equal(p,saved[key]):raise ValueError('Hybrid reconstruction differs')
                for role in saved.files:
                    p=saved[role];ev=p[:,0]-teacher[:,0];eq=(p[:,1]-teacher[:,1])*lanes/station
                    n=len(p);sv=float(ev@ev);sq=float(eq@eq);score=.54*max(0,1-(sv/n)**.5/25)+.46*max(0,1-(sq/n)**.5/600)
                    wanted=study['selection']['cal_metrics'][role] if phase=='selection' else study['assessment'][role]
                    row=next(v for v in wanted['days'] if v['date']==path.stem)
                    if abs(score-row['daily_State'])>1e-12:raise ValueError('Saved tree/hybrid metric differs')
                    metrics.append({'phase':phase,'date':path.stem,'role':role,'regime':str(frame.mask_regime.iloc[0]),'n':n,'speed_sse':sv,'lane_flow_sse':sq})
                if path in [dates[0],dates[-1]]:
                    p=primary.predict(f,target)[VALUES].to_numpy(float);p[:,1]/=lanes;error=float(np.abs(p-saved[study['selection']['primary']]).max());maximum_prediction=max(maximum_prediction,error)
                    if error>1e-8:raise ValueError('Selected serialized policy differs')
                    selected.append({'phase':phase,'date':path.stem,'maximum_lane_unit_error':error})
        print('tree_policy_verified',phase,'complete',flush=True)
    for phase in ['selection','assessment']:
        for role in study['assessment']:
            rows=[r for r in metrics if r['phase']==phase and r['role']==role];wanted=study['selection']['cal_metrics'][role] if phase=='selection' else study['assessment'][role]
            if abs(state_from_sufficient_stats(rows)['S_state']-wanted['S_state'])>1e-12:raise ValueError('Regime aggregate differs')
    result={'registered':registered,'source_snapshot':snapshot,'n_training_groups_rebuilt':len(group_checks),'dataset_checks':group_checks,
        'maximum_weight_error':maximum_weight,'n_direct_tree_role_day_calls':len(calls),'direct_tree_calls':calls,
        'n_direct_selected_policy_calls':len(selected),'selected_policy_calls':selected,'n_saved_role_day_metrics_verified':len(metrics),
        'maximum_prediction_error_lane_units':maximum_prediction,'seconds':time.perf_counter()-start,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'scope':'All training tables and saved model/metric interfaces checked;no independent retraining or new traffic scenes'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('remask_tree_verified','groups',len(group_checks),'calls',len(calls),'selected',len(selected),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
