"""Export frozen policies and replay all cal/assessment roles from public input."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import Release,VALUES
from trafficflow.state import StateModel,StateFeatures,neighbors
from trafficflow.state_ensemble import StateEnsemble
from trafficflow.recurrent_state import WindowedStateModel
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.experiments import partition_date,day_template
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from train_graph_experiment import stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite recurrent deployment verification')
    study=json.loads(args.study.read_text());config=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem
    selection=json.loads((source/'selection_sealed.json').read_text())
    if selection!=study['selection'] or sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256']:
        raise ValueError('Model choice changed')
    if study['phase_read_order']!=[{'part':'fit','prior_selection_seal_sha256':None},
        {'part':'selection','prior_selection_seal_sha256':None},
        {'part':'assessment','prior_selection_seal_sha256':study['selection_seal_sha256']}]:raise ValueError('Label read order differs')
    for field in ['baseline','incumbent']:
        if sha256_file(config[field+'_file'])!=config[field+'_sha256']:raise ValueError('Strong reference changed')
    for name,digest in study['model_sha256'].items():
        if sha256_file(source/name)!=digest:raise ValueError('Checkpoint changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Recurrent_State_complete_role_deployment_replay','source_study_sha256':sha256_file(args.study),
        'source_seal_sha256':study['selection_seal_sha256'],'parts':['selection','assessment'],
        'scope':'All9frozen predictor roles and primary export;inputs rebuilt from masked Parquet,raw labels remain evaluator only',
        'prediction_tolerance_absolute_lane_units':1e-8,'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    release=Release('data/kaggle_public');panel=config['panel'];net,topology=release.network(panel),release.topology(panel)
    baseline=StateModel.load(Path(config['baseline_file']));incumbent=StateModel.load(Path(config['incumbent_file']));cache=Path(config['cache'])
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train')
    roles={name:StateModel.load(source/('State_'+name+'.pkl')) for name in selection['choices']}
    for name in roles:
        if sha256_file(source/('State_'+name+'.pkl'))!=study['exported_model_sha256']['State_'+name+'.pkl']:
            raise ValueError('Exported per-seed model changed')
    roles['incumbent']=incumbent
    first=StateModel.load(source/('State_'+next(iter(selection['choices']))+'.pkl'));first.alpha=0.;roles['lgb']=first
    roles['incumbent_windowed']=WindowedStateModel(baseline,incumbent.state,topology,incumbent.alpha,'tcn',
        config['hidden'],config['context'],config['core'])
    for kind in ['tcn','tcn_gru']:
        members=[StateModel.load(source/(f'State_seed{s}_{kind}.pkl')) for s in config['seeds']]
        roles[kind+'_seed_mean']=StateEnsemble(members,np.ones(len(members)))
    exports={}
    for name,model in roles.items():
        path=folder/('State_'+name+'.pkl');model.save(path);exports[name]={'path':path.as_posix(),'sha256':sha256_file(path)}
    primary=folder/'State_primary.pkl';roles[selection['primary']].save(primary)
    roles['primary_export']=StateModel.load(primary);exports['primary_export']={'path':primary.as_posix(),'sha256':sha256_file(primary)}
    rows=[];cache_hashes={};input_checks=[];maximum_difference=0.
    reconstructed=None
    for part in ['selection','assessment']:
        scores={name:[] for name in roles}
        for path in sorted((cache/part).glob('*.npz')):
            if sha256_file(path)!=study['cache_sha256'][path.as_posix()]:raise ValueError('Cached evaluator changed')
            day=dict(np.load(path));features=StateFeatures(release.read(files[path.stem]),net,topology,baseline.profile,version=baseline.feature_version)
            if not np.array_equal(reconstruction_input(features),day['x']):raise ValueError('Independent legal inputs differ')
            target=day_template(release,panel,'train',features.frame,template)
            if not np.array_equal(features.lookup(target),day['target_indices']):raise ValueError('Independent target order differs')
            if reconstructed is None:
                matrices=[]
                for direction in ['incoming','outgoing']:
                    a=np.zeros((len(features.links),len(features.links)),np.float32)
                    for j,connected in enumerate(neighbors(topology,features.links,direction)):
                        if connected:a[j,connected]=1/len(connected)
                    matrices.append(a)
                lanes=net.lanes.reindex(features.links).to_numpy(float)
                fd=np.stack([net.free_speed_kmh.reindex(features.links),net.capacity_vph.reindex(features.links)/lanes,
                    net.k_jam.reindex(features.links)/lanes]).astype(np.float32)
                reconstructed=dict(zip(['incoming','outgoing','fd'],matrices+[fd]))
                saved=dict(np.load(cache/'graph.npz'))
                if any(not np.array_equal(saved[k],v) for k,v in reconstructed.items()):raise ValueError('Public graph reconstruction differs')
            lane=target.link_id.map(net.lanes).to_numpy(float)
            with np.load(source/'predictions'/part/(path.stem+'.npz')) as reference:
                for role,model in roles.items():
                    prediction=model.predict(features,target)[VALUES].to_numpy(float);prediction[:,1]/=lane
                    wanted=reference[selection['primary'] if role=='primary_export' else role]
                    difference=float(np.abs(prediction-wanted).max());maximum_difference=max(maximum_difference,difference)
                    if difference>1e-8:raise ValueError('Frozen predictor replay differs: '+role)
                    scores[role].append(stats(day,prediction))
                    rows.append({'part':part,'date':path.stem,'role':role,'maximum_absolute_difference_lane_units':difference})
            input_checks.append({'part':part,'date':path.stem,'legal_inputs_and_target_order_exact':True})
        for role,values in scores.items():
            measured=state_from_sufficient_stats(values)
            expected=(selection['cal_metrics'] if part=='selection' else study['assessment'])[selection['primary'] if role=='primary_export' else role]
            if abs(measured['S_state']-expected['S_state'])>1e-10:raise ValueError('Independent role metric differs')
    for path in [cache/'prepared.json',cache/'graph.npz',cache/'lightgbm.pkl']:
        cache_hashes[path.as_posix()]=sha256_file(path)
    result={'registered':registered,'source_snapshot':snapshot,'exports':exports,'primary_role':selection['primary'],
        'rows':rows,'input_checks':input_checks,'all_legal_inputs_exact':True,'all_graph_arrays_exact':True,
        'all_predictions_within_absolute_tolerance':True,'maximum_absolute_prediction_difference_lane_units':maximum_difference,
        'n_role_day_replays':len(rows),'n_unique_days_reconstructed':len(input_checks),'cache_aux_sha256':cache_hashes,
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limitations':'Complete saved-policy inference verification,not independent retraining/hidden accuracy. '
        'Equal-member arithmetic in total-flow versus lane-flow units can differ at floating roundoff.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'verification.json').write_text(text,encoding='utf-8')
    print('roles',len(roles),'role_days',len(rows),'max_diff_lane_units',maximum_difference,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
