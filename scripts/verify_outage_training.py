"""Reconstruct matched supervision and independently deploy every saved role."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import Release,VALUES
from trafficflow.state import StateFeatures,StateModel
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.experiments import partition_date,day_template,raw_for,target_truth
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot,sha256_file
from evaluate_outage_augmentation import policy_paths


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Use new training verification output')
    study=json.loads(args.study.read_text());config=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem
    original=json.loads(Path(config['source_study']).read_text());cache=Path(original['registered']['protocol']['cache'])
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256'] or json.loads((source/'selection_sealed.json').read_text())!=study['selection']:
        raise ValueError('Training choice changed')
    if study['phase_read_order']!=[{'part':'fit','prior_selection_seal_sha256':None},
        {'part':'selection','prior_selection_seal_sha256':None},{'part':'assessment','prior_selection_seal_sha256':study['selection_seal_sha256']}]:
        raise ValueError('Assessment read order differs')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Augmentation_matched_exposure_and_all_role_deployment_replay','source_study_sha256':sha256_file(args.study),
        'scope':'Independent original RNG/core exposure reconstruction;both controls tensor-exact with original checkpoints;all28cal/assessmentdays x10roles',
        'absolute_prediction_tolerance_lane_units':1e-8,'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    fit=[]
    for path in sorted((cache/'fit').glob('*.npz')):
        if sha256_file(path)!=study['cache_sha256'][path.as_posix()]:raise ValueError('Fit labels changed')
        with np.load(path) as a:fit.append({'date':path.stem,'mask':a['mask'],'regime':str(a['regime'])})
    groups=[[d for d in fit if d['regime']==r] for r in ['R1','R2','R3']];exposure_checks=[]
    for seed in config['seeds']:
        plan_path=Path(study['plans'][str(seed)]['path'])
        if sha256_file(plan_path)!=study['plans'][str(seed)]['sha256']:raise ValueError('Training plan changed')
        with np.load(plan_path) as a:plan=a['plan']
        patch_rng=np.random.default_rng(seed);aug_rng=np.random.default_rng(np.random.SeedSequence([seed,20261109]))
        expected={d['date']:np.zeros(d['mask'].shape,np.int32) for d in fit};patch_hash=hashlib.sha256()
        for step in range(24*120):
            for sample in range(4):
                group=int(patch_rng.integers(3));index=int(patch_rng.integers(len(groups[group])));d=groups[group][index]
                center=int(patch_rng.integers(len(d['mask'])));use=aug_rng.random()<.5;view=int(aug_rng.integers(4)) if use else -1
                if not np.array_equal(plan[step,sample],[group,index,center,view]):raise ValueError('Training RNG plan differs')
                first=min(max(center-32,0),len(d['mask'])-64);last=first+64;at=min(max(first-32,0),len(d['mask'])-128)
                expected[d['date']][first:last]+=d['mask'][first:last]
                patch_hash.update((d['date']+':'+str(at)+':'+str(first)+':'+str(last)+';').encode())
        for arm in ['control','augmented']:
            name=f'seed{seed}_{arm}';saved=source/(name+'_exposure.npz');row=study['training'][name]
            if sha256_file(saved)!=row['exposure_sha256'] or row['patch_sequence_sha256']!=patch_hash.hexdigest():raise ValueError('Exposure source changed')
            with np.load(saved) as a:
                if any(not np.array_equal(a[d],v) for d,v in expected.items()):raise ValueError('Actual exposure differs')
            if any(not (expected[d['date']][d['mask']]>0).all() for d in fit):raise ValueError('Unsupervised available fit cell')
        a,b=[study['training'][f'seed{seed}_{arm}'] for arm in ['control','augmented']]
        if a['initial_sha256']!=b['initial_sha256']:raise ValueError('Initialized weights not matched')
        if study['selection']['choices'][f'seed{seed}_control']['alpha']!=original['selection']['choices'][f'seed{seed}_tcn_gru']['alpha']:
            raise ValueError('Original control calibration policy differs')
        old_path=Path('artifacts/research')/Path(config['source_study']).stem/(f'seed{seed}_tcn_gru.pt')
        old=torch.load(old_path,map_location='cpu',weights_only=True);new=torch.load(source/(f'seed{seed}_control.pt'),map_location='cpu',weights_only=True)
        exact=old.keys()==new.keys() and all(torch.equal(old[k],new[k]) for k in old)
        if not exact:raise ValueError('Control checkpoint differs from original deterministic training')
        exposure_checks.append({'seed':seed,'both_actual_exposure_arrays_exact':True,'all_available_targets_supervised':True,
            'original_control_checkpoint_tensor_exact':exact,'original_control_cal_alpha_exact':True,
            'total_supervised_exposures':sum(int(v.sum()) for v in expected.values())})
    paths=policy_paths(args.study)
    for arm in ['control','augmented']:paths[arm+'_seed_mean']=source/('State_'+arm+'_seed_mean.pkl')
    models={n:StateModel.load(p) for n,p in paths.items()};primary=study['selection']['primary'];primary_path=folder/'State_primary.pkl'
    models[primary].save(primary_path);models['primary_export']=StateModel.load(primary_path)
    exports={n:{'path':p.as_posix(),'sha256':sha256_file(p)} for n,p in paths.items()}
    exports['primary_export']={'path':primary_path.as_posix(),'sha256':sha256_file(primary_path)}
    release=Release('data/kaggle_public');panel=config['panel'];net,topology=release.network(panel),release.topology(panel)
    profile=models['incumbent'].profile;version=models['incumbent'].feature_version
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train')
    checks=[];maximum=0.
    for phase in ['selection','assessment']:
        for path in sorted((cache/phase).glob('*.npz')):
            if sha256_file(path)!=study['cache_sha256'][path.as_posix()]:raise ValueError('Evaluator cache changed')
            frame=release.read(files[path.stem]);features=StateFeatures(frame,net,topology,profile,version=version)
            target=day_template(release,panel,'train',frame,template);lane=target.link_id.map(net.lanes).to_numpy(float)
            raw=next(raw_for(release,panel,[files[path.stem]]));truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float);truth[:,1]/=lane
            with np.load(path) as day:
                if not np.array_equal(day['x'],reconstruction_input(features)) or not np.array_equal(day['target_indices'],features.lookup(target)):
                    raise ValueError('Legal input/target index differs')
                if not np.array_equal(truth,day['station_truth']):raise ValueError('Raw teacher differs')
            with np.load(source/'predictions'/phase/(path.stem+'.npz')) as a:
                for role,model in models.items():
                    prediction=model.predict(features,target)[VALUES].to_numpy(float);prediction[:,1]/=lane
                    wanted=a[primary if role=='primary_export' else role];error=float(np.abs(prediction-wanted).max());maximum=max(maximum,error)
                    if error>1e-8:raise ValueError('Deployed role differs: '+role)
                    checks.append({'phase':phase,'date':path.stem,'role':role,'maximum_absolute_lane_unit_error':error})
        print('augmentation_training_replay',phase,'complete',flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'exposure_checks':exposure_checks,'exports':exports,
        'n_role_day_predictions_replayed':len(checks),'maximum_absolute_lane_unit_error':maximum,'prediction_checks':checks,
        'seconds':time.perf_counter()-start,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'Exact current-machine control replication and deployed-input verification;not independent retraining/generalization evidence'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('augmentation_training_verified',len(checks),'maximum',maximum,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
