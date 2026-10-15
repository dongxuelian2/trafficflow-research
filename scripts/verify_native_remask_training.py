"""Actual new-label exposure reconstruction and full deployed-role replay."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import Release,VALUES
from trafficflow.state import StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.experiments import partition_date,day_template,target_truth
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot,sha256_file
from native_remask_roles import load_roles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite training replay')
    study=json.loads(args.study.read_text());config=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem
    original=json.loads(Path(config['source_study']).read_text());cache=Path(original['registered']['protocol']['cache'])
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256'] or json.loads((source/'selection_sealed.json').read_text())!=study['selection']:
        raise ValueError('Selection changed')
    if study['phase_read_order']!=[{'part':'fit','prior_selection_seal_sha256':None},{'part':'selection','prior_selection_seal_sha256':None},
        {'part':'assessment','prior_selection_seal_sha256':study['selection_seal_sha256']}]:raise ValueError('Label read order changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Native_remask_exposure_and_full_deployment_verification',
        'source_study_sha256':sha256_file(args.study),'scope':'Independent full RNG/core/loss-mask exposure;all28cal/assessmentdates x12roles',
        'controls':'Reused verified original checkpoints,not retrained by this study','formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    view_report=Path(config['views'])/'cache.json'
    if sha256_file(view_report)!=config['views_report_sha256']:raise ValueError('View report changed')
    views=json.loads(view_report.read_text());view_rows={r['date']:r for r in views['rows']};fit=[]
    for path in sorted((cache/'fit').glob('*.npz')):
        if sha256_file(path)!=study['cache_sha256'][path.as_posix()]:raise ValueError('Original labels changed')
        with np.load(path) as a:day={'date':path.stem,'mask':a['mask'],'regime':str(a['regime'])}
        row=view_rows[day['date']]
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Remask labels changed')
        with np.load(row['path']) as a:
            for variant in ['point','queue']:day[variant]=a['mask_'+variant]
        fit.append(day)
    groups=[[d for d in fit if d['regime']==r] for r in ['R1','R2','R3']];exposure_checks=[]
    for seed in config['seeds']:
        plan_path=Path(study['plans'][str(seed)]['path'])
        if sha256_file(plan_path)!=study['plans'][str(seed)]['sha256']:raise ValueError('Training plan changed')
        with np.load(plan_path) as a:plan=a['plan']
        patch_rng=np.random.default_rng(seed);aug_rng=np.random.default_rng(np.random.SeedSequence([seed,20261109]))
        counters={v:{d['date']:np.zeros(d['mask'].shape,np.int32) for d in fit} for v in ['point','queue']};patch=hashlib.sha256()
        for step in range(24*120):
            for sample in range(4):
                group=int(patch_rng.integers(3));index=int(patch_rng.integers(len(groups[group])));day=groups[group][index]
                center=int(patch_rng.integers(len(day['mask'])));view=int(aug_rng.integers(4)) if aug_rng.random()<.5 else -1
                if not np.array_equal(plan[step,sample],[group,index,center,view]):raise ValueError('Actual RNG plan differs')
                first=min(max(center-32,0),len(day['mask'])-64);last=first+64;left=min(max(first-32,0),len(day['mask'])-128)
                patch.update((day['date']+':'+str(left)+':'+str(first)+':'+str(last)+';').encode())
                for variant in ['point','queue']:
                    mask=day[variant][view] if view>=0 else day['mask'];counters[variant][day['date']][first:last]+=mask[first:last]
        for variant in ['point','queue']:
            name=f'seed{seed}_{variant}';record=study['training'][name];path=source/(name+'_exposure.npz')
            if sha256_file(path)!=record['exposure_sha256'] or patch.hexdigest()!=record['patch_sequence_sha256']:raise ValueError('Exposure hash changed')
            expected=counters[variant]
            with np.load(path) as a:
                if any(not np.array_equal(a[d],v) for d,v in expected.items()):raise ValueError('Actual label exposures differ')
            total=sum(int(v.sum()) for v in expected.values());unique=sum(int((v>0).sum()) for v in expected.values())
            extra=sum(int(((expected[d['date']]>0)&~d['mask']).sum()) for d in fit)
            if total!=record['total_label_exposures'] or unique!=record['unique_target_cells_supervised'] or extra!=record['new_distinct_target_cells_outside_original_mask']:
                raise ValueError('Recorded label population/exposure count differs')
            exposure_checks.append({'role':name,'full_actual_exposure_arrays_exact':True,'total_label_exposures':total,'unique_supervised_grid_cells':unique,
                'additional_distinct_target_cells':extra,'not_equal_to_control_label_population':True})
    paths,models=load_roles(args.study,folder/'State_primary.pkl');release=Release('data/kaggle_public');panel=config['panel']
    net,topology=release.network(panel),release.topology(panel);profile=models['incumbent'].profile;version=models['incumbent'].feature_version
    files={partition_date(release,p):p for p in release.files(panel,'train')};raw_files={p.name:p for p in release.files(panel,'train',False)};template=release.template(1,panel,'train')
    rows=[];maximum=0.
    for phase in ['selection','assessment']:
        for path in sorted((cache/phase).glob('*.npz')):
            if sha256_file(path)!=study['cache_sha256'][path.as_posix()]:raise ValueError('Original evaluator changed')
            frame=release.read(files[path.stem]);features=StateFeatures(frame,net,topology,profile,version);target=day_template(release,panel,'train',frame,template)
            teacher=target_truth(release,panel,target,release.read(raw_files[files[path.stem].name]))[VALUES].to_numpy(float)
            lanes=target.link_id.map(net.lanes).to_numpy(float);teacher[:,1]/=lanes
            with np.load(path) as a:
                if not np.array_equal(a['x'],reconstruction_input(features)) or not np.array_equal(a['station_truth'],teacher) or not np.array_equal(a['target_indices'],features.lookup(target)):
                    raise ValueError('Fresh legal input/raw teacher/target index differs')
            with np.load(source/'predictions'/phase/(path.stem+'.npz')) as saved:
                for role,model in models.items():
                    prediction=model.predict(features,target)[VALUES].to_numpy(float);prediction[:,1]/=lanes
                    wanted=saved[study['selection']['primary'] if role=='primary_export' else role]
                    error=float(np.abs(prediction-wanted).max());maximum=max(maximum,error)
                    if error>1e-8:raise ValueError('Deployed saved role differs: '+role)
                    rows.append({'phase':phase,'date':path.stem,'role':role,'maximum_lane_unit_prediction_difference':error})
        print('native_training_verify',phase,'complete',flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'exposure_checks':exposure_checks,'n_role_day_predictor_calls':len(rows),
        'maximum_lane_unit_prediction_difference':maximum,'prediction_checks':rows,
        'exports':{n:{'path':p.as_posix(),'sha256':sha256_file(p)} for n,p in paths.items()},'seconds':time.perf_counter()-start,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'All actual training label-mask exposures and deployed policy calls checked. Not independent model retraining or equal-label-budget architecture evidence.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('native_training_verified',len(rows),'maxpred',maximum,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
