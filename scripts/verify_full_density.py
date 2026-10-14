"""Rebuild public masked inputs and raw TRAIN evaluator labels for saved models."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from trafficflow.data import Release, VALUES, eligible
from trafficflow.state import StateModel, StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.density_label_metrics import public_increment_stats, aggregate_public_increments
from trafficflow.experiments import partition_date, day_template
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import state_from_sufficient_stats
from train_graph_experiment import stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--old-control',type=Path,required=True)
    parser.add_argument('--control-format',choices=['quality','adaptive'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite an independent full density replay')
    result=json.loads((args.study/'experiment.json').read_text())
    if json.loads((args.study/'selection_sealed.json').read_text())!=result['selected']:raise ValueError('Cal choices changed')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip')
    cache=Path(result['registered']['protocol']['cache']);panel=result['prepared']['panel']
    release=Release('data/kaggle_public');net,topology=release.network(panel),release.topology(panel)
    masked={partition_date(release,p):p for p in release.files(panel,'train')}
    raw={partition_date(release,p):p for p in release.files(panel,'train',False)}
    template=release.template(1,panel,'train');roles={}
    for role,choice in result['selected'].items():
        model_path=args.study/f'State_{role}.pkl'
        if sha256_file(model_path)!=result['deployable_model_sha256'][role]:raise ValueError('Model changed')
        model=StateModel.load(model_path);rows=[];increments=[]
        for path in sorted((cache/'selection').glob('*.npz')):
            d=dict(np.load(path,allow_pickle=False));d['x']=d.get('x_control',d['x'])
            features=StateFeatures(release.read(masked[path.stem]),net,topology,model.profile,version=model.feature_version)
            if not np.array_equal(reconstruction_input(features),d['x']):raise ValueError('Strict12 inputs differ')
            target=day_template(release,panel,'train',features.frame,template)
            if not np.array_equal(features.lookup(target),d['target_indices']):raise ValueError('Target lookup differs')
            # Model prediction is made before accessing raw evaluator labels.
            prediction=model.predict(features,target)[VALUES].to_numpy(float,copy=True)
            prediction[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
            truth=release.read(raw[path.stem]);truth=truth.loc[eligible(truth)]
            full=np.stack([truth.pivot_table(index='timestamp',columns='link_id',values=c,aggfunc='mean').reindex(
                index=features.times,columns=features.links).to_numpy(float) for c in VALUES],axis=-1)
            full[...,1]/=net.lanes.reindex(features.links).to_numpy(float)
            valid=np.isfinite(full).all(axis=-1);full/=np.array([25.,600.])
            labels=np.where(valid[...,None],full,0.).astype(np.float32)
            if not np.array_equal(labels,d['y_full']) or not np.array_equal(valid,d['eligible_full']):
                raise ValueError('Independent raw TRAIN loss labels differ')
            rows.append(stats(d,prediction))
            factor=(net.lanes*net.length_km).reindex(features.links).to_numpy(np.float32)
            increments.append(public_increment_stats(d,prediction,factor))
        measured={**state_from_sufficient_stats(rows),**aggregate_public_increments(increments)}
        for field in ['S_state','public_coverage_increment_relative_L1']:
            if abs(measured[field]-choice['metrics'][field])>1e-7:raise ValueError('Sealed cal replay failed: '+field)
        roles[role]={'model_sha256':sha256_file(model_path),'n_calibration_days':len(rows),
            'strict_inputs_and_raw_TRAIN_labels_rebuilt_exact':True,'replayed_metrics':measured}
    pairs=[('state_only','attention12' if args.control_format=='quality' else 'adaptive_attention'),
           ('legacy_density','attention12_density' if args.control_format=='quality' else 'adaptive_attention_density')]
    controls=[]
    for new_name,old_name in pairs:
        if args.control_format=='quality':
            previous=json.loads((args.old_control/'experiment.json').read_text())['primary_per_model'][old_name]
            old_path=args.old_control/f"{old_name}_epoch{previous['epoch']:02d}.pt"
        else:old_path=args.old_control/(old_name+'.pt')
        old_state=torch.load(old_path,map_location='cpu',weights_only=True)
        matches=[]
        for path in sorted(args.study.glob(new_name+'_epoch*.pt')):
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state.keys()==old_state.keys() and all(torch.equal(state[k],old_state[k]) for k in state):matches.append(path.name)
        if len(matches)!=1:raise ValueError('Original training control not exactly reproduced: '+new_name)
        controls.append({'new_name':new_name,'old_file':old_path.as_posix(),'old_sha256':sha256_file(old_path),
                         'matching_new_checkpoint':matches[0],'all_tensors_exact':True})
    output={'source_experiment_sha256':sha256_file(args.study/'experiment.json'),'source_snapshot':snapshot,
        'roles':roles,'original_training_controls':controls,
        'four_variant_patch_sequence_identical':len({r['sample_sequence_sha256'] for r in result['training'].values()})==1,
        'source':evidence(gpu_used=torch.cuda.is_available()),'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limitations':'Independent cal inputs/raw TRAIN labels, saved prediction and prior-control tensor replay. Public pair validity is immutable cached data, not independently regenerated here. Not a new fit seed/final test or actual boundary flux.'}
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print('verified',[(k,v['replayed_metrics']['S_state']) for k,v in roles.items()],controls,flush=True)


if __name__=='__main__':main()
