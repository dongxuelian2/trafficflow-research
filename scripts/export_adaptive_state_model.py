"""Export and independently replay a sealed adaptive State candidate."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from trafficflow.adaptive_graph_state import AdaptiveGraphState,AdaptiveStateModel,reconstruction_input
from trafficflow.state import StateFeatures,StateModel
from trafficflow.data import Release,VALUES
from trafficflow.experiments import partition_date,day_template
from trafficflow.research_eval import state_from_sufficient_stats
from trafficflow.runtime import sha256_file,evidence
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    manifest=args.directory/'deployment_verification.json'
    if manifest.exists():raise FileExistsError('Never overwrite deployment verification')
    r=json.loads((args.directory/'experiment.json').read_text())
    config=r['registered']['protocol']
    choice=r['selected']
    sealed=json.loads((args.directory/'selection_sealed.json').read_text())
    if choice!=sealed:raise ValueError('Selection changed after sealing')
    checkpoint=args.directory/(choice['name']+'.pt')
    if sha256_file(checkpoint)!=r['model_sha256'][checkpoint.name]:raise ValueError('Neural checkpoint changed')
    baseline_path=Path(config['baseline_file'])
    if sha256_file(baseline_path)!=config['baseline_sha256']:raise ValueError('Strong baseline changed')
    baseline=StateModel.load(baseline_path)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    release=Release('data/kaggle_public');panel=config['panel']
    net,topo=release.network(panel),release.topology(panel)
    deployed=AdaptiveStateModel(baseline,state,topo,choice['alpha'],hidden=config['hidden'],attention=choice['name']!='graph_geometry')
    output=args.directory/'State_selected.pkl'
    if output.exists():
        if sha256_file(output)!=r.get('deployable_model_sha256'):
            raise ValueError('Existing standalone model changed after its recorded export')
        deployed=StateModel.load(output)
        if not isinstance(deployed,AdaptiveStateModel) or deployed.alpha!=choice['alpha']:
            raise ValueError('Unexpected existing standalone model')
    cache=Path(config['cache'])
    paths={partition_date(release,p):p for p in release.files(panel,'train')}
    template=release.template(1,panel,'train')
    rows=[];inc=[];checked=[]
    for path in sorted((cache/'selection').glob('*.npz')):
        day=dict(np.load(path,allow_pickle=False))
        f=StateFeatures(release.read(paths[path.stem]),net,topo,baseline.profile,version=baseline.feature_version)
        if not np.array_equal(reconstruction_input(f),day['x']):raise ValueError('Standalone input differs from cached input')
        target=day_template(release,panel,'train',f.frame,template)
        if not np.array_equal(f.lookup(target),day['target_indices']):raise ValueError('Target order mismatch')
        prediction=deployed.predict(f,target)[VALUES].to_numpy(float,copy=True)
        prediction[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
        rows.append(stats(day,prediction))
        factor=(net.lanes*net.length_km).reindex(day['links']).to_numpy(float)
        inc.append(increment_stats(day,prediction,factor))
        checked.append(path.stem)
    replay=state_from_sufficient_stats(rows)
    replay['density_target_pair_relative_L1']=sum(i['error_L1'] for i in inc)/sum(i['truth_increment_L1'] for i in inc)
    original=next(c['metrics'] for c in r['selection'][choice['name']]['ensemble_grid'] if c['alpha']==choice['alpha'])
    if abs(replay['S_state']-original['S_state'])>1e-7 or abs(replay['density_target_pair_relative_L1']-original['density_target_pair_relative_L1'])>1e-7:
        raise ValueError('Standalone predictions failed sealed calibration replay')
    if not output.exists():deployed.save(output)
    result={'selected':choice,'model_sha256':sha256_file(output),'checkpoint_sha256':sha256_file(checkpoint),
        'independent_public_frame_inputs_match':True,'n_checked_selection_days':len(checked),'selection_days':checked,
        'sealed_selection_metrics':original,'standalone_replay_metrics':replay,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'verification_scope':'Independent feature construction and standalone State-model prediction on the sealed calibration dates; all channels and aggregate metrics match.',
        'source':evidence(gpu_used=torch.cuda.is_available())}
    manifest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    (Path('reports/research')/(args.directory.name+'.deployment.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(output,result['model_sha256'],replay['S_state'])
