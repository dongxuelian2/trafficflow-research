"""Independent saved-model cal replay and real-input sparse direct reference."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits
from trafficflow.data import Release, VALUES
from trafficflow.state import StateModel, StateFeatures
from trafficflow.quality_state import quality_channels
from trafficflow.state_assimilation import (AssimilationParameters, assimilate_residuals,
    observation_residuals, precision_matrix)
from trafficflow.experiments import partition_date, day_template
from trafficflow.research_eval import state_from_sufficient_stats
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from train_graph_experiment import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():
        raise FileExistsError('Never overwrite an independent replay')
    source = args.study/'experiment.json'
    run = json.loads(source.read_text())
    folder.mkdir(parents=True)
    snapshot = capture_source_snapshot(folder/'source_snapshot.zip')
    start = time.perf_counter()
    panel = run['registered']['protocol']['panel']
    cache = Path(run['registered']['protocol'].get('cache', ''))
    # Older runs registered the prepared-cache SHA and retain its full metadata.
    if not (cache/'selection').is_dir():
        # Source path supplied explicitly in the start archive command record
        # is unnecessary: identify the original immutable cache by prepared SHA.
        expected = run['registered']['protocol']['cache_sha256']
        matches = [p.parent for p in Path('artifacts/research').glob('*/prepared.json') if sha256_file(p) == expected]
        if len(matches) != 1:
            raise ValueError('Original cache cannot be identified uniquely')
        cache = matches[0]
    release = Release('data/kaggle_public')
    paths = {partition_date(release, p): p for p in release.files(panel, 'train')}
    net, topo = release.network(panel), release.topology(panel)
    template = release.template(1, panel, 'train')
    roles, direct_reference = {}, None
    with threadpool_limits(limits=1):
        for role in ['primary', 'joint']:
            model_path = args.study/('State_'+role+'.pkl')
            if sha256_file(model_path) != run['deployed_model_sha256'][role]:
                raise ValueError('Saved model changed')
            if role == 'joint' and run['deployed_model_sha256'][role] == run['deployed_model_sha256']['primary']:
                if abs(roles['primary']['replayed_State']-run['selected'][role]['metrics']['S_state']) > 1e-10:
                    raise ValueError('Identical model roles have different sealed metrics')
                roles[role] = {**roles['primary'], 'replay_reused_identical_model_bytes': True}
                continue
            model = StateModel.load(model_path)
            choice = run['selected'][role]
            records, drift = [], []
            for path in sorted((cache/'selection').glob('*.npz')):
                day = dict(np.load(path, allow_pickle=False))
                inputs_path = args.study/'selection'/path.name
                expected_hash = next(r['inference_inputs_sha256'] for r in run['inference_records'] if r['part']=='selection' and r['date']==path.stem)
                if sha256_file(inputs_path) != expected_hash:
                    raise ValueError('Inference cache changed')
                inputs = dict(np.load(inputs_path, allow_pickle=False))
                frame = release.read(paths[path.stem])
                features = StateFeatures(frame, net, topo, model.profile, version=model.feature_version)
                target = day_template(release, panel, 'train', frame, template)
                if not np.array_equal(quality_channels(features), inputs['quality']) or not np.array_equal(
                        features.raw/np.array([25.,600.]), inputs['strict'], equal_nan=True):
                    raise ValueError('Independent visible observation inputs differ')
                prediction = model.predict(features, target)[VALUES].to_numpy(float, copy=True)
                prediction[:, 1] /= target.link_id.map(net.lanes).to_numpy(float)
                if choice['strength']:
                    delta, _ = assimilate_residuals(**inputs, parameters=AssimilationParameters(**choice['parameters']))
                else:
                    delta = np.zeros_like(inputs['prior'])
                expected = (inputs['prior']+choice['strength']*delta).reshape(-1,2)[day['target_indices']]*np.array([25.,600.])
                expected[:,0] = np.clip(expected[:,0],0,target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                expected[:,1] = np.clip(expected[:,1],0,target.link_id.map(net.capacity_vph/net.lanes).to_numpy(float)*1.05)
                difference = float(np.max(np.abs(prediction-expected)))
                if difference > 1e-7:
                    raise ValueError('Saved model differs from sealed cache prediction')
                drift.append(difference)
                records.append(stats(day, prediction))
                if direct_reference is None:
                    # All producer solves retain full-grid residual certificates.
                    # This independent direct reference covers a real 24-slot
                    # subset, not an independent full-day or model fit.
                    subset = {k: v[:24] if k != 'edges' else v for k,v in inputs.items()}
                    parameter = AssimilationParameters(.25,4.,.5,.25,(.4,.2),(.4,.2))
                    correction, certificates = assimilate_residuals(**subset, parameters=parameter)
                    anchor = subset.get('anchor_prior',subset['prior'])
                    innovation, weights = observation_residuals(anchor,subset['strict'],subset['quality'],parameter)
                    differences = []
                    for channel in range(2):
                        matrix = precision_matrix(subset['prior'].shape[:2],weights,subset['prior'],subset['edges'],parameter,channel)
                        rhs=(weights[...,channel]*innovation[...,channel]).ravel()
                        direct=spsolve(matrix.tocsc(),rhs)
                        differences.append(float(np.max(np.abs(direct-correction[...,channel].ravel()))))
                    if max(differences)>2e-6:
                        raise ValueError('CG differs from independent sparse direct solve')
                    direct_reference={'date':path.stem,'time_slots':24,'nodes':inputs['prior'].shape[1],
                        'parameters':parameter.__dict__,'max_channel_abs_difference':differences,
                        'CG_certificates':certificates,'reference':'SciPy SuperLU spsolve, real 24-slot subset'}
                print(role,path.stem,'verified',flush=True)
            metric=state_from_sufficient_stats(records)
            if abs(metric['S_state']-choice['metrics']['S_state'])>1e-10:
                raise ValueError('Full-cal standalone score differs')
            roles[role]={'model_sha256':sha256_file(model_path),'n_calibration_days':len(records),
                'max_target_prediction_abs_drift':max(drift),'replayed_State':metric['S_state'],
                'visible_inputs_independently_rebuilt':True,
                'counterfactual_groups':getattr(model,'anchor_folds',0),
                'legacy_anchor_attribute_absent':not hasattr(model,'anchor_folds')}
    gpu_used=getattr(model.prior_model,'_network',None) is not None and next(model.prior_model._network.parameters()).is_cuda
    result={'source_experiment_sha256':sha256_file(source),'source_snapshot':snapshot,
        'roles':roles,'sparse_direct_reference':direct_reference,'seconds':time.perf_counter()-start,
        'source':evidence(gpu_used=gpu_used),'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'All cal dates rebuilt from public masked observations for each distinct saved artifact; identical role file SHA reuses that replay explicitly. Independent numerical direct reference on a 24-slot subset. No independent prior training or hidden score.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('verified',roles,'direct',direct_reference['max_channel_abs_difference'],flush=True)


if __name__=='__main__':
    main()
