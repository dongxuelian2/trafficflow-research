"""Replay all sealed deployable State choices from masked public cal frames."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from trafficflow.data import Release, VALUES
from trafficflow.state import StateModel, StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.quality_state import quality_reconstruction_input
from trafficflow.experiments import partition_date, day_template
from trafficflow.research_eval import state_from_sufficient_stats
from trafficflow.runtime import sha256_file, evidence
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats


def verify(directory):
    result = json.loads((directory/'experiment.json').read_text())
    if json.loads((directory/'selection_sealed.json').read_text()) != result['selected']:
        raise ValueError('Selection changed after seal')
    prepared = result['prepared']
    cache = Path(result['registered']['protocol']['cache'])
    output_mode = result['registered']['protocol'].get('output_coordinates', 'vq')
    release = Release('data/kaggle_public')
    panel = prepared['panel']
    net, topology = release.network(panel), release.topology(panel)
    files = {partition_date(release, p): p for p in release.files(panel, 'train')}
    template = release.template(1, panel, 'train')
    roles = {}
    for role, choice in result['selected'].items():
        path = directory/f'State_{role}.pkl'
        if sha256_file(path) != result['deployable_model_sha256'][role]:
            raise ValueError('Deployed model changed')
        model = StateModel.load(path)
        schema = 'quality18' if choice['channels'] == 18 else 'strict12'
        if model.input_schema != schema or getattr(model, 'output_coordinates', 'vq') != output_mode:
            raise ValueError('Deployment coordinates/input schema differ')
        rows, increments = [], []
        for source in sorted((cache/'selection').glob('*.npz')):
            day = dict(np.load(source, allow_pickle=False))
            features = StateFeatures(release.read(files[source.stem]), net, topology, model.profile, version=model.feature_version)
            strict = reconstruction_input(features)
            extended = quality_reconstruction_input(features, strict)
            if not np.array_equal(strict, day['x_control']) or not np.array_equal(extended, day['x']):
                raise ValueError('Independent deploy input differs')
            target = day_template(release, panel, 'train', features.frame, template)
            if not np.array_equal(features.lookup(target), day['target_indices']):
                raise ValueError('Deployed target lookup differs')
            prediction = model.predict(features, target)[VALUES].to_numpy(float, copy=True)
            prediction[:, 1] /= target.link_id.map(net.lanes).to_numpy(float)
            rows.append(stats(day, prediction))
            day['x'] = day['x_control']
            factor = (net.lanes*net.length_km).reindex(day['links']).to_numpy(float)
            increments.append(increment_stats(day, prediction, factor))
        measured = {**state_from_sufficient_stats(rows),
            'density_target_pair_relative_L1': sum(r['error_L1'] for r in increments)/sum(r['truth_increment_L1'] for r in increments)}
        for field in ['S_state', 'density_target_pair_relative_L1']:
            if abs(measured[field]-choice['metrics'][field]) > 1e-7:
                raise ValueError('Sealed cal replay failed: '+field)
        roles[role] = {'model_sha256': sha256_file(path), 'n_calibration_days': len(rows),
            'schema': schema, 'output_coordinates': output_mode, 'replayed_metrics': measured,
            'independent_masked_frame_inputs_exact': True}
    source = evidence(gpu_used=torch.cuda.is_available())
    return {'source_report_sha256': sha256_file(directory/'experiment.json'), 'roles': roles,
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None, 'source': source,
        'scope': 'Independent full-cal input construction and saved-model prediction replay; not independent training, untouched final or hidden score'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    path = args.directory/'deployment_verification.json'
    if path.exists():
        raise FileExistsError('Never overwrite deployment replay')
    result = verify(args.directory)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    (Path('reports/research')/(args.directory.name+'.deployment.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({role: (r['n_calibration_days'], r['schema'], r['output_coordinates'], r['replayed_metrics']['S_state']) for role, r in result['roles'].items()}, flush=True)


if __name__ == '__main__':
    main()
