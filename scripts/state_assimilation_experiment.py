"""Calibrate frozen-prior residual assimilation, seal, then assess.

Uses masked public State frames only for inference. Existing raw-train labels
in immutable graph caches are evaluator data, never observation anchors.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from trafficflow.data import Release, VALUES
from trafficflow.state import StateModel, StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.quality_state import quality_channels
from trafficflow.state_assimilation import (AssimilationParameters,
    AssimilatedStateModel, normalized_prior_grid, compressed_edges, assimilate_residuals, counterfactual_anchor_prior)
from trafficflow.experiments import partition_date, day_template
from trafficflow.runtime import capture_source_snapshot, sha256_file, evidence
from trafficflow.research_eval import register_protocol, state_from_sufficient_stats, paired_block_interval
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats


def configurations():
    result = []
    for family, low, robust, spatial, adaptive in [
        ('strict_temporal', 0., False, 0., False),
        ('quality_temporal', .25, False, 0., False),
        ('quality_robust_temporal', .25, True, 0., False),
        ('quality_robust_graph', .25, True, .5, False),
        ('quality_robust_adaptive_graph', .25, True, .5, True)]:
        for prior in [.25, 1.]:
            for temporal in [1., 4.]:
                parameter = AssimilationParameters(prior, temporal, spatial, low,
                    (.4, .2) if robust else None, (.4, .2) if adaptive else None)
                result.append({'id': f'{family}_p{prior}_t{temporal}', 'family': family,
                               'parameters': asdict(parameter)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--prior-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--anchor-folds', type=int, default=0)
    parser.add_argument('--anchor-seed', type=int, default=20261013)
    args = parser.parse_args()
    report = Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():
        raise FileExistsError('Never overwrite assimilation research evidence')
    info = json.loads((args.cache/'prepared.json').read_text())
    options = configurations()
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'Frozen_learned_prior_quality_residual_assimilation',
        'cache': args.cache.as_posix(), 'cache_sha256': sha256_file(args.cache/'prepared.json'), 'panel': info['panel'],
        'fold': info['fold'], 'prior_file': args.prior_file.as_posix(),
        'prior_sha256': sha256_file(args.prior_file), 'configurations': options,
        'anchor_folds': args.anchor_folds, 'anchor_seed': args.anchor_seed,
        'anchor_centering': 'Original conditional prior' if not args.anchor_folds else 'Four measurement channels withheld at each visible cell once, deterministic random groups; original target prior unchanged',
        'strengths': [0., .25, .5, 1.], 'fit': 'No learned prior refitting or raw-train feature inputs',
        'precision': 'lambda_prior I + observed precision + temporal and compressed undirected spatial Laplacians',
        'quality': 'Strict finite observations weight1; extra values weight .25*min(pct/.75,1)^2 with known quality',
        'robustness': 'Fixed normalized innovation cutoffs speed .4, lane-flow .2; same cutoffs optionally gate time edges',
        'solver': 'Jacobi-preconditioned CG rtol1e-8, max400; verify actual relative residual<=2e-8',
        'selection': 'Highest calibration pooled State, ties favor strength0/lower strength; architecture/hyperparameters cal only',
        'joint_selection': 'Minimum incomplete target-touching increment L1, cal State >= frozen prior -.001',
        'assessment': 'Existing public-train development dates; all choices sealed before reading assessment caches',
        'causality': 'Offline State with bidirectional observations; prohibited as future Queue features',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    release = Release('data/kaggle_public')
    panel = info['panel']
    prior = StateModel.load(args.prior_file)
    net, topology = release.network(panel), release.topology(panel)
    template = release.template(1, panel, 'train')
    paths = {partition_date(release, p): p for p in release.files(panel, 'train')}
    solver_records, prepared_records = [], []

    def load(part):
        days = []
        (args.output/part).mkdir()
        for path in sorted((args.cache/part).glob('*.npz')):
            day = dict(np.load(path, allow_pickle=False))
            frame = release.read(paths[path.stem])
            features = StateFeatures(frame, net, topology, prior.profile, version=prior.feature_version)
            target = day_template(release, panel, 'train', frame, template)
            strict_encoded = reconstruction_input(features)
            expected = day.get('x_control', day['x'])
            if not np.array_equal(strict_encoded, expected[..., :12]) or not np.array_equal(features.lookup(target), day['target_indices']):
                raise ValueError('Original cache input/target lookup differs')
            mean = normalized_prior_grid(prior, features, target)
            strict, quality = features.raw/np.array([25., 600.]), quality_channels(features)
            original = prior.predict(features, target)[VALUES].to_numpy(float, copy=True)
            original[:, 1] /= target.link_id.map(net.lanes).to_numpy(float)
            reconstructed = mean.reshape(-1, 2)[day['target_indices']]*np.array([25., 600.])
            max_drift = float(np.max(np.abs(original-reconstructed)))
            if max_drift > 1e-8:
                raise ValueError('Frozen prior grid differs from standalone target prediction')
            edges = compressed_edges(topology, features.links)
            inputs = dict(prior=mean, strict=strict, quality=quality, edges=edges)
            anchor_certificates = None
            if args.anchor_folds:
                inputs['anchor_prior'], anchor_certificates = counterfactual_anchor_prior(
                    prior, features, target, mean, args.anchor_folds, args.anchor_seed)
            np.savez_compressed(args.output/part/path.name, **inputs)
            prepared_records.append({'part': part, 'date': path.stem,
                'source_cache_sha256': sha256_file(path),
                'masked_public_source_sha256': sha256_file(paths[path.stem]),
                'inference_inputs_sha256': sha256_file(args.output/part/path.name),
                'counterfactual_anchor_certificates': anchor_certificates,
                'max_frozen_prior_grid_target_drift': max_drift})
            days.append((path.stem, day, inputs))
        return days

    def evaluate(days, choices, part):
        grouped = {}
        for choice in choices:
            grouped.setdefault(choice['id'], []).append(choice)
        records, increments = {r['key']: [] for r in choices}, {r['key']: [] for r in choices}
        factor = (net.lanes*net.length_km).reindex(days[0][1]['links']).to_numpy(float)
        for date, day, inputs in days:
            for config_id, variants in grouped.items():
                if config_id == 'frozen_prior':
                    delta = np.zeros_like(inputs['prior'])
                else:
                    parameters = AssimilationParameters(**variants[0]['parameters'])
                    delta, certificates = assimilate_residuals(**inputs, parameters=parameters)
                    solver_records.append({'part': part, 'date': date, 'config': config_id,
                                           'certificates': certificates})
                for choice in variants:
                    prediction = (inputs['prior']+choice['strength']*delta).reshape(-1, 2)[day['target_indices']]*np.array([25., 600.])
                    prediction[:, 0] = np.clip(prediction[:, 0], 0., net.free_speed_kmh.reindex(day['links']).to_numpy(float)[day['target_indices'] % len(day['links'])]*1.05)
                    lane_cap = (net.capacity_vph/net.lanes).reindex(day['links']).to_numpy(float)
                    prediction[:, 1] = np.clip(prediction[:, 1], 0., lane_cap[day['target_indices'] % len(day['links'])]*1.05)
                    records[choice['key']].append({'date': date, **stats(day, prediction)})
                    increments[choice['key']].append(increment_stats(day, prediction, factor))
            print(part, date, 'complete', flush=True)
        result = {}
        for choice in choices:
            rows, changes = records[choice['key']], increments[choice['key']]
            result[choice['key']] = {**choice, 'metrics': {**state_from_sufficient_stats(rows),
                'days': rows, 'density_target_pair_relative_L1': sum(r['error_L1'] for r in changes)/max(sum(r['truth_increment_L1'] for r in changes), 1e-9)}}
        return result

    frozen = {'id': 'frozen_prior', 'family': 'control', 'parameters': None,
              'strength': 0., 'key': 'frozen_prior'}
    all_choices = [frozen]+[{**config, 'strength': strength, 'key': config['id']+f'_a{strength}'}
        for config in options for strength in [.25, .5, 1.]]
    with threadpool_limits(limits=1):
        calibration = evaluate(load('selection'), all_choices, 'selection')
        def primary_key(row):
            return (row['metrics']['S_state'], -row['strength'])
        primary = max(calibration.values(), key=primary_key)
        control = calibration['frozen_prior']
        admissible = [r for r in calibration.values() if r['metrics']['S_state'] >= control['metrics']['S_state']-.001]
        joint = min(admissible, key=lambda r: (r['metrics']['density_target_pair_relative_L1'], r['strength']))
        family_choices = {family: max([control]+[r for r in calibration.values() if r['family'] == family], key=primary_key)
                          for family in dict.fromkeys(r['family'] for r in options)}
        sealed = {'primary': primary, 'joint': joint, 'family_choices': family_choices}
        (args.output/'selection_sealed.json').write_text(json.dumps(sealed, indent=2, allow_nan=False), encoding='utf-8')
        print('calibration_sealed', primary['key'], joint['key'], flush=True)
        selected = {r['key']: r for r in [control, primary, joint, *family_choices.values()]}
        assessment = evaluate(load('assessment'), list(selected.values()), 'assessment')
    intervals = {key: paired_block_interval([r['daily_State'] for r in assessment['frozen_prior']['metrics']['days']],
        [r['daily_State'] for r in row['metrics']['days']]) for key, row in assessment.items() if key != 'frozen_prior'}
    model_hashes = {}
    for role, choice in [('primary', primary), ('joint', joint)]:
        parameters = AssimilationParameters(**choice['parameters']) if choice['parameters'] else AssimilationParameters()
        deployed = AssimilatedStateModel(prior, parameters, choice['strength'], args.anchor_folds, args.anchor_seed)
        path = args.output/('State_'+role+'.pkl')
        deployed.save(path)
        model_hashes[role] = sha256_file(path)
    gpu_used = getattr(prior, '_network', None) is not None and next(prior._network.parameters()).is_cuda
    result = {'registered': registered, 'source_snapshot': snapshot, 'prepared': info,
        'calibration': calibration, 'selected': sealed, 'assessment': assessment,
        'paired_day_intervals': intervals, 'solver_records': solver_records,
        'inference_records': prepared_records, 'deployed_model_sha256': model_hashes,
        'seconds': time.perf_counter()-start, 'source': evidence(gpu_used=gpu_used),
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None,
        'limitations': 'No independent prior training seed. Calibration grid may overfit; assessment is previously inspected development data. SPD certificates prove numerical solve, not traffic physics. Non-target raw labels are not observation anchors. Offline future observations cannot be reused for Queue.'}
    (args.output/'experiment.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('assessment', {k: r['metrics']['S_state'] for k, r in assessment.items()}, 'seconds', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
