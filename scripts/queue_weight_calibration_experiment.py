"""Fixed-C30 checkpoint class-weight probability correction on matched events.

No new neural fitting or checkpoint selection. Calibration chooses mixtures and
decisions, then a seal precedes assessment-array loading. All labels remain
noisy public-train proxies on already-inspected development dates.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import Release
from trafficflow.experiments import partition_date, raw_for
from trafficflow.queue import QueueModel
from trafficflow.state import neighbors
from trafficflow.queue_events import sample_events
from trafficflow.causal_queue import released_origin_rows
from trafficflow.graph_queue import CausalGraphQueue
from trafficflow.probability_calibration import weighted_bce_probability
from trafficflow.research_eval import register_protocol, canonical_hash
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from causal_graph_queue_experiment import ProbabilityCalibration, evaluate, brier
from compare_queue_events import comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite calibration evidence')
    prior = json.loads((args.directory/'experiment.json').read_text())
    config = prior['registered']['protocol']
    original = json.loads(Path(config['matched_study']).read_text())
    sealed = json.loads((args.directory/'selection_sealed.json').read_text())
    if sealed['selected'] != prior['selected']:
        raise ValueError('C30 original winner changed')
    checkpoint_hashes = sealed['checkpoint_sha256']
    for name, digest in checkpoint_hashes.items():
        if sha256_file(args.directory/(name+'.pt')) != digest:
            raise ValueError('Original neural checkpoint changed')
    baseline_path = Path('artifacts/research')/Path(config['matched_study']).stem/'shared_threshold.pkl'
    if sha256_file(baseline_path) != config['baseline_sha256']:
        raise ValueError('Strong Queue comparator changed')
    report = Path('reports/research')/(args.output.name+'.json')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'Fixed_Queue_checkpoint_positive_class_weight_correction', 'panel': config['panel'],
        'source_study_sha256': sha256_file(args.directory/'experiment.json'),
        'checkpoint_sha256': checkpoint_hashes, 'baseline_sha256': config['baseline_sha256'],
        'neural_training_or_checkpoint_reselection': False, 'positive_weight': 4.,
        'transform': 'sigmoid(logit-log(4)); retain raw sigmoid as a control',
        'architectures': ['temporal', 'directed_graph'], 'mixture_grid': [0., .25, .5, .75, 1.],
        'decisions': 'Identical RefinedQueueModel calibration-only threshold/nonempty tuner',
        'assessment_choice': 'Seal all mixtures/decisions and winner before assessment arrays are loaded',
        'events': 'Original C30/C27 events, exact sampling metadata and target-array alignment checks',
        'hypothesis': 'Correcting weighted-BCE odds may improve probability mixing and reveal a calibration rather than representation failure',
        'theory_limit': 'Exact only for an ideal weighted BCE optimum; finite model, auxiliary speed loss, equal-event sampling and noise can still miscalibrate',
        'formal_Queue': None, 'formal_Total': None, 'uploaded': False})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    release = Release('data/kaggle_public')
    panel = config['panel']
    net, topo = release.network(panel), release.topology(panel)
    paths = release.files(panel, 'train')
    parts = {part: [p for p in paths if partition_date(release, p) in days] for part, days in prior['dates'].items()}
    baseline = QueueModel.load(baseline_path)
    origin_cache = {}
    def origin_reader(origin):
        token = str(origin)
        if token not in origin_cache:
            origin_cache[token] = released_origin_rows(release, panel, 'train', origin)
        return origin_cache[token]
    links = None
    graph = None
    def sample(part, seed, expected_meta, expected_digest):
        global links, graph
        frames = list(raw_for(release, panel, parts[part]))
        data, meta = sample_events(frames, net, topo, panel, seed, per_condition=30,
            spacing_minutes=360, origin_reader=origin_reader, onset_last_step=True)
        if canonical_hash(meta) != canonical_hash(expected_meta):
            raise ValueError('Event sampling changed')
        path = args.directory/(part+'_'+str(seed)+'.npz')
        if sha256_file(path) != expected_digest:
            raise ValueError('Causal sequence cache changed')
        arrays = dict(np.load(path, allow_pickle=False))
        expected_origins = [row['origin'] for row in sorted(meta['windows'], key=lambda row: row['origin'])]
        if arrays['origins'].tolist() != expected_origins:
            raise ValueError('Sequence origin ordering changed')
        labels = np.concatenate([y.ravel()[a.ravel()] for y, a in zip(arrays['labels'], arrays['available'])])
        if not np.array_equal(labels, data.y):
            raise ValueError('Sequence truth differs from matched proxy rows')
        if links is None:
            links = frames[0].link_id.drop_duplicates().tolist()
            if len(links) != arrays['x'].shape[2]:
                raise ValueError('Public graph node count differs')
            graph = []
            for direction in ('incoming', 'outgoing'):
                adjacency = np.zeros((len(links), len(links)), np.float32)
                for j, connected in enumerate(neighbors(topo, links, direction)):
                    if connected:
                        adjacency[j, connected] = 1/len(connected)
                graph.append(torch.tensor(adjacency, device=device))
            lane = net.lanes.reindex(links).to_numpy(float)
            geometry = np.stack([net.free_speed_kmh.reindex(links), net.capacity_vph.reindex(links)/lane,
                net.k_jam.reindex(links)/lane, net.v_cut.reindex(links)]).astype(np.float32)
            graph.append(torch.tensor(geometry, device=device))
        return data, arrays, meta
    def neural_probabilities(name, arrays):
        model = CausalGraphQueue(hidden=config['hidden'], use_graph=name == 'directed_graph').to(device)
        model.load_state_dict(torch.load(args.directory/(name+'.pt'), map_location=device, weights_only=True))
        model.eval()
        raw, corrected = [], []
        with torch.no_grad():
            for x, available in zip(arrays['x'], arrays['available']):
                logits, _ = model(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(device), *graph)
                raw.append(torch.sigmoid(logits).cpu().numpy()[0].ravel()[available.ravel()])
                corrected.append(weighted_bce_probability(logits.cpu().numpy()[0], 4.).ravel()[available.ravel()])
        del model
        return {'raw': np.concatenate(raw), 'corrected': np.concatenate(corrected)}
    cal, cal_arrays, cal_meta = sample('selection', 411, prior['samplers']['selection'], prior['cache_sha256']['selection'])
    reference = baseline.probability(cal.x)
    original_cal = evaluate(reference, cal, baseline.thresholds, baseline.require_nonempty)
    if original_cal['proxy_IoU'] != prior['selection']['shared_threshold']['metrics']['proxy_IoU']:
        raise ValueError('Original calibration comparator did not reproduce')
    result = {'registered': registered, 'source_snapshot': snapshot, 'dates': prior['dates'],
        'selection': {'shared_threshold': {'metrics': original_cal}}, 'assessment': [],
        'baseline_calibration_reproduced': True, 'raw_neural_calibration_reproduced': {},
        'probability_diagnostics': {}}
    choices = {'shared_threshold': {'alpha': 0., 'thresholds': baseline.thresholds, 'require_nonempty': baseline.require_nonempty}}
    winner = 'shared_threshold'
    for name in ('temporal', 'directed_graph'):
        probabilities = neural_probabilities(name, cal_arrays)
        original_pure = next(row for row in prior['selection'][name]['ensemble_grid'] if row['alpha'] == 1.)
        check = evaluate(probabilities['raw'], cal, original_pure['thresholds'], original_pure['require_nonempty'])
        if check['windows'] != original_pure['metrics']['windows']:
            raise ValueError('Uncorrected neural predictions did not reproduce')
        result['raw_neural_calibration_reproduced'][name] = True
        result['probability_diagnostics'][name] = {variant: brier(p, cal) for variant, p in probabilities.items()}
        for variant, probability in probabilities.items():
            key = name+'_'+variant
            rows = []
            for alpha in (0., .25, .5, .75, 1.):
                p = (1-alpha)*reference+alpha*probability
                tuner = ProbabilityCalibration(p)
                tuning = tuner.tune(cal)
                rows.append({'alpha': alpha, 'thresholds': tuner.thresholds, 'require_nonempty': tuner.require_nonempty,
                    'metrics': evaluate(p, cal, tuner.thresholds, tuner.require_nonempty), 'tuning_selected': tuning['selected']})
            chosen = max(rows, key=lambda row: (row['metrics']['proxy_IoU'], -row['alpha']))
            choices[key] = {k: chosen[k] for k in ('alpha', 'thresholds', 'require_nonempty')}
            result['selection'][key] = {'metrics': chosen['metrics'], 'settings': choices[key], 'ensemble_grid': rows}
            if chosen['metrics']['proxy_IoU'] > result['selection'][winner]['metrics']['proxy_IoU']:
                winner = key
    result['selected'] = winner
    (args.output/'selection_sealed.json').write_text(json.dumps({'selected': winner, 'settings': choices,
        'checkpoint_sha256': checkpoint_hashes}, indent=2), encoding='utf-8')
    for previous_run in prior['assessment']:
        seed = previous_run['seed']
        data, arrays, meta = sample('assessment', seed, previous_run['sampler'], previous_run['cache_sha256'])
        reference = baseline.probability(data.x)
        scores = {'shared_threshold': evaluate(reference, data, baseline.thresholds, baseline.require_nonempty)}
        if scores['shared_threshold']['windows'] != previous_run['models']['shared_threshold']['windows']:
            raise ValueError('Original assessment comparator did not reproduce')
        for name in ('temporal', 'directed_graph'):
            for variant, probability in neural_probabilities(name, arrays).items():
                key = name+'_'+variant
                choice = choices[key]
                p = (1-choice['alpha'])*reference+choice['alpha']*probability
                scores[key] = evaluate(p, data, choice['thresholds'], choice['require_nonempty'])
        result['assessment'].append({'seed': seed, 'sampler': meta, 'models': scores})
    result['selected_comparison'] = comparison(result, 'shared_threshold', winner)
    result['comparisons'] = {key: comparison(result, 'shared_threshold', key) for key in choices if key != 'shared_threshold'}
    result.update(seconds=time.perf_counter()-start, source=evidence(gpu_used=device.type == 'cuda'),
        formal_Queue=None, formal_Total=None, uploaded=False,
        limitations='No neural refitting. Class-weight correction alone does not address event sampling/noise or auxiliary-loss calibration. All assessment dates were previously inspected development dates, not a new final test.')
    (args.output/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(winner, result['probability_diagnostics'], result['selected_comparison'], result['seconds'], flush=True)
