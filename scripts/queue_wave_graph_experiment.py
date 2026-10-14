"""Matched causal graph-range ablation using public travel-time geometry."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.state import neighbors
from trafficflow.graph_queue import CausalGraphQueue, queue_event_loss
from trafficflow.queue_graph_geometry import travel_time_neighborhoods
from trafficflow.probability_calibration import weighted_bce_probability
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot, evidence, sha256_file
from causal_graph_queue_experiment import ProbabilityCalibration, evaluate, brier
from matched_queue_cache import MatchedQueueCache
from compare_queue_events import comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite Queue range evidence')
    cache = MatchedQueueCache(args.directory)
    seed = cache.config['seed']
    report = Path('reports/research')/(args.output.name+'.json')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'Causal_Queue_public_wave_range_graph_ablation', 'panel': cache.panel,
        'source_cache_report_sha256': sha256_file(args.directory/'experiment.json'), 'seed': seed,
        'ranges': {'direct': None, 'wave_15min': 15., 'wave_30min': 30.},
        'model': 'Same CausalGraphQueue, hidden 32, 20076 parameters; only spatial operators differ',
        'inputs': 'Same nine legal past/current channels, supplied condition and public FD geometry',
        'labels': 'Same noisy public-train observation labels and source event metadata',
        'epochs': 20, 'steps_per_epoch': 80, 'batch_size': 4,
        'optimizer': 'AdamW lr=.001 weight_decay=.0001 clip_norm=1',
        'loss': 'Same equal-event BCE positive weight 4 + .1 speed MAE; onset classification at final step only',
        'checkpoint': 'Corrected domain-condition event Brier every five epochs; same rule for all three ranges',
        'probability': 'sigmoid(logit-log(4)) for all three models', 'mixture_grid': [0., .25, .5, .75, 1.],
        'selection': 'Checkpoint, range, mixture and conditional threshold/nonempty decision by calibration only; seal before assessment loading',
        'physics_limit': 'Travel-time neighborhoods are a representation. Repeated layers/GroupNorm do not enforce a hard propagation cone.',
        'promotion': 'Same original Queue gate: >=.01 proxy gain with positive day-block CI on two folds/directions, no direction loss >.01, multiple fit seeds',
        'formal_Queue': None, 'formal_Total': None, 'uploaded': False})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    fit = cache.fit()
    cal, cal_arrays, cal_meta = cache.sample('selection', 411)
    lane = cache.net.lanes.reindex(cache.links).to_numpy(float)
    geometry = np.stack([cache.net.free_speed_kmh.reindex(cache.links), cache.net.capacity_vph.reindex(cache.links)/lane,
        cache.net.k_jam.reindex(cache.links)/lane, cache.net.v_cut.reindex(cache.links)]).astype(np.float32)
    graphs = {}
    for name, horizon in registered['protocol']['ranges'].items():
        if horizon is None:
            operators = []
            for direction in ('incoming', 'outgoing'):
                matrix = np.zeros((len(cache.links), len(cache.links)), np.float32)
                for j, connected in enumerate(neighbors(cache.topo, cache.links, direction)):
                    if connected:
                        matrix[j, connected] = 1/len(connected)
                operators.append(matrix)
        else:
            operators = travel_time_neighborhoods(cache.links, cache.net, cache.topo, horizon)
        np.savez(args.output/(name+'_geometry.npz'), incoming=operators[0], outgoing=operators[1], geometry=geometry)
        graphs[name] = [torch.tensor(matrix, device=device) for matrix in (*operators, geometry)]
    def predict(model, arrays, graph):
        probabilities = []
        with torch.no_grad():
            for x, valid in zip(arrays['x'], arrays['available']):
                logits, _ = model(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(device), *graph)
                probabilities.append(weighted_bce_probability(logits.cpu().numpy()[0], 4.).ravel()[valid.ravel()])
        return np.concatenate(probabilities)
    reference = cache.baseline.probability(cal.x)
    reference_metrics = evaluate(reference, cal, cache.baseline.thresholds, cache.baseline.require_nonempty)
    if reference_metrics['proxy_IoU'] != cache.report['selection']['shared_threshold']['metrics']['proxy_IoU']:
        raise ValueError('Strong tree calibration did not reproduce')
    result = {'registered': registered, 'source_snapshot': snapshot, 'dates': cache.report['dates'],
        'selection': {'shared_threshold': {'metrics': reference_metrics}}, 'training': {}, 'assessment': []}
    settings = {'shared_threshold': {'alpha': 0., 'thresholds': cache.baseline.thresholds, 'require_nonempty': cache.baseline.require_nonempty}}
    selected = 'shared_threshold'
    condition_groups = {value: np.flatnonzero(fit['ongoing'] == value) for value in (False, True)}
    if any(not len(indices) for indices in condition_groups.values()):
        raise ValueError('Both training conditions are required')
    for name, graph in graphs.items():
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = CausalGraphQueue(hidden=32, use_graph=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        best = np.inf
        history = []
        for epoch in range(20):
            model.train()
            losses = []
            for _ in range(80):
                indices = []
                for _ in range(4):
                    group = condition_groups[bool(rng.integers(2))]
                    indices.append(group[rng.integers(len(group))])
                x = torch.from_numpy(fit['x'][indices].transpose(0, 3, 1, 2).copy()).to(device)
                targets = [torch.from_numpy(fit[k][indices]).to(device) for k in ('labels', 'truth_speed', 'available', 'ongoing')]
                optimizer.zero_grad(set_to_none=True)
                logits, speed = model(x, *graph)
                loss = queue_event_loss(logits, speed, *targets)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite graph-range training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if (epoch+1) % 5 == 0:
                model.eval()
                score = brier(predict(model, cal_arrays, graph), cal)
                history.append({'epoch': epoch+1, 'training_loss': float(np.mean(losses)), 'corrected_selection_domain_Brier': score})
                if score < best:
                    best = score
                    torch.save(model.state_dict(), args.output/(name+'.pt'))
                print(name, epoch+1, score, flush=True)
        model.load_state_dict(torch.load(args.output/(name+'.pt'), map_location=device, weights_only=True))
        model.eval()
        probability = predict(model, cal_arrays, graph)
        choices = []
        for alpha in (0., .25, .5, .75, 1.):
            mixed = (1-alpha)*reference+alpha*probability
            tuner = ProbabilityCalibration(mixed)
            tuning = tuner.tune(cal)
            choices.append({'alpha': alpha, 'thresholds': tuner.thresholds, 'require_nonempty': tuner.require_nonempty,
                'metrics': evaluate(mixed, cal, tuner.thresholds, tuner.require_nonempty), 'tuning_selected': tuning['selected']})
        choice = max(choices, key=lambda row: (row['metrics']['proxy_IoU'], -row['alpha']))
        settings[name] = {k: choice[k] for k in ('alpha', 'thresholds', 'require_nonempty')}
        result['selection'][name] = {'metrics': choice['metrics'], 'settings': settings[name], 'ensemble_grid': choices}
        result['training'][name] = {'history': history, 'parameters': sum(p.numel() for p in model.parameters())}
        if choice['metrics']['proxy_IoU'] > result['selection'][selected]['metrics']['proxy_IoU']:
            selected = name
        del model, optimizer
    result['selected'] = selected
    result['checkpoint_sha256'] = {n: sha256_file(args.output/(n+'.pt')) for n in graphs}
    (args.output/'selection_sealed.json').write_text(json.dumps({'selected': selected, 'settings': settings,
        'checkpoint_sha256': result['checkpoint_sha256']}, indent=2), encoding='utf-8')
    for previous in cache.report['assessment']:
        data, arrays, metadata = cache.sample('assessment', previous['seed'])
        reference = cache.baseline.probability(data.x)
        scores = {'shared_threshold': evaluate(reference, data, cache.baseline.thresholds, cache.baseline.require_nonempty)}
        if scores['shared_threshold']['windows'] != previous['models']['shared_threshold']['windows']:
            raise ValueError('Matched tree assessment did not reproduce')
        for name, graph in graphs.items():
            model = CausalGraphQueue(hidden=32, use_graph=True).to(device)
            model.load_state_dict(torch.load(args.output/(name+'.pt'), map_location=device, weights_only=True))
            model.eval()
            choice = settings[name]
            mixed = (1-choice['alpha'])*reference+choice['alpha']*predict(model, arrays, graph)
            scores[name] = evaluate(mixed, data, choice['thresholds'], choice['require_nonempty'])
            del model
        result['assessment'].append({'seed': previous['seed'], 'sampler': metadata, 'models': scores})
    result['selected_comparison'] = comparison(result, 'shared_threshold', selected)
    result['comparisons'] = {name: comparison(result, 'shared_threshold', name) for name in graphs}
    result.update(seconds=time.perf_counter()-start, source=evidence(gpu_used=device.type == 'cuda'),
        gpu_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
        formal_Queue=None, formal_Total=None, uploaded=False,
        limitations='One direction/fold/fit seed, public noisy proxy labels. Same sequence inputs/loss/budget across three spatial ranges. Spatial reach is not a strict wave cone. Assessment dates were already inspected development dates. No formal Queue claim.')
    (args.output/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(selected, result['selected_comparison'], result['seconds'], flush=True)
