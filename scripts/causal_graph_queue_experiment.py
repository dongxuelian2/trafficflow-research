"""Temporal/graph Queue forecasts paired with preserved public-event LGB.

Only public train labels are used. Each sequence contains thirteen timestamps
at or before T. Future observations enter training targets/scoring only.
"""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.graph_queue import queue_history_input, CausalGraphQueue, queue_event_loss
from trafficflow.data import Release, eligible
from trafficflow.experiments import partition_date, raw_for
from trafficflow.queue import QueueModel
from trafficflow.state import neighbors
from trafficflow.queue_events import sample_events
from trafficflow.causal_queue import released_origin_rows
from trafficflow.queue_calibration import RefinedQueueModel, public_onset_timing
from trafficflow.metrics import grouped_iou
from trafficflow.research_eval import register_protocol, canonical_hash
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from compare_queue_events import comparison


class ProbabilityCalibration(RefinedQueueModel):
    """Use the existing decision tuner on already-computed causal probabilities."""
    def __init__(self, probability):
        super().__init__()
        self.p = np.asarray(probability, float)
        self.uses_public_onset_timing = True

    def probability(self, x):
        if len(x) != len(self.p):
            raise ValueError('Probability calibration row mismatch')
        return self.p


def predict(model, events, graph, device):
    probabilities = []
    with torch.no_grad():
        for event in events:
            x = torch.from_numpy(event['x'].transpose(2, 0, 1)[None].copy()).to(device)
            logits, _ = model(x, *graph)
            probabilities.append(torch.sigmoid(logits).cpu().numpy()[0].ravel()[event['available'].ravel()])
    return np.concatenate(probabilities)


def brier(p, data):
    domain = (data.condition == 'queue_ongoing') | (data.x.lead_minutes.to_numpy() == 30)
    frame = pd.DataFrame({'window': data.window[domain], 'condition': data.condition[domain], 'squared': (p[domain]-data.y[domain])**2})
    return float(frame.groupby(['window', 'condition']).squared.mean().reset_index().groupby('condition').squared.mean().mean())


def evaluate(p, data, thresholds, nonempty):
    model = ProbabilityCalibration(p)
    model.thresholds, model.require_nonempty = thresholds, nonempty
    pred = model.classify(data.x, data.window, data.condition)
    return {**grouped_iou(data.y, pred, data.window, data.condition),
        'domain_condition_macro_event_Brier': brier(p, data)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matched-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite causal graph research')
    previous = json.loads(args.matched_study.read_text())
    old_config = previous['registered']['protocol']
    panel, seed = old_config['panel'], old_config['fit_seed']
    baseline_path = Path('artifacts/research')/args.matched_study.stem/'shared_threshold.pkl'
    baseline = QueueModel.load(baseline_path)
    report = Path('reports/research')/(args.output.name+'.json')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'Causal_sequence_Queue_temporal_graph_control', 'panel': panel,
        'matched_study': args.matched_study.as_posix(), 'matched_study_sha256': sha256_file(args.matched_study),
        'baseline_sha256': sha256_file(baseline_path), 'seed': seed, 'dates': previous['dates'],
        'architectures': ['temporal', 'directed_graph'], 'hidden': 32, 'epochs': 20, 'steps_per_epoch': 80,
        'batch_size': 4, 'optimizer': 'AdamW lr=.001 weight_decay=.0001 clip_norm=1',
        'loss': 'Equal-event BCE positive weight 4 + .1 speed MAE; onset classification only at T+30',
        'sampling': 'Uniform onset/ongoing fit events; exactly matched prior event sampler and seeds',
        'inputs': 'Thirteen legal past/current slots, availability, occupancy, day phase, public condition, public FD geometry',
        'tree_comparison_limit': 'The LGB uses engineered summaries and no occupancy or provided-condition feature. Both neural arms use identical richer history inputs; only their graph messages differ.',
        'checkpoint': 'Lowest calibration domain-condition-macro event Brier every five epochs',
        'mixture_grid': [0., .25, .5, .75, 1.],
        'decision_selection': 'Existing calibration-only conditional threshold/nonempty tuner with public onset timing',
        'assessment': 'Load assessment labels only after all choices are sealed; preserve original winner',
        'label_source': 'Eligible noisy public train observations; unknown latent queue truth',
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
    release = Release('data/kaggle_public')
    net, topo = release.network(panel), release.topology(panel)
    paths = release.files(panel, 'train')
    parts = {key: [p for p in paths if partition_date(release, p) in dates] for key, dates in previous['dates'].items()}
    origin_cache = {}
    def origin_reader(origin):
        key = str(origin)
        if key not in origin_cache:
            origin_cache[key] = released_origin_rows(release, panel, 'train', origin)
        return origin_cache[key]
    fixed_links = None
    def sample(part, sample_seed, quota, spacing, expected_meta):
        global fixed_links
        frames = list(raw_for(release, panel, parts[part]))
        data, meta = sample_events(frames, net, topo, panel, sample_seed, per_condition=quota,
            spacing_minutes=spacing, origin_reader=origin_reader, onset_last_step=True)
        if canonical_hash(meta) != canonical_hash(expected_meta):
            raise ValueError('Neural events differ from the registered LGB comparison')
        by_date = {str(frame.timestamp.iloc[0])[:10]: frame for frame in frames}
        events = []
        labels = []
        for chosen in sorted(meta['windows'], key=lambda row: row['origin']):
            origin = pd.Timestamp(chosen['origin'])
            raw = by_date[origin.date().isoformat()]
            times = sorted(raw.timestamp.unique())
            at = chosen['at']
            if pd.Timestamp(times[at]) != origin:
                raise ValueError('Matched event origin changed')
            links = raw.link_id.drop_duplicates().tolist()
            if fixed_links is None:
                fixed_links = links
            elif links != fixed_links:
                raise ValueError('Queue node order changed')
            history = pd.concat([raw[raw.timestamp.isin(times[at-12:at])], origin_reader(origin)], ignore_index=True)
            x = queue_history_input(history, links, net, origin, chosen['condition'])
            # Future observations are targets only, never passed to the network.
            quality = raw[raw.timestamp.isin(times[at+1:at+7])].copy()
            quality.loc[~eligible(quality), 'speed_kmh'] = np.nan
            truth = quality.pivot_table(index='timestamp', columns='link_id', values='speed_kmh', aggfunc='mean').reindex(
                index=times[at+1:at+7], columns=links).to_numpy(float)
            available = np.isfinite(truth)
            y = (truth <= net.v_cut.reindex(links).to_numpy(float)).astype(np.float32)
            events.append({'x': x, 'labels': y, 'truth_speed': np.nan_to_num(truth/25.).astype(np.float32),
                'available': available, 'ongoing': chosen['condition'] == 'queue_ongoing',
                'origin': chosen['origin'], 'condition': chosen['condition']})
            labels.append(y.ravel()[available.ravel()])
        if not np.array_equal(np.concatenate(labels), data.y):
            raise ValueError('Neural target order differs from LGB proxy rows')
        cache_path = args.output/(part+'_'+str(sample_seed)+'.npz')
        np.savez_compressed(cache_path, x=np.stack([e['x'] for e in events]),
            labels=np.stack([e['labels'] for e in events]), truth_speed=np.stack([e['truth_speed'] for e in events]),
            available=np.stack([e['available'] for e in events]), ongoing=np.array([e['ongoing'] for e in events]),
            origins=np.array([e['origin'] for e in events]))
        return data, events, meta, sha256_file(cache_path)
    fit, fit_events, fit_meta, fit_hash = sample('fit', seed, 500, 30, previous['samplers']['fit'])
    cal, cal_events, cal_meta, cal_hash = sample('selection', 411, 30, 360, previous['samplers']['selection'])
    graph = []
    for direction in ('incoming', 'outgoing'):
        matrix = np.zeros((len(fixed_links), len(fixed_links)), np.float32)
        for j, connected in enumerate(neighbors(topo, fixed_links, direction)):
            if connected:
                matrix[j, connected] = 1/len(connected)
        graph.append(torch.tensor(matrix, device=device))
    lane = net.lanes.reindex(fixed_links).to_numpy(float)
    geometry = np.stack([net.free_speed_kmh.reindex(fixed_links), net.capacity_vph.reindex(fixed_links)/lane,
        net.k_jam.reindex(fixed_links)/lane, net.v_cut.reindex(fixed_links)]).astype(np.float32)
    graph.append(torch.tensor(geometry, device=device))
    baseline_cal_p = baseline.probability(cal.x)
    baseline_cal = evaluate(baseline_cal_p, cal, baseline.thresholds, baseline.require_nonempty)
    if abs(baseline_cal['proxy_IoU']-previous['selection']['shared_threshold']['metrics']['proxy_IoU']) > 1e-12:
        raise ValueError('Preserved LGB calibration did not reproduce')
    result = {'registered': registered, 'source_snapshot': snapshot, 'dates': previous['dates'],
        'samplers': {'fit': fit_meta, 'selection': cal_meta}, 'cache_sha256': {'fit': fit_hash, 'selection': cal_hash},
        'baseline_calibration_reproduced': True, 'selection': {'shared_threshold': {'metrics': baseline_cal}},
        'training': {}, 'assessment': [], 'device': str(device), 'torch': torch.__version__}
    selected = 'shared_threshold'
    settings = {'shared_threshold': {'alpha': 0., 'thresholds': baseline.thresholds, 'require_nonempty': baseline.require_nonempty}}
    groups = {ongoing: [e for e in fit_events if e['ongoing'] == ongoing] for ongoing in (False, True)}
    if any(not events for events in groups.values()):
        raise ValueError('Both fit conditions required')
    for name, use_graph in (('temporal', False), ('directed_graph', True)):
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = CausalGraphQueue(hidden=32, use_graph=use_graph).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        history = []
        best = np.inf
        for epoch in range(20):
            model.train()
            losses = []
            for _ in range(80):
                batch = []
                for _ in range(4):
                    group = groups[bool(rng.integers(2))]
                    batch.append(group[rng.integers(len(group))])
                x = torch.from_numpy(np.stack([e['x'].transpose(2, 0, 1) for e in batch])).to(device)
                tensors = [torch.from_numpy(np.stack([e[k] for e in batch])).to(device) for k in ('labels', 'truth_speed', 'available')]
                ongoing = torch.tensor([e['ongoing'] for e in batch], device=device)
                optimizer.zero_grad(set_to_none=True)
                logits, speed = model(x, *graph)
                loss = queue_event_loss(logits, speed, *tensors, ongoing)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if (epoch+1) % 5 == 0:
                model.eval()
                p = predict(model, cal_events, graph, device)
                value = brier(p, cal)
                history.append({'epoch': epoch+1, 'training_loss': float(np.mean(losses)), 'calibration_domain_event_Brier': value})
                if value < best:
                    best = value
                    torch.save(model.state_dict(), args.output/(name+'.pt'))
                print(name, epoch+1, 'cal domain Brier', value, flush=True)
        model.load_state_dict(torch.load(args.output/(name+'.pt'), map_location=device, weights_only=True))
        model.eval()
        p = predict(model, cal_events, graph, device)
        choices = []
        for alpha in (0., .25, .5, .75, 1.):
            mixed = (1-alpha)*baseline_cal_p+alpha*p
            tuner = ProbabilityCalibration(mixed)
            tuning = tuner.tune(cal)
            choices.append({'alpha': alpha, 'thresholds': tuner.thresholds, 'require_nonempty': tuner.require_nonempty,
                'metrics': evaluate(mixed, cal, tuner.thresholds, tuner.require_nonempty), 'tuning_selected': tuning['selected']})
        chosen = max(choices, key=lambda row: (row['metrics']['proxy_IoU'], -row['alpha']))
        settings[name] = {k: chosen[k] for k in ('alpha', 'thresholds', 'require_nonempty')}
        result['selection'][name] = {'metrics': chosen['metrics'], 'ensemble_grid': choices, 'settings': settings[name]}
        result['training'][name] = {'history': history, 'parameters': sum(p.numel() for p in model.parameters())}
        if chosen['metrics']['proxy_IoU'] > result['selection'][selected]['metrics']['proxy_IoU']:
            selected = name
        del optimizer, model
    result['selected'] = selected
    (args.output/'selection_sealed.json').write_text(json.dumps({'selected': selected, 'settings': settings,
        'checkpoint_sha256': {n: sha256_file(args.output/(n+'.pt')) for n in ('temporal', 'directed_graph')}}, indent=2), encoding='utf-8')
    for old_assessment in previous['assessment']:
        assessment_seed = old_assessment['seed']
        data, events, meta, digest = sample('assessment', assessment_seed, 30, 360, old_assessment['sampler'])
        baseline_p = baseline.probability(data.x)
        metrics = {'shared_threshold': evaluate(baseline_p, data, baseline.thresholds, baseline.require_nonempty)}
        old_windows = old_assessment['models']['shared_threshold']['windows']
        if metrics['shared_threshold']['windows'] != old_windows:
            raise ValueError('Assessment control windows did not reproduce')
        for name, use_graph in (('temporal', False), ('directed_graph', True)):
            model = CausalGraphQueue(hidden=32, use_graph=use_graph).to(device)
            model.load_state_dict(torch.load(args.output/(name+'.pt'), map_location=device, weights_only=True))
            model.eval()
            choice = settings[name]
            p = (1-choice['alpha'])*baseline_p+choice['alpha']*predict(model, events, graph, device)
            metrics[name] = evaluate(p, data, choice['thresholds'], choice['require_nonempty'])
            del model
        result['assessment'].append({'seed': assessment_seed, 'sampler': meta, 'cache_sha256': digest, 'models': metrics})
    result['selected_comparison'] = comparison(result, 'shared_threshold', selected)
    result['comparisons'] = {name: comparison(result, 'shared_threshold', name) for name in ('temporal', 'directed_graph')}
    result.update(seconds=time.perf_counter()-start, source=evidence(gpu_used=device.type == 'cuda'),
        gpu_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
        formal_Queue=None, formal_Total=None, uploaded=False,
        limitations='Paired noisy-public-observation events, not latent Queue truth. Both neural arms have the same legal richer histories; their comparison with engineered LGB does not isolate architecture alone. Assessment development dates were already inspected. No promotion without cross-season/direction/fit-seed evidence.')
    (args.output/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('selected', selected, result['selected_comparison'], result['seconds'], flush=True)
