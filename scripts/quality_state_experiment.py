"""Matched quality-input/density-loss State experiment with sealed choices.

Four models share initialization of every original parameter and patch RNG.
Calibration selects State and a constrained density challenger independently;
assessment labels are loaded only after both choices are sealed.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.adaptive_graph_state import AdaptiveGraphState, AdaptiveStateModel, patch_score_loss, reconstruction_input
from trafficflow.quality_state import quality_reconstruction_input, expand_shared_state
from trafficflow.graph_state import density_change_loss
from trafficflow.state import StateModel, StateFeatures
from trafficflow.data import Release, VALUES
from trafficflow.experiments import partition_date, day_template
from trafficflow.research_eval import register_protocol, state_from_sufficient_stats, paired_block_interval
from trafficflow.runtime import capture_source_snapshot, sha256_file, evidence
from adaptive_graph_experiment import predict
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats


ARCHITECTURES = [('attention12', 12, 0.), ('attention12_density', 12, .01),
                 ('attention18_quality', 18, 0.), ('attention18_quality_density', 18, .01)]


def versioned(days, channels):
    return [{**d, 'x': d['x_control']} if channels == 12 else d for d in days]


def from_predictions(days, predictions, alpha, factor):
    scores, increments = [], []
    for day, neural in zip(days, predictions):
        mixture = (1-alpha)*day['comparator']+alpha*neural
        scores.append(stats(day, mixture))
        increments.append(increment_stats(day, mixture, factor))
    return {**state_from_sufficient_stats(scores), 'days': scores,
        'density_target_pair_relative_L1': sum(i['error_L1'] for i in increments)/sum(i['truth_increment_L1'] for i in increments)}


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--inputs', choices=['both','strict12','quality18'], default='both')
    parser.add_argument('--coordinates', choices=['vq','density'], default='vq')
    parser.add_argument('--control-study', type=Path, help='Immutable initialization and strict12 calibration reference')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite quality experiment evidence')
    info = json.loads((args.cache/'prepared.json').read_text())
    architectures = [r for r in ARCHITECTURES if args.inputs=='both' or r[1]==(12 if args.inputs=='strict12' else 18)]
    control = json.loads((args.control_study/'experiment.json').read_text()) if args.control_study else None
    if control and (control['prepared']['baseline_sha256']!=info['baseline_sha256'] or control['registered']['protocol']['seed']!=args.seed):
        raise ValueError('Coordinate comparison requires the same strong baseline and fit seed')
    baseline_path = Path(info['baseline_file'])
    if sha256_file(baseline_path) != info['baseline_sha256']:
        raise ValueError('Strong baseline changed after preparation')
    report = Path('reports/research')/(args.output.name+'.json')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'Quality18_density_matched_State_factorial', 'cache': args.cache.as_posix(),
        'prepared_sha256': sha256_file(args.cache/'prepared.json'), 'baseline_file': baseline_path.as_posix(),
        'baseline_sha256': info['baseline_sha256'], 'panel': info['panel'], 'fold': info['fold'],
        'seed': args.seed, 'architectures': architectures, 'hidden': 48, 'heads': 2,
        'output_coordinates': args.coordinates,
        'control_study_sha256': sha256_file(args.control_study/'experiment.json') if args.control_study else None,
        'epochs': 24, 'steps_per_epoch': 120, 'batch_size': 4, 'patch_steps': 64,
        'optimizer': 'AdamW .001, weight_decay 1e-4, clip_norm 1',
        'initialization': 'Identical strict12 shared tensors; six additional Conv input weights exactly zero',
        'labels': 'Same full-label LGB fit dates; neural patches repeat targets and regimes sampled uniformly',
        'primary_selection': 'Per-model best pure calibration State checkpoint, then cal alpha grid; matches original A49 procedure',
        'joint_selection': 'Lowest calibration anchored density L1 across saved checkpoint/alpha pairs with State >= full-LGB cal and >= max(current best primary State, optional strict12 control primary State)-.001',
        'blend_grid': [0., .25, .5, .75, 1.],
        'physical_scope': 'Supervised public observation increments and strict12 visible anchors only; extra low-quality values are inputs, never truth anchors',
        'assessment_scope': 'Previously inspected public-train development dates, not unopened final',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    started = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    baseline = StateModel.load(baseline_path)
    release = Release('data/kaggle_public')
    net, topology = release.network(info['panel']), release.topology(info['panel'])
    files = {partition_date(release, p): p for p in release.files(info['panel'], 'train')}
    template = release.template(1, info['panel'], 'train')
    def load(part):
        days = []
        for path in sorted((args.cache/part).glob('*.npz')):
            record = next(r for r in info['records'] if r['part'] == part and r['date'] == path.stem)
            if sha256_file(path) != record['extended_cache_sha256']:
                raise ValueError('Extended input cache changed')
            day = dict(np.load(path, allow_pickle=False))
            if part != 'fit':
                f = StateFeatures(release.read(files[path.stem]), net, topology, baseline.profile, version=baseline.feature_version)
                strict = reconstruction_input(f)
                if not np.array_equal(strict, day['x_control']) or not np.array_equal(quality_reconstruction_input(f, strict), day['x']):
                    raise ValueError('Deployable input reconstruction differs')
                target = day_template(release, info['panel'], 'train', f.frame, template)
                if not np.array_equal(f.lookup(target), day['target_indices']):
                    raise ValueError('Target lookup differs')
            days.append(day)
        return days
    fit, cal = load('fit'), load('selection')
    if sha256_file(args.cache/'graph.npz') != info['graph_sha256']:
        raise ValueError('Public graph geometry changed after preparation')
    graph_np = dict(np.load(args.cache/'graph.npz'))
    graph = [torch.tensor(graph_np[k], device=device) for k in ['incoming', 'outgoing', 'fd']]
    factor = (net.lanes*net.length_km).reindex(fit[0]['links']).to_numpy(np.float32)
    factor_tensor = torch.tensor(factor, device=device)
    zeros = [np.zeros_like(d['comparator']) for d in cal]
    reference = from_predictions(cal, zeros, 0., factor)
    torch.manual_seed(args.seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)
    common = torch.load(args.control_study/'shared_initialization.pt', map_location='cpu', weights_only=True) if control else AdaptiveGraphState().state_dict()
    torch.save(common, args.output/'shared_initialization.pt')
    result = {'registered': registered, 'source_snapshot': snapshot, 'prepared': info,
        'baseline_selection': reference, 'training': {}, 'calibration_choices': [], 'primary_per_model': {},
        'shared_initialization_fingerprint': state_fingerprint(common), 'formal_State': None, 'formal_Physics': None, 'formal_Total': None}
    for name, channels, density_weight in architectures:
        torch.manual_seed(args.seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)
        rng = np.random.default_rng(args.seed)
        days, validation = versioned(fit, channels), versioned(cal, channels)
        regimes = {r: [d for d in days if str(d['regime']) == r] for r in ['R1', 'R2', 'R3']}
        model = AdaptiveGraphState(channels=channels,output_coordinates=args.coordinates).to(device)
        model.load_state_dict(expand_shared_state(common, channels))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        best_pure, primary = -np.inf, None
        sampling = hashlib.sha256()
        history = []
        tic = time.perf_counter()
        for epoch in range(24):
            model.train()
            losses = []
            for _ in range(120):
                batch = []
                for _ in range(4):
                    regime = ['R1', 'R2', 'R3'][rng.integers(3)]
                    group = regimes[regime]
                    day = group[rng.integers(len(group))]
                    at = int(rng.integers(0, day['x'].shape[0]-64+1))
                    batch.append((day, at))
                    sampling.update(f"{regime}|{day['times'][0]}|{at};".encode())
                x = torch.from_numpy(np.stack([d['x'][at:at+64].transpose(2, 0, 1) for d, at in batch])).to(device)
                y = torch.from_numpy(np.stack([d['y'][at:at+64].transpose(2, 0, 1) for d, at in batch])).to(device)
                mask = torch.from_numpy(np.stack([d['mask'][at:at+64] for d, at in batch])).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, *graph)
                loss = patch_score_loss(prediction, y, mask)
                if density_weight:
                    loss = loss+density_weight*density_change_loss(prediction, y, mask, x, factor_tensor)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if (epoch+1) % 4:
                continue
            model.eval()
            checkpoint = args.output/f'{name}_epoch{epoch+1:02d}.pt'
            torch.save(model.state_dict(), checkpoint)
            predictions = [predict(model, d, graph, device) for d in validation]
            choices = [{'name': name, 'channels': channels, 'epoch': epoch+1, 'alpha': a,
                'checkpoint_sha256': sha256_file(checkpoint), 'metrics': from_predictions(validation, predictions, a, factor)}
                for a in [0., .25, .5, .75, 1.]]
            pure = choices[-1]['metrics']
            result['calibration_choices'].extend(choices)
            if pure['S_state'] > best_pure:
                best_pure = pure['S_state']
                primary = max(choices, key=lambda r: (r['metrics']['S_state'], -r['alpha']))
            history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)),
                'pure_calibration_State': pure['S_state'], 'pure_calibration_density_L1': pure['density_target_pair_relative_L1']})
            print(name, epoch+1, pure['S_state'], pure['density_target_pair_relative_L1'], flush=True)
        result['primary_per_model'][name] = primary
        result['training'][name] = {'history': history, 'seconds': time.perf_counter()-tic,
            'sample_sequence_sha256': sampling.hexdigest(), 'parameters': sum(p.numel() for p in model.parameters())}
        (args.output/'progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if len({r['sample_sequence_sha256'] for r in result['training'].values()}) != 1:
        raise ValueError('Factorial models saw different sampled patches')
    primary = max(result['primary_per_model'].values(), key=lambda r: r['metrics']['S_state'])
    best_reference = max(primary['metrics']['S_state'], max(r['metrics']['S_state'] for r in control['primary_per_model'].values() if r['channels']==12) if control else -np.inf)
    result['joint_calibration_State_reference'] = best_reference
    admissible = [r for r in result['calibration_choices'] if r['metrics']['S_state'] >= reference['S_state'] and r['metrics']['S_state'] >= best_reference-.001]
    joint = min(admissible, key=lambda r: (r['metrics']['density_target_pair_relative_L1'], -r['metrics']['S_state'], r['alpha'])) if admissible else None
    sealed = {'primary': primary, **({'joint':joint} if joint is not None else {})}
    result['selected'] = sealed
    (args.output/'selection_sealed.json').write_text(json.dumps(sealed, indent=2), encoding='utf-8')
    assessment = load('assessment')
    result['baseline_assessment'] = from_predictions(assessment, [np.zeros_like(d['comparator']) for d in assessment], 0., factor)
    result['assessment'] = {}
    # Each model keeps its own original cal choice. Joint is an additional sealed challenger, never an assessment winner.
    chosen = {**result['primary_per_model'], **({'joint_challenger': joint} if joint is not None else {})}
    for role, choice in chosen.items():
        model = AdaptiveGraphState(channels=choice['channels'],output_coordinates=args.coordinates).to(device)
        path = args.output/f"{choice['name']}_epoch{choice['epoch']:02d}.pt"
        if sha256_file(path) != choice['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True)); model.eval()
        days = versioned(assessment, choice['channels'])
        metrics = from_predictions(days, [predict(model, d, graph, device) for d in days], choice['alpha'], factor)
        metrics['paired_daily_interval'] = paired_block_interval(
            [d['daily_State'] for d in result['baseline_assessment']['days']], [d['daily_State'] for d in metrics['days']])
        result['assessment'][role] = metrics
    for role, choice in sealed.items():
        path = args.output/f"{choice['name']}_epoch{choice['epoch']:02d}.pt"
        state = torch.load(path, map_location='cpu', weights_only=True)
        deployed = AdaptiveStateModel(baseline, state, topology, choice['alpha'],
            input_schema='quality18' if choice['channels'] == 18 else 'strict12',output_coordinates=args.coordinates)
        deployed.save(args.output/f'State_{role}.pkl')
    result.update(seconds=time.perf_counter()-started, source=evidence(gpu_used=device.type == 'cuda'),
        peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
        deployable_model_sha256={r: sha256_file(args.output/f'State_{r}.pkl') for r in sealed},
        limitations='One direction/fold/seed and previously viewed development assessment. Added information and minor parameter increase, not an architecture-only comparison. Joint selection uses an incomplete observed-increment proxy, not real Physics. No final seeds or upload.')
    (args.output/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('primary', {k: v for k, v in primary.items() if k != 'metrics'}, 'joint', {k: v for k, v in joint.items() if k != 'metrics'} if joint is not None else None, flush=True)
    print({r: (m['S_state'], m['density_target_pair_relative_L1']) for r, m in result['assessment'].items()}, result['seconds'], flush=True)


if __name__ == '__main__':
    main()
