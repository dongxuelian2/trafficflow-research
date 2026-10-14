"""Fit fixed development-selected neural State candidates on public train.

All production training dates are used. There is no production holdout score,
checkpoint selection on future labels, or formal upload in this script.
"""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.adaptive_graph_state import AdaptiveGraphState, AdaptiveStateModel, patch_score_loss
from trafficflow.graph_state import density_change_loss
from trafficflow.data import Release
from trafficflow.state import StateModel
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import register_protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a production candidate')
    info = json.loads((args.prepared/'prepared.json').read_text())
    if info['panel'] != 'D12_I5_N':
        raise ValueError('Current production protocol is supported only for the researched I5 north direction')
    studies = {}
    for name in ('A33_adaptive_attention_strong_autumn', 'A34_adaptive_attention_strong_winter',
                 'A35_adaptive_attention_autumn_seed20261029'):
        path = Path('reports/research')/(name+'.json')
        row = json.loads(path.read_text())
        if row['selected']['alpha'] != .75 or row['paired_daily_interval']['ci95'][0] <= 0:
            raise ValueError('Development evidence no longer supports the fixed production protocol')
        studies[name] = {'sha256': sha256_file(path), 'selected': row['selected'],
            'State_gain': row['selected_assessment']['S_state']-row['baseline_assessment']['S_state']}
    baseline_path = args.prepared/'lightgbm_full_public_train.pkl'
    if sha256_file(baseline_path) != info['large_State_model_sha256']:
        raise ValueError('Production full-label baseline changed')
    if sha256_file(args.prepared/'graph.npz') != info['graph_sha256']:
        raise ValueError('Production graph changed')
    cache_hashes = {}
    for row in info['sources']:
        path = args.prepared/'fit'/(row['date']+'.npz')
        if sha256_file(path) != row['cache_sha256']:
            raise ValueError('Production public-train cache changed')
        cache_hashes[path.as_posix()] = row['cache_sha256']
    report = Path('reports/research')/(args.output.name+'.json')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'Production_adaptive_State_public_train', 'panel': info['panel'],
        'prepared_sha256': sha256_file(args.prepared/'prepared.json'), 'development_studies': studies,
        'architectures': {'adaptive_attention': 0., 'adaptive_attention_density': .01},
        'seed': args.seed, 'hidden': 48, 'heads': 2, 'epochs': 24, 'steps_per_epoch': 120,
        'batch_size': 4, 'patch_steps': 64, 'alpha': .75,
        'optimizer': 'AdamW lr=.001 weight_decay=.0001 clip_norm=1',
        'loss': 'Uniform-regime equal-patch channel-scaled RMSE; optional .01 supervised density increments',
        'checkpoint_policy': 'Fixed epoch 24 for both cases; no production calibration or assessment selection',
        'public_train_scope': 'All 228 dates after the frozen 45-day profile prefix, including earlier development assessments',
        'limitations': 'Two development-selected hypotheses for one direction. No production holdout or global ten-direction gate is claimed.',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None, 'uploaded': False})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    days = [dict(np.load(p, allow_pickle=False)) for p in sorted((args.prepared/'fit').glob('*.npz'))]
    if len(days) != info['n_grid_fit_dates']:
        raise ValueError('Public-train day count changed')
    regimes = {r: [d for d in days if str(d['regime']) == r] for r in ('R1', 'R2', 'R3')}
    if any(not group for group in regimes.values()):
        raise ValueError('Every fit regime is required')
    graph_np = dict(np.load(args.prepared/'graph.npz'))
    graph = [torch.tensor(graph_np[k], device=device) for k in ('incoming', 'outgoing', 'fd')]
    release = Release('data/kaggle_public')
    net, topo = release.network(info['panel']), release.topology(info['panel'])
    factor = torch.tensor((net.lanes*net.length_km).reindex(days[0]['links']).to_numpy(np.float32), device=device)
    baseline = StateModel.load(baseline_path)
    result = {'registered': registered, 'source_snapshot': snapshot, 'cache_sha256': cache_hashes,
        'device': str(device), 'torch': torch.__version__, 'models': {}, 'source_at_start': evidence(gpu_used=device.type == 'cuda')}
    for name, density_weight in registered['protocol']['architectures'].items():
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        rng = np.random.default_rng(args.seed)
        model = AdaptiveGraphState(hidden=48, heads=2, attention=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        history = []
        tic = time.perf_counter()
        for epoch in range(24):
            model.train()
            losses = []
            for _ in range(120):
                batch = []
                for _ in range(4):
                    group = regimes[('R1', 'R2', 'R3')[rng.integers(3)]]
                    day = group[rng.integers(len(group))]
                    at = rng.integers(0, day['x'].shape[0]-64+1)
                    batch.append((day, at))
                x = torch.from_numpy(np.stack([d['x'][at:at+64].transpose(2, 0, 1) for d, at in batch])).to(device)
                y = torch.from_numpy(np.stack([d['y'][at:at+64].transpose(2, 0, 1) for d, at in batch])).to(device)
                mask = torch.from_numpy(np.stack([d['mask'][at:at+64] for d, at in batch])).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, *graph)
                loss = patch_score_loss(prediction, y, mask)
                if density_weight:
                    loss = loss+density_weight*density_change_loss(prediction, y, mask, x, factor)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite production training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append({'epoch': epoch+1, 'training_loss_only': float(np.mean(losses))})
            if (epoch+1) % 4 == 0:
                print(name, epoch+1, 'training loss', history[-1]['training_loss_only'], flush=True)
                (args.output/(name+'.progress.json')).write_text(json.dumps(history, indent=2), encoding='utf-8')
        model.eval()
        checkpoint = args.output/(name+'.pt')
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
        deployed = AdaptiveStateModel(baseline, model.state_dict(), topo, .75, hidden=48, attention=True)
        deployed_path = args.output/(name+'.pkl')
        deployed.save(deployed_path)
        result['models'][name] = {'density_weight': density_weight, 'fixed_epoch': 24, 'alpha': .75,
            'training_history': history, 'seconds': time.perf_counter()-tic,
            'checkpoint_sha256': sha256_file(checkpoint), 'deployable_model_sha256': sha256_file(deployed_path)}
        (args.output/'progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        del optimizer, model, deployed
    result.update(seconds=time.perf_counter()-start,
        gpu_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
        source_at_end=evidence(gpu_used=device.type == 'cuda'), formal_State=None, formal_Physics=None,
        formal_Total=None, production_holdout_State=None, uploaded=False)
    (args.output/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(args.output, result['seconds'], 'No production holdout or formal score', flush=True)
