"""Frozen real-train State checkpoints on independent CTM assessment seeds.

No simulator fitting, blending selection, or final-seed generation. This tests
zero-shot shift, not formal competition Physics or a calibrated digital twin.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.adaptive_graph_state import AdaptiveGraphState
from trafficflow.ctm_state_adapter import synthetic_state_input
from trafficflow.data import Release
from trafficflow.digital_twin import Scenario, simulate, simulated_physics
from trafficflow.ctm_scenarios import CASES
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot, evidence, sha256_file


def metrics(sim, v, q):
    target, s = sim['target'], sim['scenario']
    rv = float(np.sqrt(np.mean((v[target]-sim['observation_speed'][target])**2)))
    rq = float(np.sqrt(np.mean(((q[target]-sim['observation_flow'][target])/s['lanes'])**2)))
    latent = {}
    drop = np.zeros_like(target)
    recovery = np.zeros_like(target)
    change = np.diff(sim['latent_speed'], axis=0)
    drop[1:] = change <= -20.
    recovery[1:] = (change >= 20.) & (sim['latent_speed'][:-1] <= .6*s['free_speed'])
    for name, mask in [('target', target), ('detector_failure', sim['failure']),
                       ('five_minute_drop_target', drop & target), ('recovery_target', recovery & target)]:
        n = int(mask.sum())
        latent[name] = {'n': n,
            'speed_rmse': float(np.sqrt(np.mean((v[mask]-sim['latent_speed'][mask])**2))) if n else None,
            'lane_flow_rmse': float(np.sqrt(np.mean(((q[mask]-sim['latent_flow'][mask])/s['lanes'])**2))) if n else None,
            'density_rmse': float(np.sqrt(np.mean((q[mask]/np.maximum(v[mask], 1.)-sim['density'][mask])**2))) if n else None}
    return {'simulated_observation_State': .54*max(0., 1-rv/25)+.46*max(0., 1-rq/600),
        'speed_rmse': rv, 'lane_flow_rmse': rq, 'latent_diagnostics': latent,
        'ctm_flux': simulated_physics(sim, v, q),
        'projected_observation_flux': simulated_physics(sim, v, q, 'observation_projected_synthetic_flux'),
        'formal_State': None, 'formal_Physics': None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path('artifacts/research')/args.output.stem
    if args.output.exists() or directory.exists():
        raise FileExistsError('Never overwrite stress results or source snapshots')
    source = json.loads((args.source/'experiment.json').read_text())
    names = ['adaptive_attention', 'adaptive_attention_density']
    hashes = {name: sha256_file(args.source/(name+'.pt')) for name in names}
    if any(hashes[n] != source['model_sha256'][n+'.pt'] for n in names):
        raise ValueError('Frozen source checkpoint changed')
    protocol = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())['protocol']
    release = Release('data/kaggle_public')
    panel = source['registered']['protocol']['panel']
    net = release.network(panel)
    sensors = release.read(release.files(panel, 'train', False)[0]).link_id.unique()
    fd = net.loc[sensors]
    lanes = float(fd.lanes.median())
    base = Scenario(seed=0, days=2, lanes=lanes, free_speed=float(fd.free_speed_kmh.median()),
        capacity=float((fd.capacity_vph/fd.lanes).median()*lanes), jam_density=float((fd.k_jam/fd.lanes).median()*lanes))
    registered = register_protocol(args.output.with_suffix('.registered.json'), {
        'id': 'Frozen_adaptive_State_independent_CTM_stress', 'source_panel': panel,
        'source_report_sha256': sha256_file(args.source/'experiment.json'), 'checkpoint_sha256': hashes,
        'seeds': protocol['simulator_assessment_seeds'], 'final_seeds_generated': False,
        'cases': CASES, 'base': base.__dict__, 'mask_rates': [.2, .3, .5],
        'quality_policy': 'All candidate numeric inputs filter quality-ineligible values; differs from B01',
        'profile': 'Geometry only vf and .4 lane capacity; no simulator-truth profile',
        'occupancy': 'Unavailable, zero value and availability. Never constructed from latent density.',
        'methods': ['interpolation', *names, *['half_interpolation_'+n for n in names], 'observation_oracle', 'latent_oracle'],
        'inference': 'First 288 and remaining 289 snapshots in separate daily grids, constant public FD and directed cell chain',
        'selection': 'None; all fixed source-cal-selected checkpoints and equal .5 blends evaluated without simulator retuning',
        'scope': 'Own Godunov simulator, not organizer generator; FD snapshot flow differs from interval CTM flux',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
    directory.mkdir(parents=True)
    snapshot = capture_source_snapshot(directory/'source_snapshot.zip')
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    models = {}
    for name in names:
        model = AdaptiveGraphState(hidden=source['registered']['protocol']['hidden'], attention=True).to(device)
        model.load_state_dict(torch.load(args.source/(name+'.pt'), map_location=device, weights_only=True))
        models[name] = model.eval()
    rows = []
    for seed in protocol['simulator_assessment_seeds']:
        for name, config in CASES.items():
            for rate in (.2, .3, .5):
                sim = simulate(replace(base, seed=seed, mask_rate=rate, **config))
                predictions = {n: np.empty((*sim['target'].shape, 2)) for n in ['interpolation', *names]}
                digest = hashlib.sha256()
                for left, right in [(0, 288), (288, len(sim['target']))]:
                    x, interp, incoming, outgoing, geometry = synthetic_state_input(
                        sim['masked_speed'][left:right], sim['masked_flow'][left:right], sim['eligible'][left:right],
                        lanes=base.lanes, free_speed=base.free_speed, capacity=base.capacity, jam_density=base.jam_density)
                    digest.update(x.tobytes())
                    predictions['interpolation'][left:right] = interp
                    tensors = [torch.tensor(a, device=device) for a in (incoming, outgoing, geometry)]
                    inp = torch.tensor(x.transpose(2, 0, 1)[None].copy(), device=device)
                    for method, model in models.items():
                        with torch.no_grad():
                            pred = model(inp, *tensors).cpu().numpy()[0].transpose(1, 2, 0)
                        pred[..., 0] = np.clip(pred[..., 0]*25, 0, 1.05*base.free_speed)
                        pred[..., 1] = np.clip(pred[..., 1]*600, 0, 1.05*base.capacity/base.lanes)
                        predictions[method][left:right] = pred
                for method in names:
                    predictions['half_interpolation_'+method] = .5*(predictions['interpolation']+predictions[method])
                candidate_metrics = {n: metrics(sim, p[..., 0], p[..., 1]*base.lanes) for n, p in predictions.items()}
                candidate_metrics['observation_oracle'] = metrics(sim, sim['observation_speed'], sim['observation_flow'])
                candidate_metrics['latent_oracle'] = metrics(sim, sim['latent_speed'], sim['latent_flow'])
                rows.append({'seed': seed, 'scenario': name, 'mask_rate': rate, 'configuration': sim['scenario'],
                    'input_sha256': digest.hexdigest(), 'n_targets': int(sim['target'].sum()),
                    'congestion_share': float(sim['queue_truth'].mean()), 'conservation_error': sim['max_conservation_error_vehicles'],
                    'candidates': candidate_metrics})
            print(seed, name, 'complete', flush=True)
    summary = {method: {'mean_simulated_observation_State': float(np.mean([r['candidates'][method]['simulated_observation_State'] for r in rows])),
        'ctm_flux_mean': float(np.mean([r['candidates'][method]['ctm_flux']['simulated_S_physics'] for r in rows])),
        'observation_projected_mean': float(np.mean([r['candidates'][method]['projected_observation_flux']['simulated_S_physics'] for r in rows]))}
        for method in rows[0]['candidates']}
    result = {'registered': registered, 'source_snapshot': snapshot, 'rows': rows, 'summary': summary,
        'seconds': time.perf_counter()-start, 'source': evidence(gpu_used=device.type == 'cuda'),
        'peak_cuda_allocated_MiB': torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
        'formal_Total': None, 'uploaded': False, 'promotion': 'No production model selected from simulator assessment'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(summary, result['seconds'], flush=True)


if __name__ == '__main__':
    main()
