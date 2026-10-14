"""Fixed real-train State networks under own contiguous CTM missingness.

These patterns do not recreate official R3 rules or any hidden scenario.
Pattern selection uses our simulation geometry/time, never latent state values.
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
from trafficflow.digital_twin import Scenario, simulate
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot, sha256_file, evidence
from ctm_adaptive_stress import CASES, metrics


PATTERNS = ['uniform_half', 'central_half_corridor', 'incident_rectangle_120min', 'network_missing_45min']


def missing_pattern(shape, config, seed, kind):
    times, nodes = shape
    minute = (np.arange(times)*5) % 1440
    pattern = np.zeros(shape, bool)
    if kind == 'uniform_half':
        pattern = np.random.default_rng(seed+20003).random(shape) < .5
    elif kind == 'central_half_corridor':
        pattern[:, nodes//4:3*nodes//4] = True
    elif kind == 'incident_rectangle_120min':
        start = config['incident_start_hour']*60
        slots = (minute >= start) & (minute < start+120)
        first = min(max(0, config['incident_cell']-nodes//4), nodes-nodes//2)
        pattern[slots, first:first+nodes//2] = True
    elif kind == 'network_missing_45min':
        start = config['incident_start_hour']*60+30
        pattern[(minute >= start) & (minute < start+45)] = True
    else:
        raise ValueError('Unknown synthetic missing pattern')
    return pattern


def evaluate_sim(sim, models, base, device, historical_prior=None, base_policy='temporal_first'):
    methods = ['interpolation', *models]
    predictions = {m: np.empty((*sim['target'].shape, 2)) for m in methods}
    digest = hashlib.sha256()
    for left, right in [(0, 288), (288, len(sim['target']))]:
        x, interp, incoming, outgoing, fd = synthetic_state_input(
            sim['masked_speed'][left:right], sim['masked_flow'][left:right], sim['eligible'][left:right],
            lanes=base.lanes, free_speed=base.free_speed, capacity=base.capacity, jam_density=base.jam_density,
            historical_prior=None if historical_prior is None else historical_prior[left:right], base_policy=base_policy)
        digest.update(x.tobytes())
        predictions['interpolation'][left:right] = interp
        graph = [torch.tensor(a, device=device) for a in (incoming, outgoing, fd)]
        inp = torch.tensor(x.transpose(2, 0, 1)[None].copy(), device=device)
        for method, model in models.items():
            with torch.no_grad():
                p = model(inp, *graph).cpu().numpy()[0].transpose(1, 2, 0)
            p[..., 0] = np.clip(p[..., 0]*25, 0, 1.05*base.free_speed)
            p[..., 1] = np.clip(p[..., 1]*600, 0, 1.05*base.capacity/base.lanes)
            predictions[method][left:right] = p
    for method in models:
        predictions['half_interpolation_'+method] = .5*(predictions[method]+predictions['interpolation'])
    result = {m: metrics(sim, p[..., 0], p[..., 1]*base.lanes) for m, p in predictions.items()}
    result['observation_oracle'] = metrics(sim, sim['observation_speed'], sim['observation_flow'])
    result['latent_oracle'] = metrics(sim, sim['latent_speed'], sim['latent_flow'])
    return result, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--base-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path('artifacts/research')/args.output.stem
    if args.output.exists() or directory.exists():
        raise FileExistsError('Never overwrite contiguous-stress evidence')
    source, previous = [json.loads(p.read_text()) for p in [args.source/'experiment.json', args.base_study]]
    names = ['adaptive_attention', 'adaptive_attention_density']
    hashes = {n: sha256_file(args.source/(n+'.pt')) for n in names}
    if any(hashes[n] != source['model_sha256'][n+'.pt'] for n in names):
        raise ValueError('Frozen real-train checkpoint changed')
    setting = previous['registered']['protocol']['base'].copy()
    setting['mask_rate'] = 0.
    base = Scenario(**setting)
    seeds = previous['registered']['protocol']['seeds']
    registered = register_protocol(args.output.with_suffix('.registered.json'), {
        'id': 'Independent_CTM_contiguous_missing_State_control', 'source_model_sha256': hashes,
        'base_study_sha256': sha256_file(args.base_study), 'base': setting,
        'cases': CASES, 'assessment_seeds': seeds, 'patterns': PATTERNS,
        'pattern_scope': 'Own fixed clock/geometry masks only; not official R3 or hidden-instance reconstruction',
        'target': 'Eligible cells inside pattern; all pattern numeric observations withheld before input construction',
        'selection': 'None. Same A33 source-cal checkpoints and fixed .5 interpolation blends as B07.',
        'quality_profile_occupancy': 'Same filtered inputs, geometry-only profile and unavailable occupancy as B07',
        'final_seeds_generated': False, 'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
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
    for seed in seeds:
        for case, params in CASES.items():
            original = simulate(replace(base, seed=seed, **params))
            if original['target'].any():
                raise ValueError('Original simulation must have no artificial targets')
            for pattern_name in PATTERNS:
                pattern = missing_pattern(original['target'].shape, original['scenario'], seed, pattern_name)
                sim = dict(original)
                sim['target'] = pattern & original['eligible']
                sim['masked_speed'] = np.where(pattern, np.nan, original['masked_speed'])
                sim['masked_flow'] = np.where(pattern, np.nan, original['masked_flow'])
                if not sim['target'].any() or np.isfinite(sim['masked_speed'][sim['target']]).any() or np.isfinite(sim['masked_flow'][sim['target']]).any():
                    raise ValueError('Empty target or unwithheld target observation')
                candidates, digest = evaluate_sim(sim, models, base, device)
                rows.append({'seed': seed, 'scenario': case, 'pattern': pattern_name,
                    'configuration': sim['scenario'], 'input_sha256': digest, 'n_targets': int(sim['target'].sum()),
                    'target_share_of_eligible': float(sim['target'].sum()/sim['eligible'].sum()),
                    'conservation_error': sim['max_conservation_error_vehicles'], 'candidates': candidates})
            (directory/'progress.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(seed, case, 'complete', flush=True)
    summary = {pattern: {method: float(np.mean([r['candidates'][method]['simulated_observation_State'] for r in rows if r['pattern'] == pattern]))
                for method in rows[0]['candidates']} for pattern in PATTERNS}
    result = {'registered': registered, 'source_snapshot': snapshot, 'rows': rows, 'summary_by_pattern': summary,
        'seconds': time.perf_counter()-start, 'source': evidence(gpu_used=device.type == 'cuda'),
        'peak_cuda_allocated_MiB': torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
        'formal_Total': None, 'uploaded': False,
        'limitations': 'Three seed clusters, masks share latent/noisy truth. Pattern scores use their own differing target populations, not a paired mask-effect gain. Within-pattern model contrasts only. No simulator assessment selection or production promotion.'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(summary, result['seconds'], flush=True)


if __name__ == '__main__':
    main()
