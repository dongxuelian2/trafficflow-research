"""Matched 2x2 historical-prior / interpolation-policy CTM State diagnostic.

Profile is fit from independent development simulations' visible eligible noisy
observations. Fixed real-train weights are never refit on simulated labels.
An additional one-hour incident shift probes the daily-history assumption.
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
from trafficflow.digital_twin import Scenario, simulate
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot, sha256_file, evidence
from ctm_adaptive_stress import CASES, metrics
from ctm_contiguous_stress import PATTERNS, missing_pattern, evaluate_sim


def fit_profile(base, seeds):
    total = np.zeros((288, base.cells, 2))
    count = np.zeros_like(total, dtype=np.int64)
    records = []
    for seed in seeds:
        for case, config in CASES.items():
            sim = simulate(replace(base, seed=seed, **config))
            if sim['target'].any():
                raise ValueError('Development profile uses only unwithheld synthetic observations')
            raw = np.stack([sim['masked_speed'], sim['masked_flow']/base.lanes], axis=-1)
            valid = sim['eligible'][..., None] & np.isfinite(raw)
            slots = np.arange(len(raw)) % 288
            np.add.at(total, slots, np.where(valid, raw, 0.))
            np.add.at(count, slots, valid.astype(np.int64))
            records.append({'seed': seed, 'case': case, 'eligible_input_sha256': hashlib.sha256(np.where(valid, raw, np.nan).tobytes()).hexdigest(),
                'n_eligible_channel_observations': int(valid.sum())})
    geometry = np.broadcast_to([base.free_speed, .4*base.capacity/base.lanes], total.shape)
    profile = np.divide(total, count, out=geometry.copy(), where=count > 0)
    return profile, count, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--base-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path('artifacts/research')/args.output.stem
    if args.output.exists() or directory.exists():
        raise FileExistsError('Never overwrite history-ablation evidence')
    source, previous = [json.loads(p.read_text()) for p in [args.source/'experiment.json', args.base_study]]
    base_config = previous['registered']['protocol']['base'].copy()
    base_config['mask_rate'] = 0.
    base = Scenario(**base_config)
    protocol = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())['protocol']
    dev, assessment = protocol['simulator_development_seeds'], protocol['simulator_assessment_seeds']
    if set(dev) & set(assessment):
        raise ValueError('Historical profile and assessment seeds overlap')
    names = ['adaptive_attention', 'adaptive_attention_density']
    hashes = {n: sha256_file(args.source/(n+'.pt')) for n in names}
    if any(hashes[n] != source['model_sha256'][n+'.pt'] for n in names):
        raise ValueError('Frozen real-train weight changed')
    modes = [(p, policy) for p in ['geometry', 'development_history'] for policy in ['temporal_first', 'deployed_weighted']]
    registered = register_protocol(args.output.with_suffix('.registered.json'), {
        'id': 'Independent_CTM_history_and_input_policy_factorial', 'source_model_sha256': hashes,
        'base_study_sha256': sha256_file(args.base_study), 'base': base_config,
        'profile_seeds': dev, 'assessment_seeds': assessment, 'cases': CASES, 'patterns': PATTERNS,
        'adapter_modes': modes, 'incident_hour_shifts': [0., 1.],
        'history': 'One global 288-slot/cell noisy-observation mean across all six development cases and both days; eligible visible observations only; no case-specific profile',
        'weighted_base': 'Matches StateFeatures temporal age weight 1/(1+age/10), spatial weight .35, prior fallback',
        'source_weight_training': 'Original public-train A33 only; no synthetic fine-tuning',
        'profile_limit': 'Same fixed generator clock in dev histories can be overly informative; +1h incident assessment shift reported separately',
        'selection': 'No model/mode/shift selection; all preregistered factors retained',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None, 'final_seeds_generated': False})
    directory.mkdir(parents=True)
    snapshot = capture_source_snapshot(directory/'source_snapshot.zip')
    start = time.perf_counter()
    profile, count, history = fit_profile(base, dev)
    np.savez_compressed(directory/'historical_profile.npz', profile=profile, count=count)
    (directory/'history_sources.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
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
    for seed in assessment:
        for case, config in CASES.items():
            for shift in [0., 1.]:
                sim_config = {**config, 'incident_start_hour': base.incident_start_hour+shift}
                original = simulate(replace(base, seed=seed, **sim_config))
                historical = profile[np.arange(len(original['target'])) % 288]
                for pattern_name in PATTERNS:
                    pattern = missing_pattern(original['target'].shape, original['scenario'], seed, pattern_name)
                    sim = dict(original)
                    sim['target'] = pattern & original['eligible']
                    sim['masked_speed'] = np.where(pattern, np.nan, original['masked_speed'])
                    sim['masked_flow'] = np.where(pattern, np.nan, original['masked_flow'])
                    adapters = {}
                    for prior_name, policy in modes:
                        candidates, digest = evaluate_sim(sim, models, base, device,
                            historical if prior_name == 'development_history' else None, policy)
                        prior = historical if prior_name == 'development_history' else np.broadcast_to([base.free_speed, .4*base.capacity/base.lanes], historical.shape)
                        candidates['prior_only'] = metrics(sim, prior[..., 0], prior[..., 1]*base.lanes)
                        adapters[prior_name+'__'+policy] = {'input_sha256': digest, 'candidates': candidates}
                    rows.append({'seed': seed, 'scenario': case, 'pattern': pattern_name, 'incident_hour_shift': shift,
                        'configuration': sim['scenario'], 'n_targets': int(sim['target'].sum()), 'adapters': adapters})
            print(seed, case, 'complete', flush=True)
        (directory/'progress.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    summary = {str(shift): {pattern: {mode: {method: float(np.mean([r['adapters'][mode]['candidates'][method]['simulated_observation_State'] for r in rows
                if r['incident_hour_shift'] == shift and r['pattern'] == pattern])) for method in rows[0]['adapters'][mode]['candidates']}
                for mode in rows[0]['adapters']} for pattern in PATTERNS} for shift in [0., 1.]}
    result = {'registered': registered, 'source_snapshot': snapshot, 'historical_profile_sha256': sha256_file(directory/'historical_profile.npz'),
        'history_sources_sha256': sha256_file(directory/'history_sources.json'), 'history_input_records': history,
        'rows': rows, 'summary': summary, 'seconds': time.perf_counter()-start, 'source': evidence(gpu_used=device.type == 'cuda'),
        'peak_cuda_allocated_MiB': torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
        'formal_Total': None, 'uploaded': False,
        'limitations': 'Own fixed-clock history and same generator family, not organizer distribution. Only three assessment seed clusters. Factor comparisons share targets within each case; different patterns and clocks have different target populations. No assessment tuning or production promotion.'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('seconds', result['seconds'], 'profile_input_coverage_min', int(count.min()), flush=True)
    print({shift: {pattern: {mode: round(methods['interpolation'], 6) for mode, methods in modes.items()} for pattern, modes in patterns.items()}
           for shift, patterns in summary.items()}, flush=True)


if __name__ == '__main__':
    main()
