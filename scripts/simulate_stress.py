"""Stress baselines on our own CTM truth, report both flux conventions."""
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.digital_twin import Scenario, simulate, simulated_physics
from trafficflow.state import temporal, kalman_residual
from trafficflow.data import Release
from trafficflow.runtime import evidence


def metrics(sim, v, q):
    target = sim['target']
    s = sim['scenario']
    speed_error = v[target] - sim['observation_speed'][target]
    flow_error = (q[target] - sim['observation_flow'][target]) / s['lanes']
    rmsev, rmseq = np.sqrt(np.mean(speed_error**2)), np.sqrt(np.mean(flow_error**2))
    return {'simulated_observation_State': float(.54*max(0,1-rmsev/25) + .46*max(0,1-rmseq/600)),
        'speed_rmse': float(rmsev), 'lane_flow_rmse': float(rmseq),
        'ctm_flux': simulated_physics(sim, v, q),
        'projected_observation_flux': simulated_physics(sim, v, q, 'observation_projected_synthetic_flux'),
        'formal_total': None}


def main():
    output = Path('reports/research/B01_CTM_stress.json')
    if output.exists():
        raise FileExistsError('Use a new stress experiment version')
    started = time.perf_counter()
    protocol = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())
    release = Release('data/kaggle_public')
    net = release.network('D12_I5_N')
    sensors = release.read(release.files('D12_I5_N', 'train', False)[0]).link_id.unique()
    fd = net.loc[sensors]
    lanes = float(fd.lanes.median())
    base = Scenario(seed=0, days=2, lanes=lanes, free_speed=float(fd.free_speed_kmh.median()),
        capacity=float((fd.capacity_vph/fd.lanes).median()*lanes),
        jam_density=float((fd.k_jam/fd.lanes).median()*lanes))
    cases = {
        'mild': dict(capacity_retained=.85, demand_multiplier=.9, incident_cell=8, failure_fraction=0.),
        'onset': dict(capacity_retained=.5, demand_multiplier=1.1, incident_cell=20),
        'severe_upstream': dict(capacity_retained=.3, demand_multiplier=1.25, incident_cell=8),
        'severe_downstream': dict(capacity_retained=.3, demand_multiplier=1.25, incident_cell=27),
        'failures': dict(capacity_retained=.5, failure_fraction=.4, failure_steps=36),
        'noisy': dict(capacity_retained=.5, speed_noise=3., relative_flow_noise=.05)
    }
    rows = []
    for seed in protocol['protocol']['simulator_development_seeds']:
        for name, kwargs in cases.items():
            for rate in (.2, .3, .5):
                sim = simulate(replace(base, seed=seed, mask_rate=rate, **kwargs))
                predictions = {'observation_oracle': (sim['observation_speed'], sim['observation_flow']),
                               'latent_oracle': (sim['latent_speed'], sim['latent_flow'])}
                interpolated = []
                smooth = []
                for raw, default, scale in [(sim['masked_speed'], base.free_speed, 25.),
                                             (sim['masked_flow'], .5*base.capacity, 600.*lanes)]:
                    v, _ = temporal(raw)
                    spatial = np.empty_like(raw)
                    positions = np.arange(raw.shape[1])
                    for t in range(len(raw)):
                        good = np.isfinite(raw[t])
                        spatial[t] = np.interp(positions, positions[good], raw[t,good]) if good.any() else default
                    v = np.where(np.isfinite(v), v, spatial)
                    interpolated.append(v)
                    smooth.append(kalman_residual(raw/scale, np.full_like(raw, default/scale))*scale)
                predictions['temporal_spatial'] = interpolated
                predictions['RTS'] = smooth
                rows.append({'seed': seed, 'scenario': name, 'mask_rate': rate,
                    'configuration': sim['scenario'], 'conservation_error': sim['max_conservation_error_vehicles'],
                    'congestion_share': float(sim['queue_truth'].mean()),
                    'candidates': {method: metrics(sim, *prediction) for method, prediction in predictions.items()}})
    result = {'experiment': 'B01_CTM_stress', 'data': 'independent CTM simulation; public FD aggregates',
        'simulator_is_organizer_generator': False, 'source': evidence(),
        'protocol_sha256': protocol['protocol_sha256'], 'rows': rows,
        'seconds': time.perf_counter()-started, 'formal_Kaggle_score': None}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({'scenarios': len(rows), 'seconds': result['seconds'],
        'max_conservation_error': max(r['conservation_error'] for r in rows),
        'mean_simulated_State': {method: np.mean([r['candidates'][method]['simulated_observation_State'] for r in rows])
            for method in predictions}})


if __name__ == '__main__':
    main()
