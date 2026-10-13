"""Independent Godunov/CTM stress simulator using public aggregate parameters.

This is not the organizer generator. Its seeds, shocks and truth are our own.
It never seeks to recreate any validation/private scenario or hidden answer.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Scenario:
    seed: int
    cells: int = 32
    days: int = 3
    length_km: float = .75
    lanes: float = 4.
    free_speed: float = 110.
    capacity: float = 7600.
    jam_density: float = 400.
    demand_multiplier: float = 1.
    incident_cell: int = 20
    incident_start_hour: float = 8.
    incident_hours: float = 1.5
    capacity_retained: float = .45
    speed_noise: float = 1.
    relative_flow_noise: float = .015
    failure_fraction: float = .1
    failure_steps: int = 18
    mask_rate: float = .5


def simulate(scenario: Scenario):
    s = scenario
    if not 0 <= s.incident_cell < s.cells or s.days < 1:
        raise ValueError('Invalid scenario geometry')
    if min(s.length_km, s.free_speed, s.capacity, s.lanes) <= 0:
        raise ValueError('Non-positive geometry or FD')
    if s.jam_density <= s.capacity / s.free_speed:
        raise ValueError('Jam density must exceed critical density')
    if not all(0 <= p <= 1 for p in (s.capacity_retained, s.failure_fraction, s.mask_rate)):
        raise ValueError('Invalid scenario probability')
    rng = np.random.default_rng(s.seed)
    steps = s.days * 288
    wave = s.capacity / (s.jam_density - s.capacity / s.free_speed)
    # CFL < 1; retain every interval's actual average boundary flux.
    substeps = int(np.ceil((5 / 60) / (.8 * s.length_km / max(s.free_speed, wave))))
    dt = (5 / 60) / substeps
    rho = np.full(s.cells, .25 * s.capacity / s.free_speed)
    density = np.empty((steps + 1, s.cells))
    inflow = np.empty((steps, s.cells))
    outflow = np.empty_like(inflow)
    on = np.zeros_like(inflow)
    off = np.zeros_like(inflow)
    density[0] = rho
    peak_jitter = rng.uniform(-.4, .4, s.days)
    scale = rng.lognormal(0, .05, s.days)
    for t in range(steps):
        hour = (t % 288) / 12
        day = t // 288
        demand = s.capacity * s.demand_multiplier * scale[day] * (
            .30 + .65 * np.exp(-.5 * ((hour - 8 - peak_jitter[day]) / 1.7) ** 2)
            + .48 * np.exp(-.5 * ((hour - 17) / 2) ** 2))
        cap = np.full(s.cells, s.capacity)
        if s.incident_start_hour <= hour < s.incident_start_hour + s.incident_hours:
            cap[s.incident_cell] *= s.capacity_retained
        sum_in, sum_out = np.zeros(s.cells), np.zeros(s.cells)
        sum_on, sum_off = np.zeros(s.cells), np.zeros(s.cells)
        for _ in range(substeps):
            sending = np.minimum(s.free_speed * rho, cap)
            receiving = np.minimum(wave * np.maximum(s.jam_density - rho, 0), cap)
            flux = np.empty(s.cells + 1)
            flux[0] = min(demand, receiving[0])
            flux[1:-1] = np.minimum(sending[:-1], receiving[1:])
            flux[-1] = sending[-1]
            offr = np.zeros(s.cells)
            offr[s.cells // 3] = .025 * sending[s.cells // 3]
            onr = np.zeros(s.cells)
            ramp_cell = s.cells // 2
            # Ramps share receiving supply, so no density clipping conceals vehicles.
            room = max(0., receiving[ramp_cell] - flux[ramp_cell])
            onr[ramp_cell] = min(.02 * demand, room)
            rho = rho + dt / s.length_km * (flux[:-1] - flux[1:] + onr - offr)
            if rho.min() < -1e-8 or rho.max() > s.jam_density + 1e-8:
                raise RuntimeError('CTM violated invariant density bounds')
            sum_in += flux[:-1]
            sum_out += flux[1:]
            sum_on += onr
            sum_off += offr
        density[t + 1] = rho
        inflow[t], outflow[t], on[t], off[t] = (
            a / substeps for a in (sum_in, sum_out, sum_on, sum_off))
    q = np.minimum(s.free_speed * density, wave * (s.jam_density - density))
    v = q / np.maximum(density, 1e-8)
    obs_v = np.maximum(.5, v + rng.normal(0, s.speed_noise, v.shape))
    obs_q = np.maximum(0, q * rng.lognormal(-s.relative_flow_noise ** 2 / 2,
                                           s.relative_flow_noise, q.shape))
    eligible = rng.random(q.shape) < .756
    failure = np.zeros(q.shape, bool)
    count = int(np.ceil(s.cells * s.failure_fraction))
    for cell in rng.choice(s.cells, count, replace=False):
        start = rng.integers(0, max(1, steps - s.failure_steps))
        failure[start:start + s.failure_steps, cell] = True
    eligible &= ~failure
    target = eligible & (rng.random(q.shape) < s.mask_rate)
    observed = ~target & ~failure
    # Quality-degraded finite observations remain legal input; failures are NaN.
    masked_v = np.where(observed, obs_v, np.nan)
    masked_q = np.where(observed, obs_q, np.nan)
    balance_error = (np.diff(density, axis=0) * s.length_km
                     - (5 / 60) * (inflow + on - outflow - off))
    return {'scenario': asdict(s), 'density': density, 'latent_speed': v, 'latent_flow': q,
            'observation_speed': obs_v, 'observation_flow': obs_q,
            'masked_speed': masked_v, 'masked_flow': masked_q,
            'target': target, 'eligible': eligible, 'failure': failure,
            'inflow': inflow, 'outflow': outflow, 'on_ramp': on, 'off_ramp': off,
            'queue_truth': v <= .6 * s.free_speed, 'substeps': substeps,
            'max_conservation_error_vehicles': float(np.max(np.abs(balance_error)))}


def simulated_physics(sim, speed, flow, contract='ctm_latent_flux'):
    """Anchored field: modify scored targets only, preserve other observations.

    Two explicit synthetic contracts: physical CTM flux, and a flux net projected
    onto our noisy public observation truth (as described in the public scorer).
    Neither uses or estimates the actual competition's hidden boundary flux.
    """
    s = sim['scenario']
    target = sim['target']
    v = np.where(target, speed, sim['observation_speed'])
    q = np.where(target, flow, sim['observation_flow'])
    k = q / np.maximum(v, 1)
    critical = s['capacity'] / s['free_speed']
    wave = s['capacity'] / (s['jam_density'] - critical)
    qfd = np.minimum(s['free_speed'] * k, wave * np.maximum(s['jam_density'] - k, 0))
    valid = sim['eligible']
    sfd = max(0., 1 - np.abs(q[valid] - qfd[valid]).sum() / (np.abs(q[valid]).sum() + 1e-9))
    if target.any() and np.mean(q[target] < 50) > .2:
        sfd = 0.
    rhs = (5 / 60) * (sim['inflow'] + sim['on_ramp'] - sim['outflow'] - sim['off_ramp'])
    if contract == 'observation_projected_synthetic_flux':
        ntruth = sim['observation_flow'] / np.maximum(sim['observation_speed'], 1) * s['length_km']
        rhs = np.diff(ntruth, axis=0)
    elif contract != 'ctm_latent_flux':
        raise ValueError('Unknown synthetic flux contract')
    pair = valid[:-1] & valid[1:]
    error = np.abs(np.diff(k * s['length_km'], axis=0) - rhs)[pair].sum()
    denom = np.abs(rhs[pair]).sum()
    slwr = max(0., 1 - error / (denom + 1e-9))
    return {'simulated_S_FD': float(sfd), 'simulated_S_LWR': float(slwr),
            'simulated_S_physics': float((sfd + 2 * slwr) / 3),
            'n_transitions': int(pair.sum()), 'flux_contract': contract,
            'formal_Physics': None}
