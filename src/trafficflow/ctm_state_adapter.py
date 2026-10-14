"""Public-input adapter for independently simulated offline State grids.

The signature excludes latent states and reconstruction truth. Occupancy is
unavailable in this simulator and stays unavailable; it is not fabricated from
latent density. Geometry-only priors deliberately test a distribution shift.
"""
import numpy as np
from .state import temporal


def synthetic_state_input(masked_speed, masked_flow, quality_eligible, *, lanes,
                          free_speed, capacity, jam_density, start_minute=0):
    speed, flow = np.asarray(masked_speed, float), np.asarray(masked_flow, float)
    eligible = np.asarray(quality_eligible, bool)
    if speed.ndim != 2 or speed.shape != flow.shape or speed.shape != eligible.shape or min(speed.shape) < 1:
        raise ValueError('Matching nonempty time/node grids required')
    if not np.isfinite([lanes, free_speed, capacity, jam_density]).all() or min(lanes, free_speed, capacity) <= 0 or jam_density <= capacity/free_speed:
        raise ValueError('Valid finite traffic geometry required')
    if np.isinf(speed).any() or np.isinf(flow).any():
        raise ValueError('Observations may be NaN but not infinite')
    raw = np.stack([np.where(eligible, speed, np.nan),
                    np.where(eligible, flow/lanes, np.nan)], axis=-1)
    available = np.isfinite(raw)
    defaults = np.array([free_speed, .4*capacity/lanes])
    base = np.empty_like(raw)
    position = np.arange(speed.shape[1])
    for channel in range(2):
        series = raw[..., channel]
        interpolated, _ = temporal(series)
        spatial = np.empty_like(series)
        for time, row in enumerate(series):
            good = np.isfinite(row)
            spatial[time] = np.interp(position, position[good], row[good]) if good.any() else defaults[channel]
        base[..., channel] = np.where(np.isfinite(interpolated), interpolated, spatial)
    scale = np.array([25., 600.])
    prior = np.broadcast_to(defaults, raw.shape)
    shape = (*speed.shape, 1)
    angle = 2*np.pi*(start_minute+5*np.arange(len(speed)))/1440
    sin = np.broadcast_to(np.sin(angle)[:, None, None], shape)
    cos = np.broadcast_to(np.cos(angle)[:, None, None], shape)
    x = np.concatenate([np.nan_to_num(raw)/scale, available, prior/scale,
                        base/scale, np.zeros(shape), np.zeros(shape), sin, cos], axis=-1).astype(np.float32)
    nodes = speed.shape[1]
    incoming, outgoing = np.zeros((nodes, nodes), np.float32), np.zeros((nodes, nodes), np.float32)
    incoming[np.arange(1, nodes), np.arange(nodes-1)] = 1.
    outgoing[np.arange(nodes-1), np.arange(1, nodes)] = 1.
    fd = np.tile(np.array([free_speed, capacity/lanes, jam_density/lanes], np.float32)[:, None], (1, nodes))
    base[..., 0] = np.clip(base[..., 0], 0, 1.05*free_speed)
    base[..., 1] = np.clip(base[..., 1], 0, 1.05*capacity/lanes)
    return x, base, incoming, outgoing, fd
