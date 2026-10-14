"""Public-geometry travel-time neighborhoods for causal Queue message passing."""
import numpy as np
from .queue import _road_distances


def travel_time_neighborhoods(links, net, topo, horizon_minutes):
    """Return upstream/downstream kernels within a public backward-wave time.

    These are spatial representation choices, not a hard propagation constraint:
    repeated neural layers and global normalization can communicate farther.
    All numeric features still come exclusively from timestamps <= origin.
    """
    horizon = float(horizon_minutes)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError('A finite positive travel horizon is required')
    nodes = tuple(net.index.astype(str))
    index = {node: i for i, node in enumerate(nodes)}
    if not set(links).issubset(index):
        raise ValueError('Observed links missing from public geometry')
    lengths = tuple(net.length_km.to_numpy(float))
    waves = tuple((net.capacity_vph/(net.k_jam-net.critical_density).clip(lower=1e-6)).clip(lower=1.).to_numpy(float))
    edges = []
    for source, row in topo.iterrows():
        edges.extend((str(source), destination.strip()) for destination in str(row.get('outgoing_link_ids', '')).split(';') if destination.strip())
    _, minutes = _road_distances(nodes, lengths, waves, tuple(edges))
    chosen = np.array([index[link] for link in links])
    minutes = minutes[np.ix_(chosen, chosen)]
    def kernel(distance):
        weights = np.where(np.isfinite(distance)&(distance <= horizon), np.exp(-np.minimum(distance, 1e4)/(horizon/2)), 0.)
        np.fill_diagonal(weights, 0.)
        denominator = weights.sum(axis=1, keepdims=True)
        return np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0).astype(np.float32)
    return kernel(minutes.T), kernel(minutes)
