"""Topology restricted, gated propagation kernels and bounded Nyström GP.

All coordinates and gates are inference inputs; residual labels are separate.
Primary experiments fit one same-day posterior from sampled lawful visible
anchors, counterfactually masked before frozen-model anchor prediction.
See reports/v2/GP_MATHEMATICS.md for the PSD proof and its restrictions.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve, solve_triangular


@dataclass
class TrustedGeometry:
    links: tuple
    component: np.ndarray
    distance_km: np.ndarray
    edges: np.ndarray
    audit: dict


def trusted_chains(topology, lengths, mileposts, links):
    """Retain reciprocal, node-consistent, nonbranching directed chains only.

    Latent links remain in the path. Mileage validates direction and relative
    distance after topology has supplied edges; it never creates an edge.
    Untrusted or isolated measured links receive component -1 and zero kernel.
    """
    links = tuple(map(str, links))
    topo = topology.copy()
    topo.index = topo.index.astype(str)
    if topo.index.has_duplicates:
        raise ValueError('Topology link IDs must be unique')
    lengths = pd.Series(lengths, dtype=float)
    mileposts = pd.Series(mileposts, dtype=float)
    known = set(topo.index)
    def adjacent(node, side):
        if side+'_link_ids' not in topo:
            return set()
        return {s.strip() for s in str(topo.loc[node, side+'_link_ids']).split(';') if s.strip() in known}
    outgoing = {a: adjacent(a, 'outgoing') for a in known}
    incoming = {a: adjacent(a, 'incoming') for a in known}
    edges, rejected = [], []
    for a in sorted(known):
        for b in sorted(outgoing[a]):
            reason = None
            if a not in incoming[b]:
                reason = 'not_reciprocal'
            elif len(outgoing[a]) != 1 or len(incoming[b]) != 1:
                reason = 'mainline_branch'
            elif 'to_node' in topo and 'from_node' in topo and str(topo.loc[a, 'to_node']) != str(topo.loc[b, 'from_node']):
                reason = 'node_mismatch'
            elif not np.isfinite([lengths.get(a, np.nan), lengths.get(b, np.nan)]).all() or min(lengths[a], lengths[b]) <= 0:
                reason = 'invalid_length'
            if reason:
                rejected.append({'from': a, 'to': b, 'reason': reason})
            else:
                edges.append((a, b))
    successor, predecessor = dict(edges), {b: a for a, b in edges}
    chains, visited = [], set()
    for start in sorted(set(successor)-set(predecessor)):
        chain, node = [], start
        while node not in visited:
            visited.add(node); chain.append(node)
            if node not in successor:
                break
            node = successor[node]
        chains.append(chain)
    cycle_nodes = (set(successor)|set(predecessor))-visited
    component = np.full(len(links), -1, dtype=int)
    distance = np.zeros(len(links))
    link_lookup = {x: i for i, x in enumerate(links)}
    accepted, compressed = [], []
    for chain in chains:
        measured = [x for x in chain if x in link_lookup and np.isfinite(mileposts.get(x, np.nan))]
        if len(measured) < 2:
            continue
        coordinate = dict(zip(chain, np.r_[0., np.cumsum([lengths[x] for x in chain[:-1]])]))
        mileage = mileposts.reindex(measured).to_numpy(float)
        dx = np.diff([coordinate[x] for x in measured])
        dm = np.diff(mileage)
        # Milepost units differ between release producers; accept km/mile scale
        # near one while rejecting reversals, coincident stations and large gaps.
        ratios = np.abs(dm)/dx
        if not ((np.all(dm > 1e-8) or np.all(dm < -1e-8)) and np.all((ratios >= .25) & (ratios <= 4.))):
            rejected.append({'chain_start': chain[0], 'reason': 'mileage_direction_or_distance', 'n_links': len(chain)})
            continue
        identifier = len(accepted)
        for x in measured:
            component[link_lookup[x]] = identifier
            distance[link_lookup[x]] = coordinate[x]
        compressed.extend((link_lookup[a], link_lookup[b]) for a, b in zip(measured[:-1], measured[1:]))
        accepted.append({'id': identifier, 'n_topology_links': len(chain), 'n_measured': len(measured),
                         'mileage_sign': int(np.sign(dm[0])), 'median_mileage_per_km': float(np.median(ratios)),
                         'length_km': float(coordinate[chain[-1]]+lengths[chain[-1]])})
    audit = {'accepted_chains': accepted, 'n_trusted_measured': int((component >= 0).sum()),
             'n_measured': len(links), 'n_reciprocal_edges_retained': len(edges),
             'n_cycle_nodes_rejected': len(cycle_nodes), 'rejected': rejected,
             'distance_source': 'sum of released link lengths along reciprocal directed topology',
             'mileage_used_to_create_edges': False}
    return TrustedGeometry(links, component, distance, np.asarray(compressed, dtype=int).reshape(-1, 2), audit)


def legal_gate(speed_kmh, free_speed_kmh):
    """Congested-branch probability from lawful frozen prediction only."""
    ratio = np.asarray(speed_kmh, float)/np.asarray(free_speed_kmh, float)
    if not np.isfinite(ratio).all():
        raise ValueError('Gate inputs must be finite')
    return 1./(1.+np.exp(np.clip((ratio-.6)/.08, -40., 40.)))


def propagation_kernel(a, b, mode='directional', length_x=2., length_t=.5,
                       downstream_kmh=80., upstream_kmh=-15., state_length=1.):
    """a/b columns: x(km), t(hours), congestion gate, component, lawful state.

    Each branch is a Gaussian kernel under (x,t)->((x-ct)/ell_x,t/ell_t),
    multiplied by outer products of gates and a component delta kernel.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] < 4:
        raise ValueError('Kernel feature matrices disagree')
    if mode not in ['directional', 'ordinary', 'gated_ordinary']:
        raise ValueError('Unknown kernel mode')
    if min(length_x, length_t, state_length) <= 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Kernel inputs/length scales invalid')
    dx, dt = a[:, None, 0]-b[None, :, 0], a[:, None, 1]-b[None, :, 1]
    component = (a[:, None, 3] == b[None, :, 3]) & (a[:, None, 3] >= 0) & (b[None, :, 3] >= 0)
    if a.shape[1] > 4:
        dist = np.maximum(np.square(a[:, 4:]).sum(1)[:, None]+np.square(b[:, 4:]).sum(1)[None, :]-2*a[:, 4:]@b[:, 4:].T, 0.)
        shared = np.exp(-.5*dist/state_length**2)*component
    else:
        shared = component.astype(float)
    def branch(c):
        return np.exp(-.5*((dx-c*dt)/length_x)**2-.5*(dt/length_t)**2)
    if mode == 'ordinary':
        return shared*branch(0.)
    if ((a[:, 2] < 0)|(a[:, 2] > 1)).any() or ((b[:, 2] < 0)|(b[:, 2] > 1)).any():
        raise ValueError('Gate outside [0,1]')
    # sqrt weights maintain a unit diagonal and avoid attenuation confounding.
    free = np.sqrt(1.-a[:, None, 2])*np.sqrt(1.-b[None, :, 2])
    congested = np.sqrt(a[:, None, 2])*np.sqrt(b[None, :, 2])
    c1, c2 = (downstream_kmh, upstream_kmh) if mode == 'directional' else (0., 0.)
    return shared*(free*branch(c1)+congested*branch(c2))


class SparseResidualGP:
    """Deterministic Nyström GP with independent normalized output channels.

    Bayesian linear regression in whitened inducing features; exact posterior
    of the finite-rank GP, not the full GP. No N-by-N dense matrix is built.
    """
    def __init__(self, mode='directional', inducing=192, noise=.10, seed=20261016, batch=512):
        if inducing < 1 or noise <= 0 or batch < 1:
            raise ValueError('Sparse GP budget/noise invalid')
        self.mode, self.inducing_count, self.noise = mode, inducing, noise
        self.seed, self.batch = seed, batch

    def _features(self, x):
        cross = propagation_kernel(x, self.inducing, self.mode)
        return solve_triangular(self.inducing_cholesky, cross.T, lower=True, check_finite=False).T

    def fit(self, x, residual):
        x, residual = np.asarray(x, float), np.asarray(residual, float)
        valid = (x[:, 3] >= 0) & np.isfinite(residual).all(axis=1)
        x, residual = x[valid], residual[valid]
        if len(x) < 2 or residual.ndim != 2:
            raise ValueError('No trusted held-out residual observations')
        rng = np.random.default_rng(self.seed)
        # Seeded uniform sampling from every trusted training cell, independent
        # of residual value. No assessment or calibration coordinates used.
        chosen = np.sort(rng.choice(len(x), min(self.inducing_count, len(x)), replace=False))
        self.inducing = x[chosen].copy()
        gram = propagation_kernel(self.inducing, self.inducing, self.mode)
        self.jitter = 1e-7
        self.inducing_cholesky = cholesky(gram+self.jitter*np.eye(len(gram)), lower=True, check_finite=False)
        precision, rhs = np.eye(len(gram)), np.zeros((len(gram), residual.shape[1]))
        for start in range(0, len(x), self.batch):
            phi = self._features(x[start:start+self.batch])
            precision += phi.T@phi/self.noise
            rhs += phi.T@residual[start:start+self.batch]/self.noise
        self.posterior_cholesky = cholesky(precision, lower=True, check_finite=False)
        self.weights = cho_solve((self.posterior_cholesky, True), rhs, check_finite=False)
        self.fit_record = {'n_fit': len(x), 'n_inducing': len(gram), 'noise_variance': self.noise,
                           'jitter': self.jitter, 'mode': self.mode,
                           'largest_training_kernel_shape': [min(self.batch, len(x)), len(gram)],
                           'uncertainty': 'finite-rank residual-process posterior; no full-GP or calibrated interval claim'}
        return self

    def predict(self, x, variance=False):
        x = np.asarray(x, float)
        means, variances = [], []
        for start in range(0, len(x), self.batch):
            phi = self._features(x[start:start+self.batch])
            means.append(phi@self.weights)
            if variance:
                whitened = solve_triangular(self.posterior_cholesky, phi.T, lower=True, check_finite=False)
                variances.append(np.square(whitened).sum(axis=0))
        mean = np.concatenate(means)
        return (mean, np.concatenate(variances)) if variance else mean


def counterfactual_sampled_anchors(model, frame, net, topology, cache, geometry,
                                 maximum=512, folds=2, seed=20261016):
    """Predict sampled legal anchors with their own four channels withheld.

    Sampling uses only lawful availability, grid keys and the seed. A fold
    masks all sampled anchors assigned to it; other visible context is legal.
    Original missing target cells remain missing in every fold. The function
    never accesses target_truth or unmasked evaluator frames.
    """
    from ..state import StateFeatures
    from ..data import VALUES
    import hashlib
    if folds < 2 or maximum < 1:
        raise ValueError('Counterfactual anchors need >=2 folds and positive budget')
    channels = ['speed_kmh', 'flow_vph', 'occupancy', 'density_occ_linear_vehpkm']
    if not set(channels).issubset(frame.columns):
        raise ValueError('All four published measurement channels required')
    times, links = cache['times'].astype(str), cache['links'].astype(str)
    shape = (len(times), len(links))
    observed = cache['legal_observed']
    available = np.isfinite(observed).all(axis=-1).ravel()
    available &= np.tile(geometry.component >= 0, len(times))
    target_idx = cache['target_indices']
    if available[target_idx].any():
        raise ValueError('A target was exposed as a visible anchor')
    choices = np.flatnonzero(available)
    if len(choices) < 2:
        raise ValueError('Insufficient lawful anchors')
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(choices, min(maximum, len(choices)), replace=False))
    grouping = rng.permutation(len(selected)) % folds
    grid = pd.MultiIndex.from_product([times, links], names=['timestamp', 'link_id'])
    frame_idx = grid.get_indexer(pd.MultiIndex.from_frame(frame[['timestamp', 'link_id']]))
    if (frame_idx < 0).any() or pd.Series(frame_idx).duplicated().any():
        raise ValueError('Counterfactual anchors require one station per grid cell')
    original_targets = np.isin(frame_idx, target_idx)
    if frame.loc[original_targets, channels].notna().any().any():
        raise ValueError('Original missing targets must withhold all four measurement channels')
    lookup = pd.Series(np.arange(len(frame)), index=frame_idx)
    prior = np.zeros((len(selected), 2))
    lawful_features, certificates = [], []
    for group in range(folds):
        position = np.flatnonzero(grouping == group)
        if not len(position):
            continue
        flat = selected[position]
        row_indices = lookup.reindex(flat).to_numpy(int)
        altered = frame.copy()
        altered.loc[altered.index[row_indices], channels] = np.nan
        if altered.loc[altered.index[row_indices], channels].notna().any().any():
            raise ValueError('Counterfactual anchor retained a measurement channel')
        features = StateFeatures(altered, net, topology, model.profile, version=model.feature_version)
        if tuple(features.links) != tuple(links) or tuple(map(str, features.times)) != tuple(times):
            raise ValueError('Counterfactual feature grid order changed')
        target = frame.iloc[row_indices][['timestamp', 'station_id', 'link_id', 'mask_regime']].copy()
        target['panel'] = str(frame.corridor_id.iloc[0]) if 'panel' not in frame else str(frame.panel.iloc[0])
        # The frozen attention loader requests four CPU threads internally;
        # this isolated inference call caps that request while sharing the GPU
        # with the phase-one process. It changes no model weights/inputs.
        import torch
        original_set_threads = torch.set_num_threads
        torch.set_num_threads = lambda count: original_set_threads(min(int(count), 2))
        try:
            original_set_threads(2)
            prediction = model.predict(features, target)[VALUES].to_numpy(float)
        finally:
            torch.set_num_threads = original_set_threads
        lane = target.link_id.map(net.lanes).to_numpy(float)
        normalized = prediction/np.column_stack([np.full(len(target), 25.), 600*lane])
        prior[position] = normalized
        # Return lawful withheld-anchor feature rows for the same state kernel
        # and gate construction as original missing targets.
        lawful_features.append((position, features.features.iloc[flat].to_numpy(float),
                                np.asarray(features.features.columns)))
        certificates.append({'fold': group, 'n_withheld': len(flat), 'four_channels_withheld': True,
                             'flat_keys_sha256': hashlib.sha256(flat.tobytes()).hexdigest()})
    observed_normalized = observed.reshape(-1, 2)[selected]/np.array([25., 600.])
    return {'flat_indices': selected, 'prior': prior, 'residual': observed_normalized-prior,
            'lawful_feature_groups': lawful_features, 'certificates': certificates,
            'n_available': len(choices), 'seed': seed, 'folds': folds}
