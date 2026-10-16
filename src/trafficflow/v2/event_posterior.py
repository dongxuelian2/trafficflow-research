"""Small, causal conditional mixture of observed Queue event trajectories.

This is an empirical conditional posterior, not a recovered latent traffic state.
Each discrete latent component is a fit-period joint six-step trajectory, tagged
by an observed bottleneck location and graph-relative front summaries.  Its
weight is determined only by the legal thirteen-slot history and public event
condition.  Observation smoothing integrates a Gaussian speed/threshold error.
No prediction or decoding API accepts future eligibility or quality information.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.special import ndtr


def _history(x):
    x = np.asarray(x, float)
    if x.ndim != 4 or x.shape[1] != 13 or x.shape[3] != 9 or not np.isfinite(x).all():
        raise ValueError("Expected finite (window,13,node,9) causal histories")
    if not np.isin(x[..., [2, 3, 5, 8]], [0., 1.]).all():
        raise ValueError("History masks and public conditions must be binary")
    condition = x[:, 0, 0, 8].astype(bool)
    if not np.array_equal(x[..., 8], np.broadcast_to(condition[:, None, None], x.shape[:3])):
        raise ValueError("Public event condition must be constant within a window")
    return x, condition


def history_descriptor(x, bins=8):
    """Fixed low dimensional summaries; node bins are not propagation directions."""
    x, _ = _history(x)
    groups = np.array_split(np.arange(x.shape[2]), min(bins, x.shape[2]))
    values = []
    for group in groups:
        local = x[:, :, group]
        for channel, mask in [(0, 2), (1, 3), (4, 5)]:
            valid = local[..., mask]
            for times in [slice(-3, None), slice(0, 3)]:
                weight = valid[:, times]
                values.append((local[:, times, :, channel]*weight).sum((1, 2))/weight.sum((1, 2)).clip(min=1))
        values.append(local[..., 2].mean((1, 2)))
    values.extend([x[:, -1, 0, 6], x[:, -1, 0, 7]])
    return np.stack(values, axis=1)


def threshold_noise_scale(x):
    """Fit-only robust short-lag speed variation, not identified sensor noise.

    Actual traffic changes also enter these differences.  Clipping 1.5--8 km/h
    is a preregistered sensitivity assumption, not an estimated latent truth.
    """
    x, _ = _history(x)
    both = (x[:, 1:, :, 2] > .5) & (x[:, :-1, :, 2] > .5)
    delta = np.diff(x[..., 0]*25., axis=1)[both]
    if len(delta) < 10:
        return 3.
    median = np.median(delta)
    return float(np.clip(np.median(np.abs(delta-median))/.67448975/np.sqrt(2.), 1.5, 8.))


def signed_graph_distance(links, network, topology):
    """Sensor-graph signed shortest distances; disconnected pairs stay unknown.

    Positive means downstream from the row node.  No link ID or table order is
    interpreted as physical distance.  Missing intermediary links are not joined.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    links = list(links)
    index = {str(link): i for i, link in enumerate(links)}
    lengths = network.length_km.reindex(links).to_numpy(float)
    edges = np.full((len(links), len(links)), np.inf)
    np.fill_diagonal(edges, 0.)
    for link, i in index.items():
        if link not in topology.index:
            continue
        for downstream in str(topology.loc[link, 'outgoing_link_ids']).split(';'):
            if downstream in index:
                j = index[downstream]
                edges[i, j] = max((lengths[i]+lengths[j])/2., 1e-6)
    graph = csr_matrix(np.where(np.isfinite(edges), edges, 0.))
    directed = shortest_path(graph, directed=True)
    result = np.where(np.isfinite(directed), directed, np.nan)
    reverse = np.isfinite(directed.T) & ~np.isfinite(directed)
    result[reverse] = -directed.T[reverse]
    return result


@dataclass
class ConditionalEventPosterior:
    descriptors: np.ndarray
    descriptor_mean: np.ndarray
    descriptor_scale: np.ndarray
    templates: np.ndarray
    ongoing: np.ndarray
    bottleneck: np.ndarray
    fronts_km: np.ndarray
    directions: np.ndarray
    sigma_speed_kmh: float
    signed_distance_km: np.ndarray
    neighbors: int = 24
    bins: int = 8

    @classmethod
    def fit(cls, x, truth_speed, available, v_cut, signed_distance_km, neighbors=24, bins=8):
        x, ongoing = _history(x)
        speed, available = np.asarray(truth_speed, float)*25., np.asarray(available, bool)
        expected = (len(x), 6, x.shape[2])
        cut = np.asarray(v_cut, float)
        signed = np.asarray(signed_distance_km, float)
        if speed.shape != expected or available.shape != expected or cut.shape != (x.shape[2],):
            raise ValueError("Fit target geometry disagrees with history")
        if signed.shape != (x.shape[2], x.shape[2]) or not np.isfinite(cut).all() or neighbors < 1:
            raise ValueError("Invalid event geometry or neighbor count")
        if not np.isfinite(speed[available]).all() or not all((ongoing == c).any() for c in [False, True]):
            raise ValueError("Finite observed targets and both conditions required")
        sigma = threshold_noise_scale(x)
        # Missing supervised cells receive a causal fallback, never a future mask
        # during prediction.  The fit availability mask is a missing-target mask.
        time = np.arange(13)[None, :, None]
        last = np.where(x[..., 2] > .5, time, -1).max(axis=1)
        recent = np.take_along_axis(x[..., 0]*25., np.maximum(last, 0)[:, None], axis=1)[:, 0]
        recent = np.where(last >= 0, recent, cut[None]+2*sigma)
        filled = np.where(available, speed, recent[:, None])
        templates = ndtr((cut[None, None]-filled)/sigma)
        templates[~ongoing, :5] = 0.  # Public onset selection contract.
        bottleneck = np.argmax((templates*available).sum(axis=1), axis=1)
        fronts = np.zeros((len(x), 6, 2), float)
        direction = np.full(len(x), 2, np.int8)  # Unknown until at least two observed event times.
        for i, b in enumerate(bottleneck):
            distance = signed[b]
            observed_steps = []
            for h in range(6):
                use = (templates[i, h] >= .5) & available[i, h] & np.isfinite(distance)
                if use.any():
                    fronts[i, h] = [distance[use].min(), distance[use].max()]
                    observed_steps.append(h)
            if ongoing[i] and len(observed_steps) >= 2:
                movement = fronts[i, observed_steps[-1]].mean()-fronts[i, observed_steps[0]].mean()
                direction[i] = int(np.sign(movement)) if abs(movement) > .1 else 0
        desc = history_descriptor(x, bins)
        mean, scale = desc.mean(axis=0), desc.std(axis=0).clip(min=.05)
        return cls((desc-mean)/scale, mean, scale, templates.astype(np.float32), ongoing,
                   bottleneck, fronts, direction, sigma, signed, int(neighbors), int(bins))

    def component_weights(self, x):
        x, ongoing = _history(x)
        if x.shape[2] != self.templates.shape[2]:
            raise ValueError("Prediction node geometry changed")
        desc = (history_descriptor(x, self.bins)-self.descriptor_mean)/self.descriptor_scale
        result = np.zeros((len(x), len(self.templates)), float)
        for i, value in enumerate(desc):
            eligible = np.flatnonzero(self.ongoing == ongoing[i])
            distance = np.mean((self.descriptors[eligible]-value)**2, axis=1)
            order = np.argsort(distance, kind='stable')[:min(self.neighbors, len(eligible))]
            selected, local = eligible[order], distance[order]
            bandwidth = max(float(np.median(local)), .05)
            weight = np.exp(-.5*(local-local.min())/bandwidth)
            result[i, selected] = weight/weight.sum()
        return result

    def probability(self, x):
        weights = self.component_weights(x)
        # A convex combination can exceed one by a final floating-point ulp.
        return np.einsum('bf,fhn->bhn', weights, self.templates, optimize=True).clip(0., 1.)

    def event_summary(self, x):
        weights = self.component_weights(x)
        location = np.zeros((len(weights), self.templates.shape[2]))
        for b in range(self.templates.shape[2]):
            location[:, b] = weights[:, self.bottleneck == b].sum(axis=1)
        direction = np.stack([weights[:, self.directions == d].sum(axis=1) for d in [-1, 0, 1, 2]], axis=1)
        quantiles = np.zeros((len(weights), 3, 6, 2))
        for h in range(6):
            for side in range(2):
                order = np.argsort(self.fronts_km[:, h, side], kind='stable')
                cumulative = weights[:, order].cumsum(axis=1)
                for q, level in enumerate([.1, .5, .9]):
                    indices = (cumulative < level).sum(axis=1).clip(max=len(order)-1)
                    quantiles[:, q, h, side] = self.fronts_km[order[indices], h, side]
        return {'bottleneck_probability': location,
                'direction_probability_upstream_stationary_downstream_unknown': direction,
                'mean_front_range_km': np.einsum('bf,fhc->bhc', weights, self.fronts_km),
                'front_range_q10_q50_q90_km': quantiles,
                'graph_resolved_pair_fraction': float(np.isfinite(self.signed_distance_km).mean())}

    def sample(self, x, draws=256, seed=20261016):
        """Sample a shared trajectory component, then correlated observation noise."""
        if draws < 1:
            raise ValueError("Positive scenario count required")
        rng = np.random.default_rng(seed)
        weights = self.component_weights(x)
        result = np.empty((len(weights), draws, 6, self.templates.shape[2]), bool)
        for i, weight in enumerate(weights):
            component = rng.choice(len(weight), size=draws, p=weight)
            common = rng.normal(size=(draws, 1, self.templates.shape[2]))
            independent = rng.normal(size=(draws, 6, self.templates.shape[2]))
            uniform = ndtr(np.sqrt(.5)*common+np.sqrt(.5)*independent)
            result[i] = uniform < self.templates[component]
        return result


def stable_origin_seed(origin, seed=20261016):
    digest = hashlib.sha256((str(seed)+':'+str(origin)).encode()).digest()
    return int.from_bytes(digest[:8], 'little')


def independent_trajectories(probability, draws=256, seed=20261016):
    p = np.asarray(probability, float)
    if p.ndim != 2 or p.shape[0] != 6 or not np.isfinite(p).all() or ((p < 0)|(p > 1)).any():
        raise ValueError("Expected six-step finite probability grid")
    return np.random.default_rng(seed).random((draws, *p.shape)) < p


def expected_iou_decode(probability, scenarios, ongoing):
    """Finite-candidate full-domain Bayes action, no future scoring mask.

    Candidates: empty; fixed probability thresholds; top-k prefixes around the
    expected count; sixteen sampled trajectories.  This is not a claim of global
    set optimality.  All six horizons enter the joint IoU and onset's first five
    horizons are set empty from the public condition, before scoring.
    """
    p, z = np.asarray(probability, float), np.asarray(scenarios, bool)
    if p.ndim != 2 or p.shape[0] != 6 or z.ndim != 3 or z.shape[1:] != p.shape or not len(z):
        raise ValueError("Joint scenario geometry disagrees")
    if not np.isfinite(p).all() or ((p < 0)|(p > 1)).any():
        raise ValueError("Invalid probability")
    allowed = np.ones(p.shape, bool)
    if not ongoing:
        allowed[:5] = False
    p = np.where(allowed, p, 0.)
    z = z & allowed[None]
    candidates = [np.zeros(p.shape, bool)]
    candidates.extend((p >= t) & allowed for t in [.2, .35, .5, .65, .8])
    order = np.flatnonzero(allowed.ravel())
    order = order[np.argsort(-p.ravel()[order], kind='stable')]
    for multiplier in [.25, .5, .75, 1., 1.25, 1.5, 2.]:
        count = min(len(order), max(1, int(round(float(p.sum())*multiplier))))
        candidate = np.zeros(p.size, bool)
        candidate[order[:count]] = True
        candidates.append(candidate.reshape(p.shape))
    candidates.extend(z[index] for index in np.linspace(0, len(z)-1, min(16, len(z)), dtype=int))
    candidates = np.unique(np.stack(candidates).reshape(-1, p.size), axis=0)
    truth = z.reshape(len(z), -1).astype(np.int32)
    inter = candidates.astype(np.int32) @ truth.T
    union = candidates.sum(axis=1)[:, None]+truth.sum(axis=1)[None]-inter
    scores = np.divide(inter, union, out=np.ones(inter.shape, float), where=union > 0).mean(axis=1)
    best = np.flatnonzero(scores >= scores.max()-1e-12)
    selected = min(best, key=lambda i: (int(candidates[i].sum()), int(i)))
    return candidates[selected].reshape(p.shape).astype(np.int8), {
        'expected_IoU': float(scores[selected]), 'candidate_count': int(len(candidates)),
        'scenario_count': int(len(z)), 'selected_count': int(candidates[selected].sum())}
