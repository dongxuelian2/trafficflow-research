"""Public-observation residual assimilation around a frozen State predictor.

This is offline Task 1 inference. Bidirectional time edges are not causal Queue
features. The quadratic is an observation/noise diagnostic, not CTM flux truth.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg
from .data import VALUES, STATE_KEY
from .state import StateModel, StateFeatures, neighbors
from .quality_state import quality_channels


@dataclass(frozen=True)
class AssimilationParameters:
    prior_precision: float = 1.
    temporal_precision: float = 1.
    spatial_precision: float = 0.
    low_quality_weight: float = 0.
    innovation_cutoffs: tuple[float, float] | None = None
    time_edge_cutoffs: tuple[float, float] | None = None

    def validate(self):
        values = [self.prior_precision, self.temporal_precision,
                  self.spatial_precision, self.low_quality_weight]
        if not np.isfinite(values).all() or self.prior_precision <= 0 or min(values[1:]) < 0:
            raise ValueError('Assimilation precisions must be finite and nonnegative; prior positive')
        for cutoffs in [self.innovation_cutoffs, self.time_edge_cutoffs]:
            if cutoffs is not None and (len(cutoffs) != 2 or not np.isfinite(cutoffs).all() or min(cutoffs) <= 0):
                raise ValueError('Each channel cutoff must be finite and positive')


def observation_residuals(prior, strict, quality, parameters):
    """Normalized v/25 and lane-flow/600 innovations with legal reliability."""
    parameters.validate()
    prior, strict, quality = map(lambda x: np.asarray(x, float), [prior, strict, quality])
    if prior.shape != strict.shape or prior.ndim != 3 or prior.shape[-1] != 2 or quality.shape != prior.shape[:2]+(6,):
        raise ValueError('Assimilation observation grids disagree')
    if not np.isfinite(prior).all() or not np.isfinite(quality).all():
        raise ValueError('Prior and encoded quality inputs must be finite')
    strict_available = np.isfinite(strict)
    any_available = quality[..., 2:4] > .5
    confidence = np.minimum(np.clip(quality[..., 4:5], 0., 1.)/.75, 1.)**2
    confidence *= quality[..., 5:6] > .5
    weights = np.where(strict_available, 1.,
        parameters.low_quality_weight*confidence*any_available)
    observations = np.where(strict_available, strict, quality[..., :2])
    residuals = np.where(weights > 0, observations-prior, 0.)
    if parameters.innovation_cutoffs is not None:
        cutoffs = np.asarray(parameters.innovation_cutoffs)
        weights *= np.minimum(1., cutoffs/np.maximum(np.abs(residuals), 1e-12))
    return residuals, weights


def compressed_edges(topology, links):
    edges = set()
    for direction in ['incoming', 'outgoing']:
        for node, adjacent in enumerate(neighbors(topology, links, direction)):
            edges.update(tuple(sorted((node, other))) for other in adjacent if node != other)
    return np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)


def precision_matrix(shape, measurement_weights, prior, edges, parameters, channel):
    """Sparse Hessian of anchored residual smoothness; all unknowns regularized."""
    times, nodes = shape
    size = times*nodes
    left, right, weight = [], [], []
    if parameters.temporal_precision and times > 1:
        first = np.arange((times-1)*nodes)
        second = first+nodes
        w = np.full(len(first), parameters.temporal_precision)
        if parameters.time_edge_cutoffs is not None:
            change = np.abs(np.diff(prior[..., channel], axis=0)).ravel()
            w *= np.exp(-np.minimum((change/parameters.time_edge_cutoffs[channel])**2, 50.))
        left.append(first); right.append(second); weight.append(w)
    if parameters.spatial_precision and len(edges):
        offsets = np.arange(times)[:, None]*nodes
        left.append((offsets+edges[:, 0]).ravel())
        right.append((offsets+edges[:, 1]).ravel())
        weight.append(np.full(times*len(edges), parameters.spatial_precision))
    diagonal = parameters.prior_precision+measurement_weights[..., channel].ravel()
    matrix = sparse.diags(diagonal, format='csr')
    if left:
        a, b, w = np.concatenate(left), np.concatenate(right), np.concatenate(weight)
        matrix += sparse.coo_matrix((np.concatenate([w, w, -w, -w]),
            (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))), shape=(size, size)).tocsr()
    return matrix


def assimilate_residuals(prior, strict, quality, edges, parameters, anchor_prior=None):
    """Solve one SPD quadratic residual estimate per channel, with certificates.

    The weights are calibration-selected surrogates, not validated measurement
    variances. No physical flux or Bayesian uncertainty claim follows.
    """
    parameters.validate()
    prior = np.asarray(prior, float)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    if len(edges) and (edges.min() < 0 or edges.max() >= prior.shape[1] or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError('Invalid compressed sensor edges')
    residuals, weights = observation_residuals(prior if anchor_prior is None else anchor_prior, strict, quality, parameters)
    correction = np.zeros_like(prior)
    diagnostics = []
    for channel in range(2):
        matrix = precision_matrix(prior.shape[:2], weights, prior, edges, parameters, channel)
        rhs = (weights[..., channel]*residuals[..., channel]).ravel()
        if np.any(rhs):
            preconditioner = sparse.diags(1./matrix.diagonal(), format='csr')
            iterations = [0]
            def count(_):
                iterations[0] += 1
            solution, status = cg(matrix, rhs, M=preconditioner, rtol=1e-8, atol=0., maxiter=400, callback=count)
            relative_residual = float(np.linalg.norm(matrix@solution-rhs)/np.linalg.norm(rhs))
            if status != 0 or not np.isfinite(solution).all() or relative_residual > 2e-8:
                raise ValueError(f'Assimilation did not converge: {status}, {relative_residual}')
            correction[..., channel] = solution.reshape(prior.shape[:2])
        else:
            iterations, relative_residual = [0], 0.
        diagnostics.append({'channel': channel, 'iterations': iterations[0],
                            'relative_linear_residual': relative_residual,
                            'nonzero_observation_weights': int((weights[..., channel] > 0).sum())})
    return correction, diagnostics


def normalized_prior_grid(model, features, target):
    # The current public panels have one station per sensor link/time. Reject
    # ambiguous aggregation rather than silently applying station predictions.
    frame = features.frame
    if frame.duplicated(['timestamp', 'link_id']).any():
        raise ValueError('Assimilation requires unique observed station/link cells')
    if len(target.panel.unique()) != 1:
        raise ValueError('Assimilation needs exactly one public panel')
    if len(features.times) > 1 and not np.all(np.diff(pd.to_datetime(features.times, utc=True).as_unit('ns').asi8) == 300_000_000_000):
        raise ValueError('Assimilation time grid must have adjacent five-minute slots')
    full = frame[['timestamp', 'station_id', 'link_id', 'mask_regime']].copy()
    full['panel'] = target.panel.iloc[0]
    prediction = model.predict(features, full)[VALUES].to_numpy(float, copy=True)
    prediction[:, 1] /= full.link_id.map(features.net.lanes).to_numpy(float)
    result = np.full((len(features.times)*len(features.links), 2), np.nan)
    result[features.lookup(full)] = prediction/np.array([25., 600.])
    if not np.isfinite(result).all():
        raise ValueError('Frozen prior does not cover full visible grid')
    return result.reshape(len(features.times), len(features.links), 2)


def counterfactual_anchor_prior(model, features, target, prior, folds=5, seed=20261013):
    """Predict every visible anchor once with its own four channels withheld.

    Random groups are determined solely by public cell ordering, panel/date and
    a fixed seed. Each group's other anchors remain legal input context. The
    original target prior stays unchanged; only innovation centering uses these
    counterfactual predictions. This adds missingness and is not cross-fitting
    model-training labels or an unbiased posterior guarantee.
    """
    if not isinstance(folds, int) or folds < 2:
        raise ValueError('Counterfactual anchors require at least two groups')
    frame = features.frame
    if frame.duplicated(['timestamp', 'link_id']).any():
        raise ValueError('Counterfactual groups need unique station/link cells')
    lookup = features.lookup(frame)
    available = np.isfinite(frame[VALUES].to_numpy(float)).any(axis=1)
    identity = f'{seed}|{target.panel.iloc[0]}|{features.times[0]}'
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little'))
    assignments = rng.integers(0, folds, len(features.times)*len(features.links))
    result = np.asarray(prior, float).copy().reshape(-1, 2)
    certificates = []
    for group in range(folds):
        selected = available & (assignments[lookup] == group)
        if not selected.any():
            continue
        masked = frame.copy()
        columns = [c for c in [*VALUES, 'occupancy', 'density_occ_linear_vehpkm'] if c in masked]
        masked.loc[selected, columns] = np.nan
        if masked.loc[selected, columns].notna().any().any():
            raise ValueError('Counterfactual anchors exposed own measurement channels')
        altered = StateFeatures(masked, features.net, features_topology(model), model.profile, version=model.feature_version)
        mean = normalized_prior_grid(model, altered, target).reshape(-1, 2)
        result[lookup[selected]] = mean[lookup[selected]]
        certificates.append({'group': group, 'n_visible_cells_withheld': int(selected.sum()),
                             'four_channels_withheld': True})
    if sum(r['n_visible_cells_withheld'] for r in certificates) != int(available.sum()):
        raise ValueError('Visible anchor grouping incomplete')
    return result.reshape(prior.shape), certificates


class AssimilatedStateModel(StateModel):
    """Frozen learned prior plus public-observation residual assimilation."""
    def __init__(self, prior, parameters, strength, anchor_folds=0, anchor_seed=20261013):
        parameters.validate()
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Invalid assimilation strength')
        self.prior_model = prior
        self.parameters = parameters
        self.strength = float(strength)
        if anchor_folds != 0 and (not isinstance(anchor_folds, int) or anchor_folds < 2):
            raise ValueError('Counterfactual anchors need zero or at least two groups')
        self.anchor_folds, self.anchor_seed = anchor_folds, anchor_seed
        self.profile = prior.profile
        self.feature_version = prior.feature_version

    def predict(self, features, template):
        if self.strength == 0:
            return self.prior_model.predict(features, template)
        prior = normalized_prior_grid(self.prior_model, features, template)
        strict = features.raw/np.array([25., 600.])
        quality = quality_channels(features)
        edges = compressed_edges(features_topology(self.prior_model), features.links)
        anchor_prior = None
        if getattr(self, 'anchor_folds', 0):
            anchor_prior, _ = counterfactual_anchor_prior(self.prior_model, features, template, prior,
                self.anchor_folds, self.anchor_seed)
        correction, _ = assimilate_residuals(prior, strict, quality, edges, self.parameters, anchor_prior)
        grid = prior+self.strength*correction
        values = grid.reshape(-1, 2)[features.lookup(template)]*np.array([25., 600.])
        lanes = template.link_id.map(features.net.lanes).to_numpy(float)
        values[:, 0] = np.clip(values[:, 0], 0., template.link_id.map(features.net.free_speed_kmh).to_numpy(float)*1.05)
        values[:, 1] = np.clip(values[:, 1]*lanes, 0., template.link_id.map(features.net.capacity_vph).to_numpy(float)*1.05)
        result = template[STATE_KEY].copy()
        result[VALUES] = values
        return result

    def save(self, path):
        # Neural prior caches may be on GPU. Its own save() clears caches, but
        # nested pickle needs the same explicit behavior without mutating bytes.
        cache = getattr(self.prior_model, '_network', None)
        if hasattr(self.prior_model, '_network'):
            self.prior_model._network = None
        try:
            super().save(path)
        finally:
            if hasattr(self.prior_model, '_network'):
                self.prior_model._network = cache


def features_topology(model):
    if not hasattr(model, 'topology'):
        raise ValueError('Frozen assimilation prior must retain public topology')
    return model.topology
