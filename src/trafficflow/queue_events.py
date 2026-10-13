"""Randomized event sampling approximating only published Task 2 safeguards.

Actual underlying queue labels and selector implementation are not released.
This sampler's public-train observation labels remain a biased noisy proxy.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import eligible
from .queue import ProxyData, queue_features
from .metrics import queue_iou


def sample_events(frames, net, topo, panel, seed, per_condition=5,
                  spacing_minutes=360, maximum_persistence_iou=.9,origin_reader=None,include_occupancy=False,onset_last_step=False):
    rng = np.random.default_rng(seed)
    days = []
    candidates = []
    for raw in frames:
        times = sorted(raw.timestamp.unique())
        if len(times) < 20:
            continue
        links = raw.link_id.drop_duplicates().tolist()
        quality = raw.copy()
        quality.loc[~eligible(raw), ['speed_kmh', 'flow_vph']] = np.nan
        speeds = quality.pivot_table(index='timestamp', columns='link_id', values='speed_kmh', aggfunc='mean').reindex(index=times, columns=links).to_numpy(float)
        cutoff = net.v_cut.reindex(links).to_numpy(float)
        queue = speeds <= cutoff
        day_index = len(days)
        days.append((raw, times, links, speeds, cutoff))
        for at in range(12, len(times)-6):
            history = speeds[at-12:at]
            future = speeds[at+1:at+7]
            valid_future = np.isfinite(future)
            if np.isfinite(history).mean() < .7 or not (future <= cutoff).any():
                continue
            counts = queue[at-12:at].sum(axis=0)
            condition = 'queue_ongoing' if (counts >= 2).any() else 'queue_onset'
            if onset_last_step and condition=='queue_onset':
                # Public host announcement fixes onset timing in released
                # windows. This training sampler can only approximate it with
                # noisy public-train observations, never latent queue truth.
                if (future[:-1]<=cutoff).any() or not (future[-1]<=cutoff).any():continue
            current = np.full(len(links), np.nan)
            for lag in range(1, 13):
                value = speeds[at-lag]
                pick = ~np.isfinite(current) & np.isfinite(value)
                current[pick] = value[pick]
            persistence = np.broadcast_to(current <= cutoff, future.shape)
            iou = queue_iou((future <= cutoff)[valid_future], persistence[valid_future])
            if iou > maximum_persistence_iou:
                continue
            candidates.append({'day': day_index, 'at': at, 'condition': condition,
                'origin': pd.Timestamp(times[at]), 'persistence_proxy_iou': iou,
                'future_label_coverage': float(valid_future.mean()),
                'near_cutoff_share': float((np.abs(future-cutoff)[valid_future] <= 2.).mean())})
    chosen = []
    counts = {'queue_onset': 0, 'queue_ongoing': 0}
    # Candidate origins are uniformly permuted; spacing rejection is explicit.
    # This is an approximation, not a claim about organizer selection weights.
    for i in rng.permutation(len(candidates)):
        candidate = candidates[i]
        condition = candidate['condition']
        if counts[condition] >= per_condition:
            continue
        if any(abs((candidate['origin']-old['origin']).total_seconds()) < spacing_minutes*60 for old in chosen):
            continue
        chosen.append(candidate)
        counts[condition] += 1
        if all(n >= per_condition for n in counts.values()):
            break
    if not chosen:
        raise ValueError('No eligible proxy events')
    xs, ys, windows, conditions, dates = [], [], [], [], []
    for candidate in sorted(chosen, key=lambda c:c['origin']):
        raw, times, links, speed, cutoff = days[candidate['day']]
        at = candidate['at']
        history = raw[raw.timestamp.isin(times[at-12:at])].copy()
        if origin_reader is not None:
            current=origin_reader(candidate['origin'])
            if not pd.to_datetime(current.timestamp,utc=True).eq(candidate['origin']).all():
                raise ValueError('Origin reader returned a timestamp other than T')
            history=pd.concat([history,current],ignore_index=True)
        target = pd.MultiIndex.from_product([times[at+1:at+7], links], names=['timestamp', 'link_id']).to_frame(index=False)
        x = queue_features(history, target, net, topo, candidate['origin'],allow_origin=origin_reader is not None,include_occupancy=include_occupancy)
        future = speed[at+1:at+7]
        usable = np.isfinite(future).ravel()
        wid = f'{panel}:{times[at]}:{seed}'
        xs.append(x.loc[usable])
        ys.extend((future <= cutoff).ravel()[usable].astype(int))
        windows.extend([wid]*usable.sum())
        conditions.extend([candidate['condition']]*usable.sum())
        dates.extend([times[at][:10]]*usable.sum())
    metadata = {'seed': seed, 'candidate_counts': {c:sum(r['condition']==c for r in candidates) for c in counts},
        'selected_counts': counts, 'requested_per_condition': per_condition,
        'label_provenance': 'eligible noisy public train observations; unknown future cells omitted',
        'unknown_formal_selector': True,
        'origin_timestamp_inputs':origin_reader is not None,
        'occupancy_features':include_occupancy,
        'onset_observation_first_queue_at_30':onset_last_step,
        'windows': [{**c, 'origin': c['origin'].isoformat()} for c in chosen]}
    return ProxyData(pd.concat(xs, ignore_index=True), np.array(ys), np.array(windows),
                     np.array(conditions), np.array(dates)), metadata


class EventQueueModel:
    """Separate onset/ongoing probability models; shared input is causal."""
    def __init__(self, seed=20261013, iterations=400):
        self.seed, self.iterations = seed, iterations
        self.models = {}
        self.thresholds = {}

    def fit(self, data):
        from lightgbm import LGBMClassifier
        for condition in np.unique(data.condition):
            selection = data.condition == condition
            model = LGBMClassifier(n_estimators=self.iterations, learning_rate=.035,
                num_leaves=23, min_child_samples=60, reg_lambda=8., n_jobs=4,
                random_state=self.seed, verbosity=-1, deterministic=True, force_col_wise=True)
            model.fit(data.x.loc[selection], data.y[selection])
            self.models[condition] = model
        return self

    def probability(self, data):
        p = np.zeros(len(data.y))
        for condition in np.unique(data.condition):
            selection = data.condition == condition
            p[selection] = self.models[condition].predict_proba(data.x.loc[selection])[:, 1]
        return p

    def tune(self, data):
        from .metrics import grouped_iou
        p = self.probability(data)
        for condition in np.unique(data.condition):
            selection = data.condition == condition
            values = [(t, grouped_iou(data.y[selection], p[selection]>=t,
                       data.window[selection], data.condition[selection])['proxy_IoU']) for t in np.linspace(.15, .75, 13)]
            self.thresholds[condition] = float(max(values, key=lambda r:(r[1], -abs(r[0]-.5)))[0])
        return self

    def classify(self, data):
        p = self.probability(data)
        threshold = np.array([self.thresholds[c] for c in data.condition])
        return (p >= threshold).astype(int)
