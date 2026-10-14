"""Legally visible any-quality observations and reliability for offline State.

No raw labels or occupancy synthesis. The original twelve channels remain
unchanged; extra inputs are read only from the supplied masked public frame.
"""
import numpy as np
import pandas as pd


def quality_channels(features):
    frame = features.frame
    def grid(column):
        return frame.pivot_table(index='timestamp', columns='link_id', values=column, aggfunc='mean').reindex(
            index=features.times, columns=features.links).to_numpy(float)
    lane = features.net.lanes.reindex(features.links).to_numpy(float)
    if not np.isfinite(lane).all() or (lane <= 0).any():
        raise ValueError('Quality inputs need finite positive public lane counts')
    values = np.stack([grid('speed_kmh')/25., grid('flow_vph')/lane[None, :]/600.], axis=-1)
    available = np.isfinite(values)
    percent = grid('pct_observed')
    known_percent = np.isfinite(percent)
    confidence = np.where(known_percent, np.clip(percent, 0., 100.)/100., 0.)
    result = np.concatenate([np.where(available, values, 0.), available.astype(float),
        confidence[..., None], known_percent[..., None]], axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Quality input overflows float32')
    return result


def quality_reconstruction_input(features, strict_input=None):
    if strict_input is None:
        from .adaptive_graph_state import reconstruction_input
        strict_input = reconstruction_input(features)
    if strict_input.shape[-1] != 12:
        raise ValueError('Quality extension requires the original twelve channels')
    return np.concatenate([strict_input, quality_channels(features)], axis=-1)


def expand_shared_state(state, channels):
    """Exact shared parameter initialization; added input weights start at zero."""
    original = state['input.weight']
    if channels < original.shape[1]:
        raise ValueError('Expanded model cannot remove shared input channels')
    result = {k: v.detach().clone() for k, v in state.items()}
    expanded = original.new_zeros(original.shape[0], channels, *original.shape[2:])
    expanded[:, :original.shape[1]] = original
    result['input.weight'] = expanded
    return result
