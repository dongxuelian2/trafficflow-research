"""Causal history-only temporal/graph Queue forecasting research models."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import eligible
from .graph_state import GraphBlock


def queue_history_input(history, links, net, origin, condition):
    """Nine channels on exactly thirteen past/current five-minute slots."""
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None or condition not in ('queue_onset', 'queue_ongoing'):
        raise ValueError('Timezone-aware origin and public condition required')
    timestamps = pd.to_datetime(history.timestamp, utc=True)
    if (timestamps > origin).any() or (timestamps < origin-pd.Timedelta(minutes=60)).any():
        raise ValueError('Queue future input leakage or history outside the allowed hour')
    slots = pd.date_range(origin-pd.Timedelta(minutes=60), origin, freq='5min')
    if not timestamps.isin(slots).all():
        raise ValueError('Queue history is off the five-minute grid')
    frame = history.copy()
    frame['_time'] = timestamps
    frame.loc[~eligible(frame), ['speed_kmh', 'flow_vph', 'occupancy']] = np.nan
    def grid(column):
        return frame.pivot_table(index='_time', columns='link_id', values=column, aggfunc='mean').reindex(index=slots, columns=links).to_numpy(float)
    speed = grid('speed_kmh')/25.
    flow = grid('flow_vph')/net.lanes.reindex(links).to_numpy(float)/600.
    occupancy = grid('occupancy')/.25
    phase = np.asarray(2*np.pi*(slots.hour*60+slots.minute)/1440)[:, None]
    shape = speed.shape
    channels = [np.nan_to_num(speed), np.nan_to_num(flow), np.isfinite(speed), np.isfinite(flow),
        np.nan_to_num(occupancy), np.isfinite(occupancy), np.broadcast_to(np.sin(phase), shape),
        np.broadcast_to(np.cos(phase), shape), np.full(shape, float(condition == 'queue_ongoing'))]
    result = np.stack(channels, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite causal Queue input')
    return result


class CausalGraphQueue(nn.Module):
    """Predict six future horizons from thirteen history slots, never targets."""
    def __init__(self, hidden=32, use_graph=True):
        super().__init__()
        if hidden % 4:
            raise ValueError('Hidden width must match GroupNorm')
        self.use_graph = use_graph
        self.input = nn.Conv2d(9, hidden, 1)
        self.geometry = nn.Conv2d(4, hidden, 1)
        self.blocks = nn.ModuleList([GraphBlock(hidden, d) for d in (1, 2, 4)])
        self.output = nn.Conv2d(2*hidden, 12, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, incoming, outgoing, geometry):
        batch, channels, times, nodes = x.shape
        if channels != 9 or times != 13 or incoming.shape != (nodes, nodes) or outgoing.shape != (nodes, nodes) or geometry.shape != (4, nodes):
            raise ValueError('Causal Queue requires a thirteen-slot history and matching geometry')
        scale = torch.tensor([25., 600., 25., 25.], device=x.device, dtype=x.dtype)[:, None]
        z = torch.nn.functional.silu(self.input(x)+self.geometry((geometry/scale)[None, :, None]))
        up, down = (incoming, outgoing) if self.use_graph else (torch.zeros_like(incoming), torch.zeros_like(outgoing))
        for block in self.blocks:
            z = block(z, up, down)
        decoded = self.output(torch.cat([z[:, :, -1:], z.mean(dim=2, keepdim=True)], dim=1))[:, :, 0]
        residual = decoded.reshape(batch, 2, 6, nodes)
        steps = torch.arange(13, device=x.device)[None, :, None]
        last_index = torch.where(x[:, 2] > .5, steps, -1).max(dim=1).values
        latest = torch.gather(x[:, 0], 1, last_index.clamp(min=0)[:, None]).squeeze(1)
        latest = torch.where(last_index >= 0, latest, geometry[0][None]/25.)
        logits = 2.5*(geometry[3][None, None]/25.-latest[:, None])+residual[:, 0]
        speed = latest[:, None]+residual[:, 1]
        return logits, speed


def queue_event_loss(logits, speed, labels, truth_speed, available, ongoing):
    """Equal-window loss, omitting known-empty early onset classification cells."""
    domain = available.clone()
    domain[:, :5] &= ongoing[:, None, None]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none',
        pos_weight=torch.tensor(4., device=logits.device))
    classification = (bce*domain).sum(dim=(1, 2))/domain.sum(dim=(1, 2)).clamp(min=1)
    regression = ((speed-truth_speed).abs()*available).sum(dim=(1, 2))/available.sum(dim=(1, 2)).clamp(min=1)
    return (classification+.1*regression).mean()
