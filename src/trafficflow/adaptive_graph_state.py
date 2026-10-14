"""Independent missing-aware adaptive spatial attention for offline State.

Uses published FD geometry and directed compressed sensor adjacency. No learned
per-corridor node-ID table and no reconstruction targets enter the input.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from .graph_state import GraphBlock
from .state import StateModel,neighbors
from .data import eligible,VALUES


class AdaptiveSpatialBlock(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.relation_bias = nn.Parameter(torch.zeros(3))
        self.gain = nn.Parameter(torch.tensor(.1))

    def forward(self, z, incoming, outgoing):
        batch, channels, times, nodes = z.shape
        token = z.permute(0, 2, 3, 1).reshape(batch*times, nodes, channels)
        normalized = self.norm(token)
        bias = (self.relation_bias[0]*(incoming > 0).to(z.dtype) +
                self.relation_bias[1]*(outgoing > 0).to(z.dtype) +
                self.relation_bias[2]*torch.eye(nodes, device=z.device, dtype=z.dtype))
        attended, _ = self.attention(normalized, normalized, normalized,
            attn_mask=bias, need_weights=False)
        token = token + self.gain*attended
        return token.reshape(batch, times, nodes, channels).permute(0, 3, 1, 2).contiguous()


class AdaptiveGraphState(nn.Module):
    def __init__(self, channels=12, hidden=48, heads=2, attention=True):
        super().__init__()
        if hidden % heads or hidden % 4:raise ValueError('Hidden width must match heads and GroupNorm')
        self.input = nn.Conv2d(channels, hidden, 1)
        self.geometry = nn.Conv2d(3, hidden, 1)
        self.blocks = nn.ModuleList([GraphBlock(hidden, d) for d in (1, 2, 4, 8)])
        self.attention_enabled=attention
        self.spatial = nn.ModuleList([AdaptiveSpatialBlock(hidden, heads) for _ in range(2)] if attention else [])
        self.output = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, incoming, outgoing, fd):
        nodes = x.shape[-1]
        if incoming.shape != (nodes, nodes) or outgoing.shape != (nodes, nodes) or fd.shape != (3, nodes):
            raise ValueError('Public graph geometry does not match the observed grid')
        scale = torch.tensor([25., 600., 25.], device=x.device, dtype=x.dtype)[:, None]
        geometry = (fd/scale)[None, :, None, :]
        z = torch.nn.functional.silu(self.input(x)+self.geometry(geometry))
        for i, block in enumerate(self.blocks):
            z = block(z, incoming, outgoing)
            if self.attention_enabled and i in (1, 3):z = self.spatial[i//2](z, incoming, outgoing)
        return x[:, 6:8]+self.output(z)


def patch_score_loss(prediction,truth,target):
    """Equal-patch speed/lane-flow RMSE in official channel scales.

    Training samples regimes uniformly. This differentiable patch objective
    approximates regime-pooled evaluation; it is not the official scorer.
    """
    squared=((prediction-truth).square()*target[:,None]).sum(dim=(2,3))
    mean=squared/target.sum(dim=(1,2))[:,None].clamp(min=1)
    weight=torch.tensor([.54,.46],device=prediction.device)[None,:]
    return (torch.sqrt(mean+1e-8)*weight).sum(dim=1).mean()


def reconstruction_input(features):
    """Construct the same 12 public-input channels as the audited graph cache."""
    scale=np.array([25.,600.])
    observed=features.raw/scale
    available=np.isfinite(observed)
    frame=features.frame.copy()
    frame.loc[~eligible(frame),'occupancy']=np.nan
    occupancy=frame.pivot_table(index='timestamp',columns='link_id',values='occupancy',aggfunc='mean').reindex(
        index=features.times,columns=features.links).to_numpy(float)
    shape=observed.shape[:2]
    dt=pd.to_datetime(features.times,utc=True)
    angle=2*np.pi*(dt.hour*60+dt.minute)/1440
    sin=np.broadcast_to(np.asarray(np.sin(angle))[:,None,None],(*shape,1))
    cos=np.broadcast_to(np.asarray(np.cos(angle))[:,None,None],(*shape,1))
    return np.concatenate([np.nan_to_num(observed),available.astype(float),features.prior/scale,
        features.base/scale,np.nan_to_num(occupancy[...,None])/.25,np.isfinite(occupancy[...,None]),sin,cos],axis=-1).astype(np.float32)


class AdaptiveStateModel(StateModel):
    """Deployable strong-LGB/attention mixture, containing CPU weights."""
    def __init__(self,baseline,state,topology,alpha,hidden=48,attention=True):
        if not 0<=alpha<=1:raise ValueError('Invalid State ensemble fraction')
        self.baseline=baseline
        self.state={k:v.detach().cpu().clone() for k,v in state.items()}
        self.topology=topology
        self.alpha=alpha;self.hidden=hidden;self.attention=attention
        self.profile=baseline.profile;self.feature_version=baseline.feature_version
        self._network=None

    def predict(self,features,template):
        result=self.baseline.predict(features,template)
        net=features.net
        lane=template.link_id.map(net.lanes).to_numpy(float)
        v=result.speed_kmh.to_numpy(float,copy=True)
        q=result.flow_vph.to_numpy(float,copy=True)
        v=np.minimum(v,template.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
        q=np.minimum(q,template.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/lane
        reference=np.column_stack([v,q])
        if self.alpha:
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self._network is None:
                torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark=False
                self._network=AdaptiveGraphState(hidden=self.hidden,heads=2,attention=self.attention).to(device)
                self._network.load_state_dict(self.state);self._network.eval()
            graph=[]
            for direction in ('incoming','outgoing'):
                matrix=np.zeros((len(features.links),len(features.links)),np.float32)
                for j,connected in enumerate(neighbors(self.topology,features.links,direction)):
                    if connected:matrix[j,connected]=1/len(connected)
                graph.append(torch.tensor(matrix,device=device))
            node_lane=net.lanes.reindex(features.links).to_numpy(float)
            fd=np.stack([net.free_speed_kmh.reindex(features.links),net.capacity_vph.reindex(features.links)/node_lane,
                net.k_jam.reindex(features.links)/node_lane]).astype(np.float32)
            x=torch.from_numpy(reconstruction_input(features).transpose(2,0,1)[None].copy()).to(device)
            with torch.no_grad():
                output=self._network(x,*graph,torch.tensor(fd,device=device)).cpu().numpy()[0].transpose(1,2,0)
            output[...,0]=np.clip(output[...,0]*25,0,fd[0][None]*1.05)
            output[...,1]=np.clip(output[...,1]*600,0,fd[1][None]*1.05)
            neural=output.reshape(-1,2)[features.lookup(template)]
            reference=(1-self.alpha)*reference+self.alpha*neural
        reference[:,1]*=lane
        if not np.isfinite(reference).all():raise ValueError('Nonfinite adaptive State prediction')
        result[VALUES]=reference
        return result

    def save(self,path):
        network=self._network
        self._network=None
        try:super().save(path)
        finally:self._network=network
