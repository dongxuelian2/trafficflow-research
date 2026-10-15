"""Static relative-milepost relation for global attention,not physical edges."""
import numpy as np
import torch
from torch import nn
from .adaptive_graph_state import AdaptiveSpatialBlock
from .recurrent_state import make_network,WindowedStateModel


def relative_distance(positions):
    p=np.asarray(positions,float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all():raise ValueError('Finite one-dimensional public positions required')
    gaps=np.diff(np.unique(p));scale=float(np.median(gaps)) if len(gaps) else 1.
    return np.minimum(np.abs(p[:,None]-p[None,:])/scale,20.).astype(np.float32)


class RelativeSpatialBlock(AdaptiveSpatialBlock):
    def __init__(self,hidden,heads,distance):
        super().__init__(hidden,heads)
        self.distance_strength=nn.Parameter(torch.tensor(0.))
        self.register_buffer('distance_matrix',torch.as_tensor(distance,dtype=torch.float32).clone())

    def forward(self,z,incoming,outgoing):
        batch,channels,times,nodes=z.shape
        if self.distance_matrix.shape!=(nodes,nodes):raise ValueError('Relative position grid mismatch')
        token=z.permute(0,2,3,1).reshape(batch*times,nodes,channels);normalized=self.norm(token)
        bias=(self.relation_bias[0]*(incoming>0).to(z.dtype)+self.relation_bias[1]*(outgoing>0).to(z.dtype)+
            self.relation_bias[2]*torch.eye(nodes,device=z.device,dtype=z.dtype)-self.distance_strength*self.distance_matrix.to(z.dtype))
        attended,_=self.attention(normalized,normalized,normalized,attn_mask=bias,need_weights=False)
        token=token+self.gain*attended
        return token.reshape(batch,times,nodes,channels).permute(0,3,1,2).contiguous()


def make_relative_network(positions,hidden=48):
    model=make_network('tcn_gru',hidden);distance=relative_distance(positions)
    for i,old in enumerate(model.spatial):
        new=RelativeSpatialBlock(hidden,2,distance);status=new.load_state_dict(old.state_dict(),strict=False)
        if status.unexpected_keys or set(status.missing_keys)!={'distance_strength','distance_matrix'}:raise ValueError('Shared attention initialization changed')
        model.spatial[i]=new
    return model


def common_initial_sha256(model):
    import hashlib
    return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items()
        if not k.endswith(('.distance_strength','.distance_matrix')))).hexdigest()


class RelativePositionStateModel(WindowedStateModel):
    def __init__(self,baseline,state,topology,alpha,link_order,positions,hidden=48,context=128,core=64):
        super().__init__(baseline,state,topology,alpha,'tcn_gru',hidden,context,core)
        self.link_order=list(link_order);self.positions=np.asarray(positions,float).copy();expected=relative_distance(self.positions)
        if len(self.link_order)!=len(self.positions) or len(set(self.link_order))!=len(self.link_order):raise ValueError('Unique ordered public link geometry required')
        for i in range(2):
            if not np.array_equal(self.state[f'spatial.{i}.distance_matrix'].numpy(),expected):raise ValueError('Saved relative geometry differs')

    def predict(self,features,template):
        if features.links!=self.link_order:raise ValueError('Relative geometry requires exact link order')
        if self.alpha and self._network is None:
            torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._network=make_relative_network(self.positions,self.hidden).to(device);self._network.load_state_dict(self.state);self._network.eval()
        return super().predict(features,template)
