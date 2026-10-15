"""Bidirectional latent GRU branch and fixed-context offline State inference.

This extends the audited graph/attention encoder,not a traffic-physics model.
Only released masked observations/profile/base features enter the network.
"""
import numpy as np
import torch
from torch import nn
from .adaptive_graph_state import AdaptiveGraphState,reconstruction_input
from .state import StateModel,neighbors
from .data import VALUES


class RecurrentGraphState(AdaptiveGraphState):
    def __init__(self,channels=12,hidden=48,heads=2):
        super().__init__(channels,hidden,heads,attention=True)
        self.temporal=nn.GRU(hidden,hidden//2,num_layers=2,bidirectional=True,batch_first=True,dropout=0.)
        self.temporal_gain=nn.Parameter(torch.tensor(.1))

    def forward(self,x,incoming,outgoing,fd):
        nodes=x.shape[-1]
        if incoming.shape!=(nodes,nodes) or outgoing.shape!=(nodes,nodes) or fd.shape!=(3,nodes):
            raise ValueError('Public graph geometry does not match observed grid')
        scale=torch.tensor([25.,600.,25.],device=x.device,dtype=x.dtype)[:,None]
        z=torch.nn.functional.silu(self.input(x)+self.geometry((fd/scale)[None,:,None,:]))
        for i,block in enumerate(self.blocks):
            z=block(z,incoming,outgoing)
            if i in (1,3):z=self.spatial[i//2](z,incoming,outgoing)
        batch,channels,times,nodes=z.shape
        sequence=z.permute(0,3,2,1).reshape(batch*nodes,times,channels)
        latent,_=self.temporal(sequence)
        latent=latent.reshape(batch,nodes,times,channels).permute(0,3,2,1).contiguous()
        return x[:,6:8]+self.output(z+self.temporal_gain*latent)


def make_network(kind,hidden=48):
    if kind=='tcn':return AdaptiveGraphState(hidden=hidden,heads=2,attention=True)
    if kind=='tcn_gru':return RecurrentGraphState(hidden=hidden,heads=2)
    raise ValueError('Unknown State temporal architecture')


def window_plan(length,context=128,core=64):
    if length<1 or context<core or core<1 or (context-core)%2:raise ValueError('Invalid fixed context/core grid')
    width=min(length,context);margin=(context-core)//2
    return [(min(max(start-margin,0),max(length-width,0)),start,min(start+core,length))
        for start in range(0,length,core)]


def training_window(length,center,context=128,core=64):
    """Choose supervision first,then clip its context so day edges can train."""
    if length<1 or not 0<=center<length or context<core or core<1 or (context-core)%2:
        raise ValueError('Invalid training window')
    width=min(length,context);supervision=min(length,core)
    start=min(max(center-supervision//2,0),length-supervision)
    left=min(max(start-(context-core)//2,0),length-width)
    return left,start,start+supervision


def windowed_prediction(model,inputs,graph,device,context=128,core=64):
    """Return every grid cell once;finite windows never inspect evaluator masks."""
    x=np.asarray(inputs,np.float32)
    if x.ndim!=3 or x.shape[-1]!=12 or not np.isfinite(x).all():raise ValueError('Finite strict12 input required')
    plan=window_plan(len(x),context,core);width=min(context,len(x))
    batch=np.stack([x[left:left+width].transpose(2,0,1) for left,start,end in plan])
    with torch.no_grad():prediction=model(torch.from_numpy(batch.copy()).to(device),*graph).cpu().numpy().transpose(0,2,3,1)
    result=np.empty((*x.shape[:2],2),np.float32)
    for i,(left,start,end) in enumerate(plan):result[start:end]=prediction[i,start-left:end-left]
    return result


class WindowedStateModel(StateModel):
    def __init__(self,baseline,state,topology,alpha,kind,hidden=48,context=128,core=64):
        if not 0<=alpha<=1:raise ValueError('Invalid State blend')
        window_plan(context,context,core)
        if kind not in {'tcn','tcn_gru'}:raise ValueError('Unknown State architecture')
        self.baseline=baseline;self.state={k:v.detach().cpu().clone() for k,v in state.items()}
        self.topology=topology;self.alpha=alpha;self.kind=kind;self.hidden=hidden;self.context=context;self.core=core
        self.profile=baseline.profile;self.feature_version=baseline.feature_version;self._network=None

    def predict(self,features,template):
        result=self.baseline.predict(features,template);net=features.net
        lane=template.link_id.map(net.lanes).to_numpy(float)
        v=np.minimum(result.speed_kmh.to_numpy(float),template.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
        q=np.minimum(result.flow_vph.to_numpy(float),template.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/lane
        values=np.column_stack([v,q])
        if self.alpha:
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self._network is None:
                torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
                self._network=make_network(self.kind,self.hidden).to(device);self._network.load_state_dict(self.state);self._network.eval()
            matrices=[]
            for direction in ['incoming','outgoing']:
                a=np.zeros((len(features.links),len(features.links)),np.float32)
                for j,connected in enumerate(neighbors(self.topology,features.links,direction)):
                    if connected:a[j,connected]=1/len(connected)
                matrices.append(torch.tensor(a,device=device))
            lanes=net.lanes.reindex(features.links).to_numpy(float)
            fd=np.stack([net.free_speed_kmh.reindex(features.links),net.capacity_vph.reindex(features.links)/lanes,
                net.k_jam.reindex(features.links)/lanes]).astype(np.float32)
            graph=matrices+[torch.tensor(fd,device=device)]
            grid=windowed_prediction(self._network,reconstruction_input(features),graph,device,self.context,self.core)
            grid[...,0]=np.clip(grid[...,0]*25,0,fd[0][None]*1.05)
            grid[...,1]=np.clip(grid[...,1]*600,0,fd[1][None]*1.05)
            neural=grid.reshape(-1,2)[features.lookup(template)]
            values=(1-self.alpha)*values+self.alpha*neural
        values[:,1]*=lane
        if not np.isfinite(values).all():raise ValueError('Nonfinite windowed State prediction')
        result[VALUES]=values
        return result

    def save(self,path):
        network=self._network;self._network=None
        try:super().save(path)
        finally:self._network=network
