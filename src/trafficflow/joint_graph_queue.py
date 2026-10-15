"""Joint speed/flow dynamics; optional learned residual graph rollout."""
import torch
from torch import nn
from .graph_queue import CausalGraphQueue,queue_event_loss
from .dense_queue_loss import dense_forecast_loss


class JointGraphQueue(CausalGraphQueue):
    def __init__(self,hidden=32,rollout=False):
        super().__init__(hidden=hidden,use_graph=True);self.rollout=bool(rollout)
        self.flow_output=nn.Conv2d(2*hidden,6,1);self.analysis_output=nn.Conv2d(2*hidden,2,1)
        self.initial_latent=nn.Conv1d(2*hidden,hidden,1);self.transition_input=nn.Conv1d(3*hidden+2,hidden,1);self.transition=nn.GRUCell(hidden,hidden);self.transition_output=nn.Conv1d(hidden,3,1)
        for head in [self.flow_output,self.analysis_output,self.transition_output]:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def _encode(self,x,incoming,outgoing,geometry):
        batch,channels,times,nodes=x.shape
        if channels!=9 or times!=13 or incoming.shape!=(nodes,nodes) or outgoing.shape!=(nodes,nodes) or geometry.shape!=(4,nodes):raise ValueError('Joint forecast needs legal13slot geometry contract')
        scale=torch.tensor([25.,600.,25.,25.],device=x.device,dtype=x.dtype)[:,None];z=torch.nn.functional.silu(self.input(x)+self.geometry((geometry/scale)[None,:,None]))
        for block in self.blocks:z=block(z,incoming,outgoing)
        pooled=torch.cat([z[:,:,-1:],z.mean(dim=2,keepdim=True)],dim=1);steps=torch.arange(13,device=x.device)[None,:,None];latest=[]
        for value,mask,fallback in [(0,2,geometry[0]/25.),(1,3,geometry[1]/1200.)]:
            at=torch.where(x[:,mask]>.5,steps,-1).max(dim=1).values;v=torch.gather(x[:,value],1,at.clamp(min=0)[:,None]).squeeze(1);latest.append(torch.where(at>=0,v,fallback[None]))
        return pooled,torch.stack(latest,dim=1)

    def analyze(self,x,incoming,outgoing,geometry):
        pooled,current=self._encode(x,incoming,outgoing,geometry)
        analysis=(current+self.analysis_output(pooled)[:,:,0]).clamp_min(0.);analysis=torch.where(x[:,2:4,-1]>.5,x[:,:2,-1],analysis)
        return analysis

    def forward(self,x,incoming,outgoing,geometry):
        batch,_,_,nodes=x.shape;pooled,current=self._encode(x,incoming,outgoing,geometry)
        residual=self.output(pooled)[:,:,0].reshape(batch,2,6,nodes);logits=2.5*(geometry[3][None,None]/25.-current[:,0,None])+residual[:,0];speed=current[:,0,None]+residual[:,1];flow=(current[:,1,None]+self.flow_output(pooled)[:,:,0]).clamp_min(0.)
        analysis=(current+self.analysis_output(pooled)[:,:,0]).clamp_min(0.);analysis=torch.where(x[:,2:4,-1]>.5,x[:,:2,-1],analysis)
        if self.rollout:
            h=torch.tanh(self.initial_latent(pooled[:,:,0]));state=analysis;ys=[];vs=[];qs=[]
            for lead in range(6):
                up=torch.einsum('bcn,mn->bcm',h,incoming);down=torch.einsum('bcn,mn->bcm',h,outgoing);u=torch.nn.functional.silu(self.transition_input(torch.cat([h,up,down,state],dim=1)))
                h=self.transition(u.transpose(1,2).reshape(batch*nodes,-1),h.transpose(1,2).reshape(batch*nodes,-1)).reshape(batch,nodes,-1).transpose(1,2)
                correction=self.transition_output(h);v=speed[:,lead]+correction[:,1];q=(flow[:,lead]+correction[:,2]).clamp_min(0.);ys.append(logits[:,lead]+correction[:,0]);vs.append(v);qs.append(q);state=torch.stack([v,q],dim=1)
            logits,speed,flow=[torch.stack(values,dim=1) for values in [ys,vs,qs]]
        return logits,speed,flow,analysis


def joint_dynamics_loss(outputs,a,flow_weight=.1,event=False):
    logits,speed,flow,analysis=outputs
    if event:loss=queue_event_loss(logits,speed,a['labels'],a['truth_speed'],a['available'],a['ongoing'])
    else:loss=dense_forecast_loss(logits,speed,a['labels'],a['truth_speed'],a['available'])
    domain=a['current_domain'];current=((analysis-a['truth_current']).abs()*domain).sum(dim=2)/domain.sum(dim=2).clamp_min(1)
    forecast=((flow-a['truth_flow']).abs()*a['available']).sum(dim=(1,2))/a['available'].sum(dim=(1,2)).clamp_min(1)
    return loss+.1*current[:,0].mean()+float(flow_weight)*(forecast+current[:,1]).mean()
