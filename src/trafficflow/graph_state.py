"""Small missing-aware temporal convolution and directed graph diffusion model."""
import torch
from torch import nn


class GraphBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.temporal = nn.Conv2d(channels, channels, (3,1), padding=(dilation,0), dilation=(dilation,1))
        self.mix = nn.Conv2d(channels*3, channels, 1)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, x, incoming, outgoing):
        z = torch.nn.functional.silu(self.temporal(x))
        up = torch.einsum('bctn,mn->bctm', z, incoming)
        down = torch.einsum('bctn,mn->bctm', z, outgoing)
        return x + torch.nn.functional.silu(self.norm(self.mix(torch.cat([z,up,down],1))))


class GraphState(nn.Module):
    def __init__(self, channels=12, hidden=32, graph=True):
        super().__init__()
        self.graph = graph
        self.input = nn.Conv2d(channels, hidden, 1)
        self.blocks = nn.ModuleList([GraphBlock(hidden, d) for d in (1,2,4,8)])
        self.output = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, incoming, outgoing):
        z = torch.nn.functional.silu(self.input(x))
        if not self.graph:
            incoming, outgoing = torch.zeros_like(incoming), torch.zeros_like(outgoing)
        for block in self.blocks:
            z = block(z, incoming, outgoing)
        return x[:,6:8] + self.output(z)


def data_loss(prediction, truth, target):
    weight = torch.tensor([.54,.46],device=prediction.device)[None,:,None,None]
    error = (prediction-truth).square()*weight
    return (error*target[:,None]).sum()/target.sum().clamp(min=1)


def fd_loss(prediction, target, fd):
    v = prediction[:,0].clamp(min=.04)*25
    q = prediction[:,1].clamp(min=0)*600
    vf, capacity, jam = [p[None,None,:] for p in fd]
    k = q/v
    critical = capacity/vf
    wave = capacity/(jam-critical).clamp(min=1)
    qfd = torch.minimum(vf*k, wave*(jam-k).clamp(min=0))
    return (((q-qfd)/600).square()*target).sum()/target.sum().clamp(min=1)
