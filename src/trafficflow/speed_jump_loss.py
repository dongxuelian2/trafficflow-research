"""Auxiliary speed-difference loss on existing adjacent supervised cells only."""
import torch
from .adaptive_graph_state import patch_score_loss


def speed_jump_pair_loss(prediction,truth,target):
    change=truth[:,0,1:]-truth[:,0,:-1]
    pair=target[:,1:]&target[:,:-1]&(change.abs()>=.8)
    error=(prediction[:,0,1:]-prediction[:,0,:-1])-change
    count=pair.sum(dim=(1,2));mean=(error.square()*pair).sum(dim=(1,2))/count.clamp(min=1)
    loss=torch.where(count>0,torch.sqrt(mean+1e-8),torch.zeros_like(mean))
    return loss.mean()


def speed_jump_objective(prediction,truth,target,coefficient=.05):
    return patch_score_loss(prediction,truth,target)+coefficient*speed_jump_pair_loss(prediction,truth,target)
