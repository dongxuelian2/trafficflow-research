"""Same all-six-step objective for dense/event-population warm-up controls."""
import torch


def dense_forecast_loss(logits,speed,labels,truth_speed,available):
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none',pos_weight=torch.tensor(4.,device=logits.device))
    count=available.sum(dim=(1,2)).clamp(min=1);classification=(bce*available).sum(dim=(1,2))/count;regression=((speed-truth_speed).abs()*available).sum(dim=(1,2))/count
    return (classification+.1*regression).mean()
