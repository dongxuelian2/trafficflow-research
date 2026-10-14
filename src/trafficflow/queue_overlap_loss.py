"""Window-level Jaccard surrogates for TRAIN evaluator labels only.

Lovasz hinge follows the set-extension equations of Berman et al. (2018),
https://arxiv.org/html/1705.08790 . This implementation is independently written.
Availability here is a training loss mask, never a forecasting decision input.
"""
import math
import torch
from .graph_queue import queue_event_loss


def lovasz_binary_hinge(logits, labels):
    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError('A flat, matching binary foreground is required')
    if not len(logits):
        return logits.sum() * 0.
    signs = 2 * labels - 1
    errors = 1 - logits * signs
    ordered, permutation = torch.sort(errors, descending=True, stable=True)
    truth = labels[permutation]
    positives = truth.sum()
    intersection = positives - truth.cumsum(0)
    union = positives + (1 - truth).cumsum(0)
    loss_at_prefix = 1 - intersection / union.clamp(min=1)
    increments = torch.cat([loss_at_prefix[:1], loss_at_prefix[1:] - loss_at_prefix[:-1]])
    return torch.dot(torch.relu(ordered), increments)


def queue_window_overlap(logits, labels, available, ongoing, kind):
    if logits.shape != labels.shape or logits.shape != available.shape or logits.ndim != 3:
        raise ValueError('Expected matching batch/horizon/node arrays')
    if logits.shape[1] != 6 or ongoing.shape != (len(logits),):
        raise ValueError('Six horizons and one condition per window required')
    domain = available.clone()
    domain[:, :5] &= ongoing[:, None, None]
    corrected = logits - math.log(4.)
    losses = []
    for score, truth, use in zip(corrected, labels, domain):
        score, truth = score[use], truth[use]
        if not len(score):
            losses.append(score.sum() * 0.)
        elif kind == 'lovasz':
            losses.append(lovasz_binary_hinge(score, truth))
        elif kind == 'soft_iou':
            p = score.sigmoid()
            if truth.sum() == 0:
                # Explicit empty-foreground extension, not an IoU identity.
                losses.append(p.mean())
            else:
                intersection = (p * truth).sum()
                union = truth.sum() + (p * (1 - truth)).sum()
                losses.append(1 - intersection / union.clamp(min=1e-8))
        else:
            raise ValueError('Unknown window-overlap surrogate')
    return torch.stack(losses).mean()


def queue_overlap_training_loss(logits, speed, labels, truth_speed, available, ongoing,
                                kind='bce', strength=.25):
    base = queue_event_loss(logits, speed, labels, truth_speed, available, ongoing)
    if kind == 'bce':
        return base
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Finite nonnegative overlap strength required')
    return base + strength * queue_window_overlap(logits, labels, available, ongoing, kind)
