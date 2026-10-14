"""Forecast on the full public domain; future eligibility belongs to scoring."""
import numpy as np
import pandas as pd
from .metrics import queue_iou


def predict_full_forecast(model, x, window, condition):
    """No future labels or future score mask are accepted by this API."""
    window, condition = np.asarray(window), np.asarray(condition)
    if window.shape != (len(x),) or condition.shape != window.shape:
        raise ValueError('Full Queue forecast domain shape mismatch')
    prediction = np.asarray(model.classify(x, window, condition))
    if prediction.shape != window.shape or not np.isin(prediction,[0,1]).all():
        raise ValueError('Full forecast must cover every public cell with a binary decision')
    return prediction.astype(int)


def score_full_forecast(labels, prediction, window, condition, score_mask):
    """Apply evaluator-only eligibility after all window set decisions exist."""
    labels, prediction, window, condition, score_mask = map(np.asarray,
        [labels, prediction, window, condition, score_mask])
    if labels.ndim != 1 or any(a.shape != labels.shape for a in [prediction,window,condition,score_mask]):
        raise ValueError('Full Queue scoring arrays disagree')
    rows=[]
    for key,indices in pd.DataFrame({'window':window}).groupby('window',sort=True).indices.items():
        conditions=np.unique(condition[indices])
        if len(conditions)!=1:raise ValueError('Each forecast window needs one public condition')
        use=indices[score_mask[indices].astype(bool)]
        value=queue_iou(labels[use],prediction[use]) if len(use) else 0.
        rows.append({'window':key,'condition':conditions[0],'IoU':float(value)})
    frame=pd.DataFrame(rows)
    by_condition=frame.groupby('condition').IoU.mean().to_dict() if len(frame) else {}
    return {'proxy_IoU':float(np.mean(list(by_condition.values()))) if by_condition else 0.,
            'conditions':by_condition,'windows':rows}
