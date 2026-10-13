"""Score contracts and explicit separation of unavailable metrics."""
import numpy as np
import pandas as pd
from .data import STATE_KEY, VALUES, check_keys

WEIGHTS = {'state':.35, 'queue':.30, 'physics':.15, 'odme':.20}

def total_score(scores: dict) -> dict:
    for k, v in scores.items():
        if k in WEIGHTS and v is not None and (not np.isfinite(v) or not 0 <= v <= 1):
            raise ValueError(f'Invalid score: {k}')
    known = sum(w*scores[k] for k,w in WEIGHTS.items() if scores.get(k) is not None)
    unknown = [k for k in WEIGHTS if scores.get(k) is None]
    return {'S_total':None if unknown else known, 'known_weighted_contribution':known,
            'unknown_components':unknown, 'bounds':[known, known+sum(WEIGHTS[k] for k in unknown)]}

def state_score(truth: pd.DataFrame, prediction: pd.DataFrame, lanes: np.ndarray) -> dict:
    check_keys(truth, STATE_KEY, 'state truth')
    check_keys(prediction, STATE_KEY, 'state prediction')
    if len(truth) != len(lanes) or not np.isfinite(lanes).all() or (lanes <= 0).any():
        raise ValueError('Invalid lane vector')
    joined = truth[STATE_KEY+VALUES].merge(prediction[STATE_KEY+VALUES], on=STATE_KEY,
               how='left', suffixes=('_true','_pred'), validate='one_to_one', sort=False)
    joined['lanes'] = lanes
    details = []
    for panel in truth.panel.unique():
        for regime in ['R1','R2','R3']:
            d = joined[(joined.panel == panel)&(joined.mask_regime == regime)]
            if not len(d):
                details.append({'panel':panel,'regime':regime,'n':0,'S_state':0.0})
                continue
            true_v = d.speed_kmh_true.to_numpy(float)
            true_q = d.flow_vph_true.to_numpy(float)
            if not np.isfinite(true_v).all() or not np.isfinite(true_q).all():
                raise ValueError('Non-finite eligible truth')
            pred_v = pd.to_numeric(d.speed_kmh_pred, errors='coerce').to_numpy(float)
            pred_q = pd.to_numeric(d.flow_vph_pred, errors='coerce').to_numpy(float)
            missing = (~np.isfinite(pred_v)|~np.isfinite(pred_q)).sum()
            pred_v = np.where(np.isfinite(pred_v), pred_v, 0.)
            pred_q = np.where(np.isfinite(pred_q), pred_q, 0.)
            rv = float(np.sqrt(np.mean((pred_v-true_v)**2)))
            rq = float(np.sqrt(np.mean(((pred_q-true_q)/d.lanes.to_numpy())**2)))
            s = .54*max(0,1-rv/25)+.46*max(0,1-rq/600)
            details.append({'panel':panel,'regime':regime,'n':len(d),'missing':int(missing),
                            'rmse_speed_kmh':rv,'rmse_flow_per_lane':rq,'S_state':s})
    df = pd.DataFrame(details)
    panels = df.groupby('panel').S_state.mean()
    families = panels.groupby(panels.index.map(lambda s:s.rsplit('_',1)[0])).mean()
    return {'S_state':float(families.mean()) if len(families) else 0.,'details':details,
            'panels':panels.to_dict(),'families':families.to_dict()}

def queue_iou(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    if y.shape != p.shape:
        raise ValueError('Queue prediction shape mismatch')
    union = (y|p).sum()
    return float((y&p).sum()/union) if union else 1.

def grouped_iou(labels: np.ndarray, pred: np.ndarray, window: np.ndarray, condition: np.ndarray) -> dict:
    rows = []
    for wid in np.unique(window):
        sel = window == wid
        rows.append({'window':str(wid), 'condition':str(condition[sel][0]),
                     'IoU':queue_iou(labels[sel], pred[sel])})
    d = pd.DataFrame(rows)
    value = d.groupby('condition').IoU.mean().mean() if len(d) else 0.
    return {'proxy_IoU':float(value), 'conditions':d.groupby('condition').IoU.mean().to_dict() if len(d) else {},
            'windows':rows}

def od_link_score(A, f, c):
    residual = np.asarray(A@f-c)
    return max(0., 1.-float(np.abs(residual).sum()/max(np.abs(c).sum(),1e-9)))
