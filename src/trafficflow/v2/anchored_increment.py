"""Observed inventory-increment proxy on the exact public Task3 pair universe."""
import numpy as np
from .score_aligned import conservation_pairs


def increment_proxy(predicted, truth, topology, mode='A'):
    p = conservation_pairs(predicted, topology, mode)
    t = conservation_pairs(truth, topology, mode)
    key = ['timestamp', 'link_id', 'next_timestamp']
    joined = p[key+['dN']].merge(t[key+['dN']], on=key, validate='one_to_one', suffixes=('_pred', '_true'))
    if len(joined) != len(p) or len(joined) != len(t):
        raise ValueError('Candidate changes official pair coverage; comparison rejected')
    numerator = float(np.abs(joined.dN_pred-joined.dN_true).sum())
    denominator = float(np.abs(joined.dN_true).sum())
    return {'increment_error_L1': numerator, 'increment_truth_L1': denominator,
            'increment_relative_L1': numerator/(denominator+1e-9), 'n_pairs': len(joined),
            'formal_Physics': None}
