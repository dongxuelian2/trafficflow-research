"""In-sample paired-day bootstrap heuristic for calibration model complexity.

Thresholds and network checkpoints have already used these calibration labels.
This is a sensitivity measure, not a valid held-out confidence bound.
"""
import numpy as np
import pandas as pd


def select_with_cal_uncertainty(candidates,calendar,seed=20261013,resamples=4000,block_days=7):
    if not candidates or not calendar or len(set(calendar))!=len(calendar):
        raise ValueError('Unique nonempty calibration calendar and candidates required')
    names=[row['name'] for row in candidates]
    if len(set(names))!=len(names):raise ValueError('Candidate names must be unique')
    first=candidates[0]['metrics']['windows'];keys=[row['window'] for row in first]
    if not first or len(set(keys))!=len(keys):raise ValueError('Unique nonempty windows required')
    conditions=sorted(set(row['condition'] for row in first));n=len(calendar)
    sums=np.zeros((len(candidates),n,len(conditions)));counts=np.zeros((n,len(conditions)))
    for i,row in enumerate(first):
        origin=row['window'].rsplit(':',1)[0]
        day=pd.Timestamp(origin.split(':',1)[1]).date().isoformat()
        d,c=calendar.index(day),conditions.index(row['condition']);counts[d,c]+=1
        for j,candidate in enumerate(candidates):
            windows=candidate['metrics']['windows']
            if len(windows)!=len(first) or windows[i]['window']!=row['window'] or windows[i]['condition']!=row['condition']:
                raise ValueError('Candidate calibration windows/conditions disagree')
            value=float(windows[i]['IoU'])
            if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Finite window IoU in [0,1] required')
            sums[j,d,c]+=value
    means=(sums.sum(axis=1)/counts.sum(axis=0)).mean(axis=1)
    complexity=lambda j:tuple(candidates[j]['complexity'])
    winner=min(range(len(candidates)),key=lambda j:(-means[j],complexity(j),names[j]))
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,(resamples,int(np.ceil(n/block_days))))
    indices=((starts[:,:,None]+np.arange(block_days))%n).reshape(resamples,-1)[:,:n]
    sampled_counts=counts[indices].sum(axis=1);valid=(sampled_counts>0).all(axis=1)
    if valid.sum()<2:raise ValueError('Insufficient calibration bootstrap resamples')
    draws=(sums[:,indices].sum(axis=2)[:,valid]/sampled_counts[valid]).mean(axis=2)
    gaps=draws[winner]-draws
    standard_errors=gaps.std(axis=1,ddof=1)
    within=(means[winner]-means)<=standard_errors+1e-12
    selected=min(np.flatnonzero(within),key=lambda j:(complexity(j),-means[j],names[j]))
    frequencies=np.zeros(len(candidates),int)
    for draw in draws.T:
        pick=min(range(len(candidates)),key=lambda j:(-draw[j],complexity(j),names[j]));frequencies[pick]+=1
    rows=[{'name':names[j],'cal_IoU':float(means[j]),'gap_to_argmax':float(means[winner]-means[j]),
           'paired_gap_bootstrap_standard_error':float(standard_errors[j]),'within_one_gap_SE':bool(within[j]),
           'bootstrap_argmax_frequency':float(frequencies[j]/valid.sum()),'complexity':candidates[j]['complexity']}
          for j in range(len(candidates))]
    return {'argmax':names[winner],'selected':names[selected],'candidates':rows,'valid_resamples':int(valid.sum()),
            'resamples':resamples,'block_days':block_days,'calendar_days':n,'days_with_events':int((counts.sum(axis=1)>0).sum()),
            'scope':'In-sample sensitivity after fitting thresholds/checkpoints; no independent confidence or promotion guarantee.'}
