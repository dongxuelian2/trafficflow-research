"""Paired, origin-deduplicated day-block uncertainty for fixed Queue studies."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def comparison(report,baseline,candidate,seed=20261013,resamples=4000,block_days=7):
    rows=[]
    for run in report['assessment']:
        left=run['models'][baseline]['windows']
        right={r['window']:r for r in run['models'][candidate]['windows']}
        if set(right)!=set(r['window'] for r in left):
            raise ValueError('Candidate windows differ from baseline')
        for row in left:
            other=right[row['window']]
            if row['condition']!=other['condition']:
                raise ValueError('Paired condition mismatch')
            origin=row['window'].rsplit(':',1)[0]
            date=pd.Timestamp(origin.split(':',1)[1]).date().isoformat()
            rows.append({'origin':origin,'date':date,'condition':row['condition'],
                         'baseline':row['IoU'],'candidate':other['IoU']})
    frame=pd.DataFrame(rows)
    for _,group in frame.groupby('origin'):
        if group.condition.nunique()!=1 or not np.allclose(group[['baseline','candidate']],group[['baseline','candidate']].iloc[0],atol=1e-12,rtol=0):
            raise ValueError('Repeated origins have inconsistent evidence')
    unique=frame.drop_duplicates('origin')
    dates=report['dates']['assessment']
    conditions=sorted(unique.condition.unique())
    arrays={k:np.zeros((len(dates),len(conditions))) for k in ('baseline','candidate','count')}
    for row in unique.itertuples(index=False):
        d,c=dates.index(row.date),conditions.index(row.condition)
        arrays['baseline'][d,c]+=row.baseline
        arrays['candidate'][d,c]+=row.candidate
        arrays['count'][d,c]+=1
    n=len(dates)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,(resamples,int(np.ceil(n/block_days))))
    indices=((starts[:,:,None]+np.arange(block_days))%n).reshape(resamples,-1)[:,:n]
    counts=arrays['count'][indices].sum(axis=1)
    valid=(counts>0).all(axis=1)
    diff=(arrays['candidate'][indices]-arrays['baseline'][indices]).sum(axis=1)
    samples=np.mean(diff[valid]/counts[valid],axis=1)
    means=unique.groupby('condition')[['baseline','candidate']].mean().mean()
    return {'baseline':baseline,'candidate':candidate,
            'origin_deduplicated_condition_macro_IoU':means.to_dict(),
            'paired_macro_gain':float(means.candidate-means.baseline),
            'ci95':np.quantile(samples,[.025,.975]).tolist() if len(samples) else None,
            'bootstrap_probability_gain_positive':float(np.mean(samples>0)) if len(samples) else None,
            'n_sampled_events':len(frame),'n_unique_origins':len(unique),
            'unique_condition_counts':unique.condition.value_counts().to_dict(),
            'n_assessment_days':n,'n_days_with_events':int(unique.date.nunique()),
            'block_days':block_days,'resamples':resamples,'valid_resamples':int(valid.sum()),
            'limitation':'Observed-proxy event sampling only. Days with zero events remain in bootstrap. Draws missing a condition are excluded; sparse event sets make this interval weak. No formal Queue score.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reports',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite uncertainty evidence')
    results=[]
    for path in args.reports:
        report=json.loads(path.read_text(encoding='utf-8'))
        score=lambda name:report['selection'][name]['metrics']['proxy_IoU']
        names=report['selection']
        baseline=max((n for n in names if n.startswith('before_T_') and not n.startswith('before_T_latest')),key=score)
        at_t=max((n for n in names if n.startswith('at_T_')),key=score)
        results.append({'report':path.as_posix(),'model_choice_source':'selection only; assessment choices unchanged',
            'at_T_vs_before_T':comparison(report,baseline,at_t),
            'selected_vs_before_T':comparison(report,baseline,report['selected'])})
    args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    for r in results:print(r['report'],r['at_T_vs_before_T']['paired_macro_gain'],r['at_T_vs_before_T']['ci95'],r['at_T_vs_before_T']['n_unique_origins'])


if __name__=='__main__':main()
