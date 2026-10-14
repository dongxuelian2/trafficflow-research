"""Aggregate ten matched direction controls with the official family weights."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.state import StateModel
from trafficflow.experiments import evaluate_state, partition_date
from trafficflow.research_eval import paired_block_interval
from trafficflow.runtime import sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite aggregate evidence')
    start=time.perf_counter()
    release=Release('data/kaggle_public')
    rows=[];day_rows=[];sources=[]
    for path in sorted(Path('reports/research').glob('A*_State_budget_*_D*.json')):
        if path.name.endswith(('.registered.json','.selection.json')):continue
        report=json.loads(path.read_text())
        if report.get('registered',{}).get('protocol',{}).get('id')!='State_500k_vs_full_suite':continue
        if report['registered']['protocol']['fold']['name']!='rolling_autumn':continue
        if report['panel']=='D12_I5_N':continue
        panel=report['panel']
        old,new=report['assessment']['500k'],report['assessment']['full']
        rows.append({'panel':panel,'family':panel.rsplit('_',1)[0],
            '500k_State':old['S_state'],'full_State':new['S_state'],
            'gain':new['S_state']-old['S_state'],'500k_labels':report['fitting_labels']['500k'],
            'full_labels':report['fitting_labels']['full']})
        for a,b in zip(old['daily_sufficient_statistics'],new['daily_sufficient_statistics']):
            if (a['date'],a['regime'],a['n'])!=(b['date'],b['regime'],b['n']):raise ValueError('Unmatched daily target groups')
            day_rows.append({'panel':panel,'family':panel.rsplit('_',1)[0],
                'date':a['date'],'old':a['daily_State'],'new':b['daily_State']})
        sources.append({'report':path.as_posix(),'sha256':sha256_file(path)})
    # Older I5_N reports lack daily sufficient statistics; regenerate only that
    # diagnostic from preserved models, never changing the registered model.
    panel='D12_I5_N'
    models=[StateModel.load(Path('artifacts/research')/n/'lightgbm_large_budget.pkl') for n in
        ('A24_autumn_500k_LGB_control','A14_full_label_LGB_control')]
    cache=json.loads(Path('artifacts/research/A01_D12_I5_N_autumn_seed20261013/prepared.json').read_text())
    paths=[p for p in release.files(panel,'train') if partition_date(release,p) in cache['dates']['assessment']]
    scored=[evaluate_state(release,panel,paths,m,0.,True) for m in models]
    old,new=scored
    counts=[json.loads(Path('reports/research',n+'.json').read_text())['n_fitting_labels'] for n in
        ('A24_autumn_500k_LGB_control','A14_full_label_LGB_control')]
    rows.append({'panel':panel,'family':panel.rsplit('_',1)[0],
        '500k_State':old['S_state'],'full_State':new['S_state'],'gain':new['S_state']-old['S_state'],
        '500k_labels':counts[0],'full_labels':counts[1]})
    for a,b in zip(old['daily_sufficient_statistics'],new['daily_sufficient_statistics']):
        if (a['date'],a['regime'],a['n'])!=(b['date'],b['regime'],b['n']):raise ValueError('Unmatched I5_N groups')
        day_rows.append({'panel':panel,'family':panel.rsplit('_',1)[0],
            'date':a['date'],'old':a['daily_State'],'new':b['daily_State']})
    for n in ('A24_autumn_500k_LGB_control','A14_full_label_LGB_control'):
        p=Path('artifacts/research')/n/'lightgbm_large_budget.pkl'
        sources.append({'model':p.as_posix(),'sha256':sha256_file(p)})
    frame=pd.DataFrame(rows).sort_values('panel')
    if frame.panel.duplicated().any() or set(frame.panel)!=set(release.panels()):
        raise ValueError('Exactly one matched control is required for all ten directions')
    days=pd.DataFrame(day_rows)
    if days.duplicated(['panel','date']).any():raise ValueError('Duplicate daily records')
    dates=set(cache['dates']['assessment'])
    if not all(set(g.date)==dates for _,g in days.groupby('panel')):raise ValueError('Direction calendars differ')
    families=frame.groupby('family')[['500k_State','full_State','gain']].mean()
    macro=families.mean().to_dict()
    # Same calendar is shared by all directions. Resample a date block jointly
    # across families, rather than claiming 10*14 independent assessment days.
    by_day=days.groupby(['date','family'])[['old','new']].mean().groupby('date').mean().sort_index()
    interval=paired_block_interval(by_day.old,by_day.new)
    result={'fold':'rolling_autumn','fit_seed':20261013,'directions':frame.to_dict('records'),
        'families':families.to_dict('index'),'family_macro_State':macro,
        'paired_day_interval_after_family_mean':interval,'assessment_dates':sorted(dates),
        'sources':sources,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'global_minimum_gain_gate_passed':macro['gain']>=.002,
        'multiple_fit_seeds_all_directions':False,'seconds':time.perf_counter()-start,
        'limitation':'Matched ten-direction public-train development scores. Daily diagnostic CI is not a CI for nonlinear pooled State or Kaggle Total. Only I5_N has three fitting seeds.'}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    frame.to_csv(args.output.with_suffix('.csv'),index=False)
    print(macro,interval,flush=True)


if __name__=='__main__':main()
