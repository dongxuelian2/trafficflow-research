"""Display ordinary-template evidence and synthetic-outage risk separately."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from trafficflow.runtime import sha256_file


def main():
    output=Path('reports/research/A85_recurrent_evidence.png');record=output.with_suffix('.json')
    if output.exists() or record.exists():raise FileExistsError('Never overwrite evidence figure')
    paths=[Path('reports/research/A72b_recurrent_I210_W_autumn_full_day.json'),Path('reports/research/A73_recurrent_I5_N_winter.json'),
        Path('reports/research/A77c_recurrent_forward_calendar_v3.json'),Path('reports/research/A81b_I210_W_public_outage_stress.json')]
    first,second,forward,stress=[json.loads(p.read_text()) for p in paths]
    state=[]
    for study,label in [(first,'I210 W | autumn, 14 d'),(second,'I5 N | winter, 14 d')]:
        row=study['comparisons']['tcn_gru_seed_mean'];state.append((label,row['paired_days_vs_incumbent'],row['gain_vs_incumbent']))
    for run in forward['studies']:
        row=run['comparisons']['primary_export'];state.append((run['panel'].split('_',1)[1].replace('_',' ')+' | February, 28 d',
            row['paired_days_vs_incumbent'],row['gain_vs_incumbent']))
    outage=[(name.replace('min_',' min / ').replace('pct','%'),rows['tcn_gru_seed_mean']) for name,rows in stress['paired_day_diagnostics'].items()]
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for i,(label,row,pooled) in enumerate(state):
        mean=row['paired_day_mean_gain'];lo,hi=row['ci95']
        axes[0].errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o' if i<2 else 's',capsize=4,color='#087e8b')
        axes[0].annotate(f'pooled {pooled:+.5f}',(hi,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
    axes[0].set_yticks(range(len(state)),[r[0] for r in state]);axes[0].set_title('Released Task 1 templates | frozen policies')
    axes[0].set_xlabel('Paired day-mean State difference vs incumbent')
    axes[0].set_xlim(-.0002,.0045)
    for i,(label,row) in enumerate(outage):
        mean=row['paired_day_mean_gain'];lo,hi=row['ci95'];color='#087e8b' if mean>=0 else '#c25139'
        axes[1].errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',capsize=4,color=color)
    axes[1].set_yticks(range(len(outage)),[r[0] for r in outage]);axes[1].set_title('Own additional sensor outages | different targets')
    axes[1].set_xlabel('Paired day-mean synthetic score difference')
    for axis in axes:
        axis.axvline(0,color='#444',linewidth=.8);axis.grid(axis='x',alpha=.18);axis.invert_yaxis()
    fig.suptitle('Public-train research: regular gains and severe-outage regressions',fontsize=12)
    fig.text(.5,-.03,'95% conditional intervals: 7-day date blocks; comparisons unadjusted. Formal Kaggle Total remains 0.82940.',
        ha='center',fontsize=9)
    fig.savefig(output,dpi=180,bbox_inches='tight');plt.close(fig)
    with record.open('x',encoding='utf-8') as stream:json.dump({'source_reports_sha256':{p.as_posix():sha256_file(p) for p in paths},
        'figure_sha256':sha256_file(output),'formal_State':None,'formal_Total':None,
        'scope':'Two different metric/population panels;visual summary only,no new experiment or model selection'},stream,indent=2,allow_nan=False)
    print('figure',output.as_posix(),'sha256',sha256_file(output),flush=True)


if __name__=='__main__':main()
