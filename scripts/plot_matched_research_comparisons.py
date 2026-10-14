"""Scientific figure from completed matched-control evidence only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__=='__main__':
    output=Path('reports/research/matched_state_queue_comparison_v3.png')
    if output.exists():raise FileExistsError('Never overwrite a research figure')
    state=json.loads(Path('reports/research/A29_A31_ten_direction_budget_comparison.json').read_text())
    queue=[json.loads(Path('reports/research',p+'.json').read_text()) for p in
        ('C27_D12_I5_N_onset_specialist','C28_D7_I10_W_onset_specialist','C29_D12_I5_N_winter_onset_specialist')]
    rows=[]
    labels=[];gain=[]
    for family,v in state['families'].items():
        labels.append(family);gain.append(v['gain'])
        rows.append({'comparison':'State full - 500k','label':family,'gain':v['gain'],'ci_low':None,'ci_high':None})
    qlabels=['I5_N autumn','I10_W autumn','I5_N winter']
    qg=[];low=[];high=[]
    for label,r in zip(qlabels,queue):
        c=r['selected_comparison'];qg.append(c['paired_macro_gain']);low.append(c['ci95'][0]);high.append(c['ci95'][1])
        rows.append({'comparison':'Queue selected expert - shared','label':label,'gain':c['paired_macro_gain'],
            'ci_low':c['ci95'][0],'ci_high':c['ci95'][1]})
    pd.DataFrame(rows).to_csv(output.with_suffix('.csv'),index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={'width_ratios':[1,1.35]})
    axes[0].barh(labels,gain,color='#307c9a')
    axes[0].axvline(.002,color='#9c5d12',linestyle='--',label='Global mean-gain gate (reference)')
    axes[0].set(xlabel='Pooled State gain, family mean',title='Full-label gains vary by family',xlim=(0,.003))
    axes[0].legend(loc='lower right',fontsize=8,frameon=False)
    for i,v in enumerate(gain):axes[0].text(v+.00004,i,f'{v:.4f}',va='center',fontsize=9)
    qg=np.asarray(qg);low=np.asarray(low);high=np.asarray(high)
    axes[1].errorbar(qg,np.arange(3),xerr=np.stack([qg-low,high-qg]),fmt='o',color='#307c9a',capsize=4)
    axes[1].axvline(0,color='#555555',linewidth=1)
    axes[1].set(yticks=np.arange(3),yticklabels=qlabels,xlabel='Origin-macro observation-proxy IoU gain',
        title='Selected Queue experts lack conclusive gains',xlim=(-.055,.12))
    axes[1].invert_yaxis()
    fig.suptitle('Matched public-train development controls | Formal best: 0.82940',fontsize=13)
    fig.text(.5,.015,'State: five-family weights. Queue: unique origins, calibration-selected experts, 95% paired day-block intervals.\nNo formal component score or Kaggle gain is implied.',ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.085,1,.93))
    fig.savefig(output,dpi=170)
    print(output)
