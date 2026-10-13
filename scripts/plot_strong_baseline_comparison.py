"""Scientific comparison of stronger State controls and Queue proxy uncertainty."""
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read(name):return json.loads((Path('reports/research')/(name+'.json')).read_text(encoding='utf-8'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('reports/research/strong_baseline_comparison.png'))
    output=parser.parse_args().output
    if output.exists():raise FileExistsError('Use a versioned plot instead of overwriting')
    cases=[('I5 north / autumn','A01_graph_state_autumn','A14_full_label_LGB_control','A15_large_LGB_graph_recomparison'),
           ('I5 north / winter','A02_graph_state_winter','A16_winter_full_label_LGB_control','A17_winter_large_LGB_graph_recomparison'),
           ('I405 north / autumn','A12_graph_D7_I405_N','A18_I405_N_large_label_LGB_control','A20_I405_N_large_LGB_graph_recomparison')]
    rows=[]
    for label,old,large,joint in cases:
        old,large,joint=read(old),read(large),read(joint)
        rows.append({'case':label,'300k_LGB':old['baseline_assessment']['S_state'],
                     'graph_with_300k':old['selected_assessment']['S_state'],
                     'full_LGB':large['assessment']['S_state'],
                     'graph_with_full':joint['selected_assessment']['S_state']})
    pd.DataFrame(rows).to_csv(output.with_suffix('.csv'),index=False)
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.7),gridspec_kw={'width_ratios':[1.1,1]})
    for row in rows:
        values=[row[k] for k in ('300k_LGB','graph_with_300k','full_LGB','graph_with_full')]
        ax[0].plot(np.arange(4),values,'o-',label=row['case'],linewidth=1.6,markersize=5)
    ax[0].set_xticks(range(4),['300k LGB','Graph +\n300k LGB','Full LGB','Graph +\nfull LGB'])
    ax[0].set_ylabel('Public-train assessment State')
    ax[0].set_title('Full-label LGB exceeds earlier graph combinations',fontsize=11)
    ax[0].grid(axis='y',alpha=.2)
    ax[0].legend(loc='center left',fontsize=8)
    studies=read('C12_C14_occupancy_uncertainty')
    for i,r in enumerate(studies):
        d=r['occupancy_vs_at_T']
        gain=d['paired_macro_gain']
        lo,hi=d['ci95']
        ax[1].errorbar(gain,i,xerr=[[gain-lo],[hi-gain]],fmt='o',color='#226b8e',capsize=4)
    ax[1].set_yticks(range(3),['I5 north / autumn','I10 west / autumn','I5 north / winter'])
    ax[1].axvline(0,color='#777777',linewidth=1)
    ax[1].set_xlabel('Occupancy proxy IoU gain vs at-T (95% day-block CI)')
    ax[1].set_title('Queue proxy gains vary by corridor',fontsize=11)
    ax[1].grid(axis='x',alpha=.2)
    fig.suptitle('Verified Kaggle best: 0.82940  |  0.95000+ target remains unmet',fontsize=13)
    fig.text(.02,.015,'Development evidence only. Queue labels are noisy observation proxies; formal task components remain unknown.\nState first two stages use fewer training labels. Event origins deduplicated; sparse-event uncertainty is limited.',fontsize=8,color='#444444')
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(output,dpi=170)
    print(output)
