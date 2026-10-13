"""Compact measured comparison tables and scientific plots, with provenance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def strip_runtime(value):
    if isinstance(value,dict):
        return {k:strip_runtime(v) for k,v in value.items() if k not in ('seconds','source_at_start')}
    if isinstance(value,list):
        return [strip_runtime(v) for v in value]
    return value


def main():
    root=Path('reports/research')
    d=json.loads((root/'D02_probabilistic_OD_deterministic.json').read_text())
    repeat=json.loads((root/'D03_OD_independent_process_repeat.json').read_text())
    parity={panel:strip_runtime(value)==strip_runtime(d['panels'][panel]) for panel,value in repeat['panels'].items()}
    if not all(parity.values()):
        raise ValueError('Independent process OD repeat differs beyond runtime fields')
    rows=[]
    for panel,result in d['panels'].items():
        means={m:np.mean([r['simulated_S_ODME'] for r in result['assessment'] if r['method']==m])
               for m in ('prior','ridge_0.05','selected_bayesian')}
        rows.append({'panel':panel,**means,'gain':means['selected_bayesian']-means['ridge_0.05'],
                     'rank':result['public_train_stability']['matrix_rank'],
                     'nullity':result['public_train_stability']['nullity']})
    od=pd.DataFrame(rows)
    od.to_csv(root/'D02_OD_comparison.csv',index=False)
    seed_gains=[]
    for seed in (2909,3253,4001):
        differences=[]
        for result in d['panels'].values():
            for regime in ('independent_paths','hierarchical_demand','prior_total_bias'):
                selected=[r for r in result['assessment'] if r['seed']==seed and r['regime']==regime]
                scores={r['method']:r['simulated_S_ODME'] for r in selected}
                differences.append(scores['selected_bayesian']-scores['ridge_0.05'])
        seed_gains.append({'seed':seed,'mean_simulated_OD_gain':float(np.mean(differences)),
                           'minimum_case_gain':float(min(differences))})
    qrows=[]
    for path in sorted(root.glob('C0*_event_queue.json')):
        result=json.loads(path.read_text())
        means={method:np.mean([r['models'][method]['proxy_IoU'] for r in result['assessment']])
               for method in result['assessment'][0]['models']}
        qrows.append({'panel':result['panel'],'fold':result['fold'],**means})
    queue=pd.DataFrame(qrows)
    queue.to_csv(root/'C01_C03_queue_comparison.csv',index=False)
    b=json.loads((root/'B01_CTM_stress.json').read_text())
    physics={method:{kind:float(np.mean([r['candidates'][method][kind]['simulated_S_physics'] for r in b['rows']]))
                     for kind in ('ctm_flux','projected_observation_flux')}
             for method in b['rows'][0]['candidates']}
    formal=json.loads((root/'formal_scores.json').read_text()) if (root/'formal_scores.json').exists() else None
    summary={'formal_Kaggle_best':formal['best_public_total'] if formal else .82940,'formal_components':None,
        'OD_simulated_means':od[['prior','ridge_0.05','selected_bayesian','gain']].mean().to_dict(),
        'OD_seed_gains':seed_gains,'OD_repeat_numeric_parity':parity,
        'Queue_proxy_means':qrows,'CTM_simulated_physics':physics,
        'uncertainty_limit':'Three synthetic assessment seeds, related public network structures, and hand-chosen generating distributions cannot validate actual hidden OD truth. Queue seed samples overlap.',
        'promotion': 'K01 rejected: formal total 0.76474 vs baseline 0.82940. No Queue promotion from these experiments.'}
    (root/'research_v2_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    axes[0].barh(od.panel,od.gain,color='#287a8b')
    axes[0].set_title('Independent simulated OD gain over Ridge')
    axes[0].set_xlabel('Known-truth simulated OD composite gain')
    axes[0].axvline(0,color='black',linewidth=.7)
    x=np.arange(len(queue))
    for i,(method,color) in enumerate([('old_sampler_joint','#9da3af'),('random_sampler_joint','#287a8b'),('separate_event_models','#d08b35')]):
        axes[1].bar(x+(i-1)*.24,queue[method],width=.24,label=method,color=color)
    axes[1].set_xticks(x,[r['panel']+'\n'+r['fold'].replace('rolling_','') for r in qrows])
    axes[1].set_ylim(0,1)
    axes[1].set_ylabel('Noisy-observation proxy IoU')
    axes[1].set_title('Same-fold Queue sampling/model comparisons')
    axes[1].legend(fontsize=8,loc='lower left')
    fig.suptitle('Offline evidence only: OD candidate failed formal control (0.76474 vs 0.82940)',fontsize=11)
    fig.savefig(root/'research_v2_comparison.png',dpi=160)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
