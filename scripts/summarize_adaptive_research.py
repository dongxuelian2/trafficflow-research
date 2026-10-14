"""Summarize matched neural studies, preserving calibration-selected models."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trafficflow.research_eval import paired_block_interval
from trafficflow.runtime import sha256_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', action='append', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a research summary')
    result = {'formal_best_Total': .82940, 'formal_target_Total': .95000, 'studies': {}}
    for name in args.study:
        path = Path('reports/research')/(name+'.json')
        ablation_path = path.with_suffix('.ablation.json')
        r = json.loads(path.read_text())
        ablation = json.loads(ablation_path.read_text())
        if not ablation['original_selection_preserved']:
            raise ValueError('Ablation changed original selection')
        baseline = r['baseline_assessment']
        selected = r['selected_assessment']
        architectures = ablation['architectures']
        attention = architectures['adaptive_attention']['metrics']
        density = architectures['adaptive_attention_density']['metrics']
        density_pair = paired_block_interval([d['daily_State'] for d in attention['days']],
            [d['daily_State'] for d in density['days']])
        row = {'panel': r['registered']['protocol']['panel'], 'fold': r['registered']['protocol']['fold'],
            'seed': r['registered']['protocol']['seed'], 'selected': r['selected'],
            'baseline_State': baseline['S_state'], 'selected_State': selected['S_state'],
            'State_gain': selected['S_state']-baseline['S_state'], 'paired_day_gain': r['paired_daily_interval'],
            'density_increment_proxy_baseline': baseline['density_target_pair_relative_L1'],
            'density_increment_proxy_selected': selected['density_target_pair_relative_L1'],
            'calibrated_architecture_comparison': {k: {'alpha': v['sealed_choice']['alpha'], 'State': v['metrics']['S_state'],
                'vs_geometry_day_interval': v['paired_daily_vs_geometry_graph']} for k, v in architectures.items()},
            'density_vs_attention_day_interval': density_pair,
            'regime_gain': {a['regime']: b['S_state']-a['S_state'] for a, b in zip(baseline['details'], selected['details'])},
            'seconds': r['seconds'], 'gpu_peak_allocated_MiB': r['gpu_peak_allocated_MiB'],
            'report_sha256': sha256_file(path), 'ablation_sha256': sha256_file(ablation_path)}
        result['studies'][name] = row
        print(name, row['State_gain'], 'regimes', row['regime_gain'],
            'attention-vs-geometry', architectures['adaptive_attention']['paired_daily_vs_geometry_graph']['ci95'],
            'density-vs-attention', density_pair['ci95'], flush=True)
    result.update(formal_State=None, formal_Physics=None, formal_Queue=None, formal_ODME=None,
        formal_candidate_Total=None, uploaded=False,
        limitations='Each study keeps its calibration winner. Public-train assessment dates were already inspected. Seeds share dates and are not independent day replicates. Density increments are an observation-derived diagnostic. No global network or private-split claim.')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    rows = list(result['studies'].values())
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5.5, .85*len(rows)+1.5)), layout='constrained')
    y = np.arange(len(rows))
    labels = [f"{r['panel']} / {r['fold'].replace('rolling_', '')}\nseed {r['seed']}" for r in rows]
    for i, r in enumerate(rows):
        mean = r['paired_day_gain']['paired_day_mean_gain']
        low, high = r['paired_day_gain']['ci95']
        axes[0].errorbar(mean, i, xerr=np.array([[mean-low], [high-mean]]), fmt='o', color='#087E8B', capsize=4)
        axes[0].text(high+.0002, i, f"pooled gain {r['State_gain']:+.5f}", va='center', fontsize=8)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].axvline(0, color='black', lw=.8)
    axes[0].axvline(.002, color='grey', linestyle='--', lw=.8)
    axes[0].set_xlabel('Paired day-mean State gain; 95% moving-block interval')
    axes[0].set_xlim(-.0004, max(r['paired_day_gain']['ci95'][1] for r in rows)+.0035)
    axes[0].set_title('Selected neural mixture vs full-label LightGBM')
    keys = ('graph_geometry', 'adaptive_attention', 'adaptive_attention_density')
    for j, (key, color, label) in enumerate(zip(keys, ('#617785', '#087E8B', '#D08C20'), ('Geometry graph', '+ spatial attention', '+ density increment loss'))):
        gains = [r['calibrated_architecture_comparison'][key]['State']-r['baseline_State'] for r in rows]
        axes[1].barh(y+(j-1)*.22, gains, height=.2, color=color, label=label)
    axes[1].set_yticks(y, [r['panel']+'\n'+r['fold'].replace('rolling_', '')+' / '+str(r['seed']) for r in rows], fontsize=8)
    axes[1].axvline(0, color='black', lw=.8)
    axes[1].set_xlabel('Regime-pooled State gain over strong baseline')
    axes[1].set_title('All calibration-fixed architectures; selection unchanged')
    axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, -.16))
    fig.suptitle('Public-train development evidence | Formal best Total remains 0.82940', fontsize=12)
    fig.savefig(args.output.with_suffix('.png'), dpi=180)
    plt.close(fig)
