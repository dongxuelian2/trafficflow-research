"""Assess one density-loss candidate selected only from its new calibration proxy."""
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from trafficflow.data import Release
from trafficflow.graph_state import GraphState
from trafficflow.research_eval import register_protocol, paired_block_interval
from train_graph_experiment import evaluate, predict
from diagnose_graph_density import increment_stats

if __name__=='__main__':
    output=Path('reports/research/A10_joint_density_assessment.json')
    if output.exists():raise FileExistsError('Do not overwrite assessment')
    tradeoff=json.loads(Path('reports/research/A09_selection_density_tradeoffs.json').read_text())
    baseline=next(r for r in tradeoff['rows'] if r['alpha']==0.)['S_state']
    eligible=[r for r in tradeoff['rows'] if r['S_state']>=baseline]
    selected=min(eligible,key=lambda r:r['public_density_increment_relative_L1'])
    registered=register_protocol(output.with_suffix('.registered.json'),{'id':'A10_joint_density_assessment',
        'selection':'min public target-pair increment L1, subject to no calibration State loss vs same-split LGB',
        'selected_model':selected['model'],'alpha':selected['alpha'],
        'public_density_loss_weight':.01,'new_assessment_for_selected_auxiliary':True,
        'formal_Physics':None})
    source=Path('artifacts/research/A09_public_density_increment')
    info=json.loads((source/'experiment.json').read_text())
    cache=Path(info['cache'])
    days=[dict(np.load(p,allow_pickle=False)) for p in sorted((cache/'assessment').glob('*.npz'))]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    g=dict(np.load(cache/'graph.npz'))
    graph=[torch.tensor(g[key],device=device) for key in ('incoming','outgoing','fd')]
    model=GraphState().to(device)
    model.load_state_dict(torch.load(source/(selected['model']+'.pt'),map_location=device,weights_only=True))
    model.eval()
    reference=evaluate(None,days,graph,device,0.)
    candidate=evaluate(model,days,graph,device,selected['alpha'])
    net=Release('data/kaggle_public').network(info['prepared']['panel'])
    factor=(net.lanes*net.length_km).reindex(days[0]['links']).to_numpy(float)
    density={key:[] for key in ('baseline','candidate')}
    for day in days:
        p=(1-selected['alpha'])*day['comparator']+selected['alpha']*predict(model,day,graph,device)
        density['baseline'].append(increment_stats(day,day['comparator'],factor))
        density['candidate'].append(increment_stats(day,p,factor))
    result={'registered':registered,'State_baseline':reference,'State_candidate':candidate,
        'paired_daily_interval':paired_block_interval([r['daily_State'] for r in reference['days']],
                                                     [r['daily_State'] for r in candidate['days']]),
        'density_target_pair_L1_ratio':{key:sum(r['error_L1'] for r in rows)/sum(r['truth_increment_L1'] for r in rows)
                                       for key,rows in density.items()},
        'formal_Physics':None,'formal_total':None,
        'limitations':'Single direction/fit seed. Public noisy density-derived proxy excludes unavailable anchors. No official flux; does not measure formal Physics.'}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(reference['S_state'],candidate['S_state'],result['density_target_pair_L1_ratio'])
