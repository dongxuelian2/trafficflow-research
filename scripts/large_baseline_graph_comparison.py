"""Recheck existing graph models against the stronger full-label LGB model."""
import json
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from trafficflow.data import Release,VALUES
from trafficflow.experiments import day_template, partition_date
from trafficflow.state import StateFeatures,StateModel
from trafficflow.graph_state import GraphState
from trafficflow.research_eval import register_protocol, state_from_sufficient_stats, paired_block_interval
from train_graph_experiment import predict,stats
from diagnose_graph_density import increment_stats


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('reports/research/A15_large_LGB_graph_recomparison.json'))
    parser.add_argument('--cache',type=Path,default=Path('artifacts/research/A01_D12_I5_N_autumn_seed20261013'))
    parser.add_argument('--models',type=Path,default=Path('artifacts/research/A09_public_density_increment'))
    parser.add_argument('--large-baseline',type=Path,default=Path('artifacts/research/A14_full_label_LGB_control'))
    args=parser.parse_args()
    output=args.output
    if output.exists():raise FileExistsError('Do not overwrite comparison')
    cache=args.cache
    source=args.models
    large=StateModel.load(args.large_baseline/'lightgbm_large_budget.pkl')
    budget=json.loads((args.large_baseline/'experiment.json').read_text())
    prepared=json.loads((cache/'prepared.json').read_text())
    available=[n for n in ('graph','graph_FD','graph_density_change') if (source/(n+'.pt')).exists()]
    registered=register_protocol(output.with_suffix('.registered.json'),{'id':'A15_strong_baseline_recomparison',
        'panel':prepared['panel'],'fold':prepared['fold'],
        'large_LGB_labels':budget['n_fitting_labels'],'existing_graph_models':available,
        'alpha_grid':[0,.25,.5,.75,1],'selection':'max pooled State on selection days; no assessment-based model choice',
        'formal_State':None,'formal_Physics':None})
    release=Release('data/kaggle_public')
    panel=prepared['panel']
    net,topo=release.network(panel),release.topology(panel)
    template=release.template(1,panel,'train')
    day_paths={partition_date(release,p):p for p in release.files(panel,'train')}

    def days(part):
        result=[]
        for path in sorted((cache/part).glob('*.npz')):
            day=dict(np.load(path,allow_pickle=False))
            frame=release.read(day_paths[path.stem])
            f=StateFeatures(frame,net,topo,large.profile,version=2)
            target=day_template(release,panel,'train',frame,template)
            if not np.array_equal(f.lookup(target),day['target_indices']):
                raise ValueError('Cached graph target order changed')
            p=large.predict(f,target)[VALUES].to_numpy(float)
            p[:,0]=np.minimum(p[:,0],target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
            p[:,1]=np.minimum(p[:,1],target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)
            p[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
            day['comparator']=p
            result.append(day)
        return result

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    g=dict(np.load(cache/'graph.npz'))
    graph=[torch.tensor(g[k],device=device) for k in ('incoming','outgoing','fd')]
    selection=days('selection')
    factor=(net.lanes*net.length_km).reindex(selection[0]['links']).to_numpy(float)
    choices=[]
    for name in registered['protocol']['existing_graph_models']:
        model=GraphState().to(device)
        model.load_state_dict(torch.load(source/(name+'.pt'),map_location=device,weights_only=True))
        model.eval()
        neural=[predict(model,d,graph,device) for d in selection]
        for alpha in registered['protocol']['alpha_grid']:
            state=[]
            density=[]
            for day,p in zip(selection,neural):
                mixture=(1-alpha)*day['comparator']+alpha*p
                state.append(stats(day,mixture))
                density.append(increment_stats(day,mixture,factor))
            choices.append({'model':name,'alpha':alpha,**state_from_sufficient_stats(state),
                'density_target_pair_relative_L1':sum(r['error_L1'] for r in density)/sum(r['truth_increment_L1'] for r in density)})
    chosen=max(choices,key=lambda c:(c['S_state'],-c['alpha']))
    seal=output.with_suffix('.selection.json')
    seal.write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    assessment=days('assessment')
    model=GraphState().to(device)
    model.load_state_dict(torch.load(source/(chosen['model']+'.pt'),map_location=device,weights_only=True))
    model.eval()
    old_rows=[]
    new_rows=[]
    old_density=[]
    new_density=[]
    for day in assessment:
        mixture=(1-chosen['alpha'])*day['comparator']+chosen['alpha']*predict(model,day,graph,device)
        old_rows.append(stats(day,day['comparator']))
        new_rows.append(stats(day,mixture))
        old_density.append(increment_stats(day,day['comparator'],factor))
        new_density.append(increment_stats(day,mixture,factor))
    result={'registered':registered,'selection':choices,'chosen':chosen,
        'large_LGB_assessment':state_from_sufficient_stats(old_rows),
        'selected_assessment':state_from_sufficient_stats(new_rows),
        'paired_day_interval':paired_block_interval([r['daily_State'] for r in old_rows],[r['daily_State'] for r in new_rows]),
        'assessment_density_target_pair_relative_L1':{key:sum(r['error_L1'] for r in rows)/sum(r['truth_increment_L1'] for r in rows)
            for key,rows in [('large_LGB',old_density),('selected',new_density)]},
        'formal_State':None,'formal_Physics':None,
        'limitation':'Existing neural weights, one fit seed and one rolling fold. All assessment is development evidence.'}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(chosen['model'],chosen['alpha'],result['large_LGB_assessment']['S_state'],result['selected_assessment']['S_state'])
