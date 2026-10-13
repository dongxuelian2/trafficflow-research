"""Selection-only State/accumulation tradeoffs, without formal Physics scoring."""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import Release
from trafficflow.graph_state import GraphState
from trafficflow.research_eval import state_from_sufficient_stats
from train_graph_experiment import predict, stats


def increment_stats(day,prediction,factor):
    shape=day['mask'].shape
    accum=np.zeros((shape[0]*shape[1],2))
    counts=np.zeros(shape[0]*shape[1])
    np.add.at(accum,day['target_indices'],prediction/np.array([25.,600.]))
    np.add.at(counts,day['target_indices'],1)
    p=day['x'][...,:2].copy().reshape(-1,2)
    selected=counts>0
    p[selected]=accum[selected]/counts[selected,None]
    p=p.reshape(*shape,2)
    truth=np.where(day['mask'][...,None],day['y'],day['x'][...,:2])
    n=p[...,1]*600/np.maximum(p[...,0]*25,1)*factor
    nt=truth[...,1]*600/np.maximum(truth[...,0]*25,1)*factor
    valid=day['mask']|np.all(day['x'][...,2:4]>.5,axis=-1)
    pair=valid[1:]&valid[:-1]&(day['mask'][1:]|day['mask'][:-1])
    error=np.abs(np.diff(n,axis=0)-np.diff(nt,axis=0))[pair]
    rhs=np.abs(np.diff(nt,axis=0))[pair]
    return {'error_L1':float(error.sum()),'truth_increment_L1':float(rhs.sum()),'n_pairs':int(pair.sum())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',type=Path,default=Path('artifacts/research/A09_public_density_increment'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Do not overwrite tradeoff evidence')
    start=time.perf_counter()
    run=json.loads((args.models/'experiment.json').read_text())
    cache=Path(run['cache'])
    days=[dict(np.load(path,allow_pickle=False)) for path in sorted((cache/'selection').glob('*.npz'))]
    # Deliberately do not load assessment labels for this new diagnostic.
    net=Release('data/kaggle_public').network(run['prepared']['panel'])
    factor=(net.lanes*net.length_km).reindex(days[0]['links']).to_numpy(float)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    graph_np=dict(np.load(cache/'graph.npz'))
    graph=[torch.tensor(graph_np[key],device=device) for key in ('incoming','outgoing','fd')]
    rows=[]
    for name in ('graph','graph_FD','graph_density_change'):
        if not (args.models/(name+'.pt')).exists():continue
        model=GraphState().to(device)
        model.load_state_dict(torch.load(args.models/(name+'.pt'),map_location=device,weights_only=True))
        model.eval()
        neural=[predict(model,day,graph,device) for day in days]
        for alpha in (0.,.25,.5,.75,1.):
            state=[]
            increments=[]
            for day,pred in zip(days,neural):
                mixture=(1-alpha)*day['comparator']+alpha*pred
                state.append(stats(day,mixture))
                increments.append(increment_stats(day,mixture,factor))
            total_error=sum(r['error_L1'] for r in increments)
            true_increment=sum(r['truth_increment_L1'] for r in increments)
            rows.append({'model':name,'alpha':alpha,**state_from_sufficient_stats(state),
                'public_density_increment_relative_L1':total_error/max(true_increment,1e-9),
                'public_density_increment_MAE_vehicles':total_error/max(sum(r['n_pairs'] for r in increments),1),
                'n_pairs':sum(r['n_pairs'] for r in increments),'formal_Physics':None})
    result={'experiment':'A09_selection_density_tradeoffs','panel':run['prepared']['panel'],
        'dates':run['prepared']['dates']['selection'],'rows':rows,'seconds':time.perf_counter()-start,
        'truth':'Public-train observation-derived accumulation increments, visible-anchor coverage only; denominator restricted to pairs touching a target',
        'limitations':'No hidden boundary flux. These supervised increment errors are not formal or locally official Physics scores. No new assessment inspection.'}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print([(r['model'],r['alpha'],round(r['S_state'],6),round(r['public_density_increment_relative_L1'],6)) for r in rows])


if __name__=='__main__':main()
