"""Selection-only neural tuning, followed by one registered assessment."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from trafficflow.graph_state import GraphState, data_loss, fd_loss, density_change_loss
from trafficflow.research_eval import state_from_sufficient_stats, paired_block_interval
from trafficflow.runtime import evidence


def tensor_input(day, device):
    return torch.from_numpy(day['x'].transpose(2,0,1)[None].copy()).to(device)


def predict(model, day, graph, device):
    with torch.no_grad():
        p = model(tensor_input(day,device),graph[0],graph[1]).cpu().numpy()[0].transpose(1,2,0)
    p[...,0] = np.clip(p[...,0]*25,0,graph[2][0].cpu().numpy()[None]*1.05)
    p[...,1] = np.clip(p[...,1]*600,0,graph[2][1].cpu().numpy()[None]*1.05)
    return p.reshape(-1,2)[day['target_indices']]


def stats(day, prediction):
    truth = day['station_truth']
    factor = day['target_link_lanes']/day['station_lanes']
    speed = prediction[:,0]-truth[:,0]
    flow = (prediction[:,1]-truth[:,1])*factor
    rv, rq = np.sqrt(np.mean(speed**2)),np.sqrt(np.mean(flow**2))
    return {'regime':str(day['regime']), 'n':len(truth),'speed_sse':float(np.square(speed).sum()),
        'lane_flow_sse':float(np.square(flow).sum()),
        'daily_State':float(.54*max(0,1-rv/25)+.46*max(0,1-rq/600))}


def evaluate(model, days, graph, device, alpha=1.):
    rows = []
    for day in days:
        neural = predict(model,day,graph,device) if alpha else day['comparator']
        p = (1-alpha)*day['comparator'] + alpha*neural
        rows.append(stats(day,p))
    return {**state_from_sufficient_stats(rows),'days':rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=16)
    parser.add_argument('--steps',type=int,default=60)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--density-change',action='store_true')
    parser.add_argument('--density-change-weight',type=float,default=.01)
    parser.add_argument('--model',choices=['all','graph_density_change'],default='all')
    args = parser.parse_args()
    if args.model=='graph_density_change':args.density_change=True
    if not np.isfinite(args.density_change_weight) or args.density_change_weight<0:
        raise ValueError('Density change weight must be finite and nonnegative')
    if args.output.exists():
        raise FileExistsError('Use a new model experiment directory')
    args.output.mkdir(parents=True)
    prepared = json.loads((args.cache/'prepared.json').read_text())
    started = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    seed = prepared['seed'] if args.seed is None else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False
    rng = np.random.default_rng(seed)
    sets = {key:[dict(np.load(path,allow_pickle=False)) for path in sorted((args.cache/key).glob('*.npz'))]
            for key in ('fit','selection')}
    graph_np = dict(np.load(args.cache/'graph.npz'))
    graph = [torch.tensor(graph_np[key],device=device) for key in ('incoming','outgoing','fd')]
    if args.density_change:
        from trafficflow.data import Release
        net=Release('data/kaggle_public').network(prepared['panel'])
        links=sets['fit'][0]['links']
        accumulation_factor=torch.tensor((net.lanes*net.length_km).reindex(links).to_numpy(np.float32),device=device)
        for days in sets.values():
            for day in days:
                times=np.array([str(t).removesuffix('Z') for t in day['times']],dtype='datetime64[s]')
                if not (np.diff(times).astype('timedelta64[s]').astype(int)==300).all():
                    raise ValueError('Density increment loss needs consecutive five-minute pairs')
    result = {'experiment':'A01_masked_graph_state','cache':str(args.cache),'prepared':prepared,
        'torch':torch.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        'source_at_start':evidence(gpu_used=device.type=='cuda'),
        'fit_seed':seed,'fixed_LGB_fit_seed':prepared['seed'],
        'epochs':args.epochs,'steps_per_epoch':args.steps,'batch_size':4,'patch_steps':64,
        'training':{},'selection':{},'formal_State':None,
        'comparison_limit':'Same public train dates; LGB sees 300000 sampled labels, neural samples patches. This does not isolate architecture from label exposure.'}
    baseline = evaluate(None,sets['selection'],graph,device,0.)
    result['baseline_selection'] = baseline
    architectures = [('temporal',False,0.),('graph',True,0.),('graph_FD',True,.01)]
    if args.density_change:
        architectures += [('graph_density_change',True,0.)]
    if args.model=='graph_density_change':
        architectures=[('graph_density_change',True,0.)]
    selected = None
    for name,use_graph,physical_weight in architectures:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = GraphState(graph=use_graph).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        history=[]
        best_score=-1
        tic=time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            losses=[]
            for _ in range(args.steps):
                batch=[]
                for _ in range(4):
                    day = sets['fit'][rng.integers(len(sets['fit']))]
                    at = rng.integers(0,day['x'].shape[0]-64+1)
                    batch.append((day,at))
                x = torch.from_numpy(np.stack([d['x'][at:at+64].transpose(2,0,1) for d,at in batch])).to(device)
                y = torch.from_numpy(np.stack([d['y'][at:at+64].transpose(2,0,1) for d,at in batch])).to(device)
                target = torch.from_numpy(np.stack([d['mask'][at:at+64] for d,at in batch])).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x,graph[0],graph[1])
                loss = data_loss(prediction,y,target)
                if physical_weight:
                    loss = loss + physical_weight*fd_loss(prediction,target,graph[2])
                if name=='graph_density_change':
                    loss = loss + args.density_change_weight*density_change_loss(prediction,y,target,x,accumulation_factor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if (epoch+1)%4==0 or epoch+1==args.epochs:
                model.eval()
                cal=evaluate(model,sets['selection'],graph,device)
                history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),
                                'selection_State':cal['S_state']})
                if cal['S_state']>best_score:
                    best_score=cal['S_state']
                    torch.save(model.state_dict(),args.output/(name+'.pt'))
                print(name,epoch+1,cal['S_state'],flush=True)
        model.load_state_dict(torch.load(args.output/(name+'.pt'),map_location=device,weights_only=True))
        model.eval()
        choices=[]
        for alpha in (0.,.25,.5,.75,1.):
            cal=evaluate(model,sets['selection'],graph,device,alpha)
            choices.append({'alpha':alpha,'selection_State':cal['S_state']})
        choice=max(choices,key=lambda c:(c['selection_State'],-c['alpha']))
        result['training'][name]={'history':history,'seconds':time.perf_counter()-tic,'FD_loss_weight':physical_weight,
            'public_train_density_increment_weight':args.density_change_weight if name=='graph_density_change' else 0.}
        result['selection'][name]={'ensemble_grid':choices,'selected':choice}
        if selected is None or choice['selection_State']>selected['selection_State']:
            selected={'name':name,**choice}
        (args.output/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    result['selected']=selected
    (args.output/'selection_sealed.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    sets['assessment']=[dict(np.load(path,allow_pickle=False)) for path in sorted((args.cache/'assessment').glob('*.npz'))]
    model=GraphState(graph=selected['name']!='temporal').to(device)
    model.load_state_dict(torch.load(args.output/(selected['name']+'.pt'),map_location=device,weights_only=True))
    model.eval()
    result['baseline_assessment']=evaluate(None,sets['assessment'],graph,device,0.)
    result['selected_assessment']=evaluate(model,sets['assessment'],graph,device,selected['alpha'])
    result['paired_daily_interval']=paired_block_interval(
        [d['daily_State'] for d in result['baseline_assessment']['days']],
        [d['daily_State'] for d in result['selected_assessment']['days']])
    result['seconds']=time.perf_counter()-started
    result['gpu_peak_allocated_MiB']=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None
    (args.output/'experiment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(selected,result['baseline_assessment']['S_state'],result['selected_assessment']['S_state'],flush=True)


if __name__=='__main__':
    main()
