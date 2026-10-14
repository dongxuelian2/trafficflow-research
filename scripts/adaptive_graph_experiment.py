"""Matched adaptive-attention ablation against preserved full-label State.

No assessment labels are loaded until all architecture, checkpoint and blend
choices are sealed on calibration. Outputs remain public-train development
evidence, never official Physics or Kaggle Total.
"""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.adaptive_graph_state import AdaptiveGraphState,patch_score_loss,AdaptiveStateModel,reconstruction_input
from trafficflow.graph_state import density_change_loss
from trafficflow.data import Release,VALUES
from trafficflow.state import StateFeatures,StateModel
from trafficflow.experiments import partition_date,day_template
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats,paired_block_interval
from trafficflow.runtime import evidence,capture_source_snapshot,sha256_file
from train_graph_experiment import stats,tensor_input
from diagnose_graph_density import increment_stats


def predict(model,day,graph,device):
    with torch.no_grad():
        p=model(tensor_input(day,device),*graph).cpu().numpy()[0].transpose(1,2,0)
    p[...,0]=np.clip(p[...,0]*25,0,graph[2][0].cpu().numpy()[None]*1.05)
    p[...,1]=np.clip(p[...,1]*600,0,graph[2][1].cpu().numpy()[None]*1.05)
    return p.reshape(-1,2)[day['target_indices']]


def evaluate(model,days,graph,device,alpha,factor):
    rows=[];increments=[]
    for day in days:
        p=(1-alpha)*day['comparator']+alpha*predict(model,day,graph,device) if alpha else day['comparator']
        rows.append(stats(day,p));increments.append(increment_stats(day,p,factor))
    return {**state_from_sufficient_stats(rows),'days':rows,
        'density_target_pair_relative_L1':sum(r['error_L1'] for r in increments)/sum(r['truth_increment_L1'] for r in increments)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--baseline-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--epochs',type=int,default=24)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--hidden',type=int,default=48)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite adaptive graph research')
    info=json.loads((args.cache/'prepared.json').read_text())
    report=Path('reports/research')/(args.output.name+'.json')
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'Adaptive_attention_strong_State_control','panel':info['panel'],'fold':info['fold'],
        'cache':args.cache.as_posix(),'baseline_file':args.baseline_file.as_posix(),
        'baseline_sha256':sha256_file(args.baseline_file),'seed':args.seed,
        'architectures':['graph_geometry','adaptive_attention','adaptive_attention_density'],
        'hidden':args.hidden,'heads':2,'epochs':args.epochs,'steps_per_epoch':args.steps,
        'batch_size':4,'patch_steps':64,'optimizer':'AdamW lr=.001 weight_decay=.0001 clip_norm=1',
        'loss':'Equal-patch channel-scaled RMSE; fit regimes sampled uniformly',
        'density_weight':.01,'selection':'Calibration State every four epochs; architecture/blend chosen on calibration only',
        'blend_grid':[0,.25,.5,.75,1.],'labels':'All fit dates of public train available to both neural methods and full-label LGB; sampled patches repeat targets.',
        'matched_ablation':'Same patch RNG, width, optimizer and training budget; geometry graph control lacks adaptive attention',
        'public_references':['https://arxiv.org/abs/2308.10425','https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html'],
        'independent_implementation':'FD geometry conditioning and topology relation biases; no third-party source, weights or predictions',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'assessment_status':'Previously inspected public-train development dates; not an untouched final test'})
    args.output.mkdir(parents=True)
    snapshot=capture_source_snapshot(args.output/'source_snapshot.zip')
    start=time.perf_counter()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if device.type=='cuda':
        torch.backends.cudnn.benchmark=False
        torch.cuda.reset_peak_memory_stats()
    release=Release('data/kaggle_public')
    panel=info['panel'];net,topo=release.network(panel),release.topology(panel)
    baseline=StateModel.load(args.baseline_file)
    old=StateModel.load(args.cache/'lightgbm.pkl')
    if not old.profile.slots.equals(baseline.profile.slots) or not old.profile.fallback.equals(baseline.profile.fallback):
        raise ValueError('Strong comparator and cached neural features have different profiles')
    files={partition_date(release,p):p for p in release.files(panel,'train')}
    template=release.template(1,panel,'train')
    def load(part):
        result=[]
        for path in sorted((args.cache/part).glob('*.npz')):
            day=dict(np.load(path,allow_pickle=False))
            if part!='fit':
                frame=release.read(files[path.stem])
                f=StateFeatures(frame,net,topo,baseline.profile,version=baseline.feature_version)
                if not np.array_equal(reconstruction_input(f),day['x']):raise ValueError('Deployable input differs from audited cached features')
                target=day_template(release,panel,'train',frame,template)
                if not np.array_equal(f.lookup(target),day['target_indices']):raise ValueError('Strong target ordering differs')
                p=baseline.predict(f,target)[VALUES].to_numpy(float,copy=True)
                p[:,0]=np.minimum(p[:,0],target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                p[:,1]=np.minimum(p[:,1],target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/target.link_id.map(net.lanes).to_numpy(float)
                day['comparator']=p
            result.append(day)
        if not result:raise ValueError('Cached partition missing')
        return result
    fit,cal=load('fit'),load('selection')
    regimes={r:[d for d in fit if str(d['regime'])==r] for r in ('R1','R2','R3')}
    if any(not v for v in regimes.values()):raise ValueError('All fit regimes required')
    graph_np=dict(np.load(args.cache/'graph.npz'))
    graph=[torch.tensor(graph_np[k],device=device) for k in ('incoming','outgoing','fd')]
    factor=(net.lanes*net.length_km).reindex(fit[0]['links']).to_numpy(np.float32)
    factor_tensor=torch.tensor(factor,device=device)
    result={'registered':registered,'source_snapshot':snapshot,'prepared':info,
        'device':str(device),'torch':torch.__version__,'training':{},'selection':{},
        'source_at_start':evidence(gpu_used=device.type=='cuda'),
        'baseline_selection':evaluate(None,cal,graph,device,0.,factor),
        'cache_integrity':{p.as_posix():sha256_file(p) for part in ('fit','selection') for p in sorted((args.cache/part).glob('*.npz'))}}
    architectures=[('graph_geometry',False,0.),('adaptive_attention',True,0.),('adaptive_attention_density',True,.01)]
    selected=None
    for name,attention,density_weight in architectures:
        torch.manual_seed(args.seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
        rng=np.random.default_rng(args.seed)
        model=AdaptiveGraphState(hidden=args.hidden,heads=2,attention=attention).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        history=[];best=-np.inf;tic=time.perf_counter()
        for epoch in range(args.epochs):
            model.train();losses=[]
            for _ in range(args.steps):
                batch=[]
                for _ in range(4):
                    group=regimes[('R1','R2','R3')[rng.integers(3)]]
                    day=group[rng.integers(len(group))]
                    at=rng.integers(0,day['x'].shape[0]-64+1)
                    batch.append((day,at))
                x=torch.from_numpy(np.stack([d['x'][at:at+64].transpose(2,0,1) for d,at in batch])).to(device)
                y=torch.from_numpy(np.stack([d['y'][at:at+64].transpose(2,0,1) for d,at in batch])).to(device)
                mask=torch.from_numpy(np.stack([d['mask'][at:at+64] for d,at in batch])).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction=model(x,*graph)
                loss=patch_score_loss(prediction,y,mask)
                if density_weight:loss=loss+density_weight*density_change_loss(prediction,y,mask,x,factor_tensor)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if (epoch+1)%4==0 or epoch+1==args.epochs:
                model.eval();score=evaluate(model,cal,graph,device,1.,factor)
                history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),
                    'selection_State':score['S_state'],'selection_density_increment_L1':score['density_target_pair_relative_L1']})
                if score['S_state']>best:
                    best=score['S_state'];torch.save(model.state_dict(),args.output/(name+'.pt'))
                print(name,epoch+1,score['S_state'],score['density_target_pair_relative_L1'],flush=True)
        model.load_state_dict(torch.load(args.output/(name+'.pt'),map_location=device,weights_only=True))
        model.eval()
        choices=[{'alpha':a,'metrics':evaluate(model,cal,graph,device,a,factor)} for a in (0.,.25,.5,.75,1.)]
        chosen=max(choices,key=lambda r:(r['metrics']['S_state'],-r['alpha']))
        result['training'][name]={'history':history,'seconds':time.perf_counter()-tic,
            'density_weight':density_weight,'parameters':sum(p.numel() for p in model.parameters())}
        result['selection'][name]={'ensemble_grid':choices,'selected_alpha':chosen['alpha'],
            'selection_State':chosen['metrics']['S_state']}
        if selected is None or chosen['metrics']['S_state']>selected['selection_State']:
            selected={'name':name,'alpha':chosen['alpha'],'selection_State':chosen['metrics']['S_state']}
        result['selected']=selected
        (args.output/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.output/'selection_sealed.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    assessment=load('assessment')
    model=AdaptiveGraphState(hidden=args.hidden,heads=2,attention=selected['name']!='graph_geometry').to(device)
    model.load_state_dict(torch.load(args.output/(selected['name']+'.pt'),map_location=device,weights_only=True));model.eval()
    result['baseline_assessment']=evaluate(None,assessment,graph,device,0.,factor)
    result['selected_assessment']=evaluate(model,assessment,graph,device,selected['alpha'],factor)
    result['paired_daily_interval']=paired_block_interval(
        [d['daily_State'] for d in result['baseline_assessment']['days']],
        [d['daily_State'] for d in result['selected_assessment']['days']])
    result['seconds']=time.perf_counter()-start
    result['gpu_peak_allocated_MiB']=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None
    result['source_at_end']=evidence(gpu_used=device.type=='cuda')
    result['model_sha256']={p.name:sha256_file(p) for p in args.output.glob('*.pt')}
    deployed=AdaptiveStateModel(baseline,model.state_dict(),topo,selected['alpha'],hidden=args.hidden,
        attention=selected['name']!='graph_geometry')
    deployed.save(args.output/'State_selected.pkl')
    result['deployable_model_sha256']=sha256_file(args.output/'State_selected.pkl')
    result['formal_State']=None;result['formal_Physics']=None;result['formal_Total']=None
    result['limitations']='One direction/fold/fit seed. All assessments are already-inspected public-train development dates. Adaptive-attention comparison controls neural optimization budget; full LGB has the same available label dates, not the same repeated patch count. No hidden labels or boundary flux.'
    (args.output/'experiment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('selected',selected,'State',result['baseline_assessment']['S_state'],result['selected_assessment']['S_state'],
          'seconds',result['seconds'],'GPU MiB',result['gpu_peak_allocated_MiB'],flush=True)


if __name__=='__main__':main()
