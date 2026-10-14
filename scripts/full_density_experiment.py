"""Matched State/legacy/coverage/full-denominator training with sealed cal choices."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.adaptive_graph_state import AdaptiveGraphState, AdaptiveStateModel, reconstruction_input, patch_score_loss
from trafficflow.graph_state import density_change_loss, public_density_change_loss
from trafficflow.density_label_metrics import public_increment_stats, aggregate_public_increments
from trafficflow.state import StateModel, StateFeatures
from trafficflow.data import Release, VALUES
from trafficflow.experiments import partition_date, day_template
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import register_protocol, state_from_sufficient_stats, paired_block_interval
from adaptive_graph_experiment import predict
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats
from quality_state_experiment import state_fingerprint


VARIANTS = [('state_only', 'none'), ('legacy_density', 'legacy'),
            ('coverage_density', 'public_target_pairs'), ('full_density', 'public_all_pairs')]


def metrics(days, predictions, alpha, factor):
    scores, public, old = [], [], []
    for day, prediction in zip(days, predictions):
        p = (1-alpha)*day['comparator']+alpha*prediction
        scores.append(stats(day, p))
        public.append(public_increment_stats(day, p, factor))
        old.append(increment_stats(day, p, factor))
    return {**state_from_sufficient_stats(scores), 'days': scores,
            **aggregate_public_increments(public), 'public_increment_days': public,
            'legacy_target_pair_relative_L1': sum(r['error_L1'] for r in old)/max(sum(r['truth_increment_L1'] for r in old),1e-9)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--baseline-file', type=Path, required=True)
    parser.add_argument('--initialization', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    report = Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists(): raise FileExistsError('Never overwrite full density experiments')
    prepared = json.loads((args.cache/'prepared.json').read_text())
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id':'Matched_public_coverage_density_loss','cache':args.cache.as_posix(),
        'prepared_sha256':sha256_file(args.cache/'prepared.json'),'panel':prepared['panel'],'fold':prepared['fold'],
        'baseline_file':args.baseline_file.as_posix(),'baseline_sha256':sha256_file(args.baseline_file),
        'initialization_sha256':sha256_file(args.initialization),'seed':args.seed,'variants':VARIANTS,
        'density_weight':.01,'channels':12,'output_coordinates':'vq','hidden':48,'heads':2,
        'epochs':24,'steps_per_epoch':120,'batch_size':4,'patch_steps':64,
        'optimizer':'AdamW .001, weight_decay1e-4, clip_norm1','sampling':'Identical RNG and shared weights across four variants',
        'selection':'Per-variant best pure cal State checkpoint, then alpha grid; primary by cal State only',
        'joint_selection':'Minimum cal macro-regime public increment L1, State >= strong LGB and >= best primary-.001',
        'blend_grid':[0.,.25,.5,.75,1.],
        'label_scope':'Full eligible raw TRAIN labels and public Mode A pair validity only in loss/evaluation; never input channels',
        'normalization_caveat':'Constant lambda .01; denominator changes effective regularization scale. Not an equal-gradient-strength comparison',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    args.output.mkdir(parents=True)
    snapshot=capture_source_snapshot(args.output/'source_snapshot.zip')
    start=time.perf_counter()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if device.type=='cuda':torch.backends.cudnn.benchmark=False;torch.cuda.reset_peak_memory_stats()
    release=Release('data/kaggle_public');panel=prepared['panel']
    baseline=StateModel.load(args.baseline_file)
    net,topology=release.network(panel),release.topology(panel)
    template=release.template(1,panel,'train')
    files={partition_date(release,p):p for p in release.files(panel,'train')}
    def load(part):
        result=[]
        for path in sorted((args.cache/part).glob('*.npz')):
            record=next(r for r in prepared['records'] if r['part']==part and r['date']==path.stem)
            if sha256_file(path)!=record['full_label_cache_sha256']:raise ValueError('Full label cache changed')
            d=dict(np.load(path,allow_pickle=False));d['x']=d.get('x_control',d['x'])
            if not np.isfinite(d['y_full']).all() or not np.array_equal(d['y_full'][d['mask']],d['y'][d['mask']]):
                raise ValueError('Scored labels changed or full labels nonfinite')
            if part!='fit':
                f=StateFeatures(release.read(files[path.stem]),net,topology,baseline.profile,version=baseline.feature_version)
                if not np.array_equal(reconstruction_input(f),d['x']):raise ValueError('Standalone strict12 inputs differ')
                target=day_template(release,panel,'train',f.frame,template)
                if not np.array_equal(f.lookup(target),d['target_indices']):raise ValueError('Target lookup differs')
                p=baseline.predict(f,target)[VALUES].to_numpy(float,copy=True)
                p[:,0]=np.clip(p[:,0],0,target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                p[:,1]=np.clip(p[:,1],0,target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/target.link_id.map(net.lanes).to_numpy(float)
                d['comparator']=p
            result.append(d)
        return result
    fit,cal=load('fit'),load('selection')
    if sha256_file(args.cache/'graph.npz')!=prepared['graph_sha256']:raise ValueError('Graph changed')
    g=dict(np.load(args.cache/'graph.npz'));graph=[torch.tensor(g[k],device=device) for k in ['incoming','outgoing','fd']]
    factor=(net.lanes*net.length_km).reindex(fit[0]['links']).to_numpy(np.float32)
    factor_tensor=torch.tensor(factor,device=device)
    common=torch.load(args.initialization,map_location='cpu',weights_only=True)
    torch.save(common,args.output/'shared_initialization.pt')
    reference=metrics(cal,[np.zeros_like(d['comparator']) for d in cal],0.,factor)
    result={'registered':registered,'source_snapshot':snapshot,'prepared':prepared,
        'shared_initialization_fingerprint':state_fingerprint(common),'baseline_selection':reference,
        'training':{},'calibration_choices':[],'primary_per_model':{}}
    regimes={r:[d for d in fit if str(d['regime'])==r] for r in ['R1','R2','R3']}
    for name,scope in VARIANTS:
        torch.manual_seed(args.seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
        rng=np.random.default_rng(args.seed);sampling=hashlib.sha256()
        model=AdaptiveGraphState().to(device);model.load_state_dict(common)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        best=-np.inf;primary=None;history=[];tic=time.perf_counter()
        for epoch in range(24):
            model.train();losses=[]
            for _ in range(120):
                batch=[]
                for _ in range(4):
                    regime=['R1','R2','R3'][rng.integers(3)];group=regimes[regime]
                    day=group[rng.integers(len(group))];at=int(rng.integers(0,day['x'].shape[0]-64+1))
                    batch.append((day,at));sampling.update(f"{regime}|{day['times'][0]}|{at};".encode())
                x=torch.from_numpy(np.stack([d['x'][a:a+64].transpose(2,0,1) for d,a in batch])).to(device)
                y=torch.from_numpy(np.stack([d['y'][a:a+64].transpose(2,0,1) for d,a in batch])).to(device)
                target=torch.from_numpy(np.stack([d['mask'][a:a+64] for d,a in batch])).to(device)
                optimizer.zero_grad(set_to_none=True);p=model(x,*graph)
                loss=patch_score_loss(p,y,target)
                if scope=='legacy':loss=loss+.01*density_change_loss(p,y,target,x,factor_tensor)
                elif scope!='none':
                    labels=torch.from_numpy(np.stack([d['y_full'][a:a+64].transpose(2,0,1) for d,a in batch])).to(device)
                    pairs=torch.from_numpy(np.stack([d['lwr_pair_valid'][a:a+63] for d,a in batch])).to(device)
                    if scope=='public_target_pairs':pairs=pairs&(target[:,1:]|target[:,:-1])
                    loss=loss+.01*public_density_change_loss(p,labels,target,pairs,factor_tensor)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if (epoch+1)%4:continue
            model.eval();checkpoint=args.output/f'{name}_epoch{epoch+1:02d}.pt';torch.save(model.state_dict(),checkpoint)
            predictions=[predict(model,d,graph,device) for d in cal]
            choices=[{'name':name,'scope':scope,'epoch':epoch+1,'alpha':a,'checkpoint_sha256':sha256_file(checkpoint),
                'metrics':metrics(cal,predictions,a,factor)} for a in [0.,.25,.5,.75,1.]]
            result['calibration_choices'].extend(choices);pure=choices[-1]['metrics']
            if pure['S_state']>best:best=pure['S_state'];primary=max(choices,key=lambda r:(r['metrics']['S_state'],-r['alpha']))
            history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'pure_cal_State':pure['S_state'],
                'pure_cal_public_increment_L1':pure['public_coverage_increment_relative_L1']})
            print(name,epoch+1,pure['S_state'],pure['public_coverage_increment_relative_L1'],flush=True)
        result['training'][name]={'history':history,'seconds':time.perf_counter()-tic,'sample_sequence_sha256':sampling.hexdigest()}
        result['primary_per_model'][name]=primary
        (args.output/'progress.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    if len({r['sample_sequence_sha256'] for r in result['training'].values()})!=1:raise ValueError('Sampled patches differ')
    primary=max(result['primary_per_model'].values(),key=lambda r:r['metrics']['S_state'])
    acceptable=[r for r in result['calibration_choices'] if r['metrics']['S_state']>=reference['S_state'] and r['metrics']['S_state']>=primary['metrics']['S_state']-.001]
    joint=min(acceptable,key=lambda r:(r['metrics']['public_coverage_increment_relative_L1'],-r['metrics']['S_state'],r['alpha']))
    result['selected']={'primary':primary,'joint':joint}
    (args.output/'selection_sealed.json').write_text(json.dumps(result['selected'],indent=2,allow_nan=False),encoding='utf-8')
    assessment=load('assessment')
    result['baseline_assessment']=metrics(assessment,[np.zeros_like(d['comparator']) for d in assessment],0.,factor)
    result['assessment']={}
    for role,choice in {**result['primary_per_model'],'joint_challenger':joint}.items():
        checkpoint=args.output/f"{choice['name']}_epoch{choice['epoch']:02d}.pt"
        if sha256_file(checkpoint)!=choice['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        model=AdaptiveGraphState().to(device);model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
        measured=metrics(assessment,[predict(model,d,graph,device) for d in assessment],choice['alpha'],factor)
        measured['paired_day_interval_vs_LGB']=paired_block_interval([r['daily_State'] for r in result['baseline_assessment']['days']],[r['daily_State'] for r in measured['days']])
        result['assessment'][role]=measured
    for role,choice in result['selected'].items():
        state=torch.load(args.output/f"{choice['name']}_epoch{choice['epoch']:02d}.pt",map_location='cpu',weights_only=True)
        AdaptiveStateModel(baseline,state,topology,choice['alpha']).save(args.output/f'State_{role}.pkl')
    result.update(seconds=time.perf_counter()-start,source=evidence(gpu_used=device.type=='cuda'),
        peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
        deployable_model_sha256={role:sha256_file(args.output/f'State_{role}.pkl') for role in result['selected']},
        formal_State=None,formal_Physics=None,formal_Total=None,
        limitations='Old public-train development dates, one fit seed. Full labels only train/evaluation. Constant lambda changes effective strength with denominator. Observed-increment proxy is not organizer flux or formal Physics.')
    (args.output/'experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    report.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('chosen',primary['name'],primary['alpha'],joint['name'],joint['alpha'],flush=True)
    print({k:(v['S_state'],v['public_coverage_increment_relative_L1']) for k,v in result['assessment'].items()},result['seconds'],flush=True)


if __name__=='__main__':main()
