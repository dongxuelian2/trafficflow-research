"""Same-window causal T-5 vs released-at-T Queue feature comparison."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.experiments import raw_for, partition_date
from trafficflow.queue_events import sample_events
from trafficflow.queue_calibration import RefinedQueueModel
from trafficflow.queue_event_loss import EventWeightedQueueModel
from trafficflow.queue import ProxyData
from trafficflow.causal_queue import released_origin_rows
from trafficflow.research_eval import Fold,register_protocol
from trafficflow.metrics import grouped_iou
from trafficflow.runtime import evidence


def paired(old,new):
    if not all(np.array_equal(getattr(old,k),getattr(new,k)) for k in ('y','window','condition','date')):
        raise ValueError('Origin feature ablation must use exactly the same windows and labels')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',default='D12_I5_N')
    parser.add_argument('--fold',default='rolling_autumn')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bookkeeping-control',action='store_true')
    parser.add_argument('--selection-per-condition',type=int,default=5)
    parser.add_argument('--assessment-per-condition',type=int,default=5)
    parser.add_argument('--fit-seed',type=int,default=20261013)
    parser.add_argument('--occupancy-features',action='store_true')
    parser.add_argument('--long-calibration',action='store_true')
    parser.add_argument('--event-loss',action='store_true')
    parser.add_argument('--fit-per-condition',type=int,default=80)
    parser.add_argument('--fit-spacing-minutes',type=int,default=360)
    parser.add_argument('--model-settings',choices=['both','600_63'],default='both')
    parser.add_argument('--published-condition',action='store_true')
    parser.add_argument('--onset-last-step',action='store_true')
    args=parser.parse_args()
    if min(args.selection_per_condition,args.assessment_per_condition,args.fit_per_condition,args.fit_spacing_minutes)<1:
        raise ValueError('Event quotas must be positive')
    if args.output.exists():raise FileExistsError('Never overwrite Queue evidence')
    protocol=json.loads(Path('reports/research/protocol_v2_registered.json').read_text())
    cfg=protocol['protocol']
    fold=Fold(**next(f for f in cfg['folds'] if f['name']==args.fold))
    if args.long_calibration:
        selection_end=(pd.Timestamp(fold.fit_end)+pd.Timedelta(days=45)).date().isoformat()
        assessment_end=(pd.Timestamp(selection_end)+pd.Timedelta(days=30)).date().isoformat()
        fold=Fold(name=fold.name+'_long_cal',profile_end=fold.profile_end,fit_end=fold.fit_end,
                  selection_end=selection_end,assessment_end=assessment_end)
    modes=['before_T','at_T']+(['before_T_latest'] if args.bookkeeping_control else [])+(['at_T_occupancy'] if args.occupancy_features else [])+(['at_T_event','at_T_balanced'] if args.event_loss else [])
    if args.published_condition:
        modes+=['at_T_condition']+(['at_T_occupancy_condition'] if args.occupancy_features else [])
    settings=((400,23),(600,63)) if args.model_settings=='both' else ((600,63),)
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'C05_same_window_origin_inputs',
        'panel':args.panel,'fold':args.fold,'methods':[f'{m}_{n}_{l}' for m in modes for n,l in settings],
        'fold_boundaries':fold.__dict__,
        'hypothesis':'Legal released T observations reduce age of the current estimate and expose early congestion.',
        'no_future_inputs':True,'label_source':'public train noisy observations, eligible future cells only',
        'selection':'selection windows only, fit/refined threshold settings frozen before assessment',
        'assessment_seeds':[2909,3253,4001],'formal_Queue':None,
        'selection_per_condition':args.selection_per_condition,
        'assessment_per_condition':args.assessment_per_condition,
        'fit_seed':args.fit_seed,
        'fit_per_condition':args.fit_per_condition,'fit_spacing_minutes':args.fit_spacing_minutes,
        'published_condition_ablation':args.published_condition,
        'onset_last_step':args.onset_last_step,
        'timing_authorization':'Official host public discussion 742350; unknown latent labels remain unavailable',
        'published_condition_hypothesis':'Use the legally supplied window condition instead of recomputing it from an extra noisy T observation.',
        'occupancy_feature_ablation':args.occupancy_features,
        'event_loss_ablation':args.event_loss,
        'event_loss_hypothesis':'Condition/window-balanced losses can reduce domination by large ongoing queues. Class balancing can be harmed by noisy rare positives.',
        'latest_valid_queue_bookkeeping_at_T':True,
        'include_before_T_latest_bookkeeping_control':args.bookkeeping_control})
    start=time.perf_counter()
    release=Release('data/kaggle_public')
    paths=release.files(args.panel,'train')
    dates=fold.assign([partition_date(release,p) for p in paths])
    parts={key:[p for p in paths if partition_date(release,p) in days] for key,days in dates.items()}
    net,topo=release.network(args.panel),release.topology(args.panel)
    origin_cache={}

    def origin_reader(origin):
        token=str(origin)
        if token not in origin_cache:
            origin_cache[token]=released_origin_rows(release,args.panel,'train',origin)
        return origin_cache[token]

    data={}
    samplers={}

    def bookkeeping(data):
        x=data.x.copy()
        x['queue_now']=(x.current<=x.cut_ratio).astype(np.float32)
        return ProxyData(x,data.y,data.window,data.condition,data.date)
    def published(data):
        x=data.x.copy()
        x['condition_ongoing']=(data.condition=='queue_ongoing').astype(np.float32)
        return ProxyData(x,data.y,data.window,data.condition,data.date)
    for key,seed,quota in [('fit',args.fit_seed,args.fit_per_condition),('selection',411,args.selection_per_condition)]:
        spacing=args.fit_spacing_minutes if key=='fit' else 360
        old,metadata=sample_events(raw_for(release,args.panel,parts[key]),net,topo,args.panel,seed,per_condition=quota,spacing_minutes=spacing,onset_last_step=args.onset_last_step)
        new,new_meta=sample_events(raw_for(release,args.panel,parts[key]),net,topo,args.panel,seed,per_condition=quota,spacing_minutes=spacing,origin_reader=origin_reader,onset_last_step=args.onset_last_step)
        paired(old,new)
        data[key]={'before_T':old,'at_T':new}
        if args.bookkeeping_control:data[key]['before_T_latest']=bookkeeping(old)
        if args.occupancy_features:
            occ,_=sample_events(raw_for(release,args.panel,parts[key]),net,topo,args.panel,seed,per_condition=quota,spacing_minutes=spacing,origin_reader=origin_reader,include_occupancy=True,onset_last_step=args.onset_last_step)
            paired(old,occ)
            data[key]['at_T_occupancy']=occ
        if args.event_loss:
            data[key]['at_T_event']=new
            data[key]['at_T_balanced']=new
        if args.published_condition:
            data[key]['at_T_condition']=published(new)
            if args.occupancy_features:data[key]['at_T_occupancy_condition']=published(occ)
        samplers[key]=metadata
    result={'registered':registered,'source':evidence(),'dates':dates,'samplers':samplers,
        'selection':{},'assessment':[],'formal_Queue':None,
        'limitations':'No latent queue truth. Assessment seeds sample overlapping windows within the same days; they are not independent trials.'}
    models={}
    model_dir=Path('artifacts/research')/args.output.stem
    if model_dir.exists():raise FileExistsError('Model artifact directory already exists')
    model_dir.mkdir(parents=True)
    for mode in data['fit']:
        for iterations,leaves in settings:
            name=f'{mode}_{iterations}_{leaves}'
            kind=EventWeightedQueueModel if mode in ('at_T_event','at_T_balanced') else RefinedQueueModel
            extra={'class_balance':mode=='at_T_balanced'} if kind is EventWeightedQueueModel else {}
            model=kind(iterations=iterations,num_leaves=leaves,seed=args.fit_seed,**extra).fit(data['fit'][mode])
            model.uses_origin_timestamp=mode.startswith('at_T')
            model.uses_occupancy_features=mode.startswith('at_T_occupancy')
            model.uses_published_condition=mode.endswith('_condition')
            model.uses_public_onset_timing=args.onset_last_step
            model.latest_valid_queue_bookkeeping=mode.startswith('at_T') or mode=='before_T_latest'
            tuning=model.tune(data['selection'][mode])
            cal=data['selection'][mode]
            result['selection'][name]={'metrics':grouped_iou(cal.y,model.classify(cal.x,cal.window,cal.condition),cal.window,cal.condition),
                                       'threshold_tuning':tuning}
            models[name]=model
            model.save(model_dir/(name+'.pkl'))
    result['selected']=max(models,key=lambda name:result['selection'][name]['metrics']['proxy_IoU'])
    args.output.with_suffix('.selection.json').write_text(json.dumps({
        'registered_protocol_sha256':registered['protocol_sha256'],
        'selection':result['selection'],'selected':result['selected'],
        'note':'Sealed before assessment data sampling or prediction.'},indent=2),encoding='utf-8')
    # All fixed candidates are compared on the matched assessment window sets;
    # those numbers never change selected architecture or hyperparameters.
    for seed in cfg['simulator_assessment_seeds']:
        old,metadata=sample_events(raw_for(release,args.panel,parts['assessment']),net,topo,args.panel,seed,per_condition=args.assessment_per_condition,onset_last_step=args.onset_last_step)
        new,_=sample_events(raw_for(release,args.panel,parts['assessment']),net,topo,args.panel,seed,per_condition=args.assessment_per_condition,origin_reader=origin_reader,onset_last_step=args.onset_last_step)
        paired(old,new)
        variants={'before_T':old,'at_T':new}
        if args.bookkeeping_control:variants['before_T_latest']=bookkeeping(old)
        if args.occupancy_features:
            occ,_=sample_events(raw_for(release,args.panel,parts['assessment']),net,topo,args.panel,seed,per_condition=args.assessment_per_condition,origin_reader=origin_reader,include_occupancy=True,onset_last_step=args.onset_last_step)
            paired(old,occ)
            variants['at_T_occupancy']=occ
        if args.event_loss:
            variants['at_T_event']=new
            variants['at_T_balanced']=new
        if args.published_condition:
            variants['at_T_condition']=published(new)
            if args.occupancy_features:variants['at_T_occupancy_condition']=published(occ)
        row={'seed':seed,'sampler':metadata,'models':{}}
        for name,model in models.items():
            current=variants[name.rsplit('_',2)[0]]
            p=model.probability(current.x)
            row['models'][name]={**grouped_iou(current.y,model.classify(current.x,current.window,current.condition),current.window,current.condition),
                                'observation_proxy_Brier':float(np.mean((p-current.y)**2))}
        result['assessment'].append(row)
    result['seconds']=time.perf_counter()-start
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('selected',result['selected'],{name:np.mean([r['models'][name]['proxy_IoU'] for r in result['assessment']]) for name in models},flush=True)


if __name__=='__main__':main()
