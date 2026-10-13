"""Same-fold production-like 500k vs full-label State controls by direction."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.state import Profile,StateModel
from trafficflow.experiments import partition_date,raw_for,training_rows,evaluate_state
from trafficflow.research_eval import Fold,register_protocol,paired_block_interval
from trafficflow.runtime import evidence,capture_source_snapshot


def budget_indices(release,panel,paths,total_rows,max_small,max_full,seed):
    template=release.template(1,panel,'train')
    counts=template.groupby(template.timestamp.str[:10]).size().to_dict()
    large_per_day=max(1,max_full//len(paths))
    small_per_day=max(1,max_small//len(paths))
    indices=[]
    offset=0
    rng=np.random.default_rng(seed)
    for path in paths:
        n=int(counts.get(partition_date(release,path),0))
        if n>large_per_day:raise ValueError('Full comparator must cover every fitting target')
        take=np.sort(rng.choice(n,small_per_day,replace=False)) if n>small_per_day else np.arange(n)
        indices.append(offset+take)
        offset+=n
    if offset!=total_rows:raise ValueError('Training row count/order mismatch')
    return np.concatenate(indices)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--panels',nargs='+',required=True)
    parser.add_argument('--fold',default='rolling_autumn')
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite budget suites')
    protocol=json.loads(Path('reports/research/protocol_v2_registered.json').read_text())
    fold=Fold(**next(f for f in protocol['protocol']['folds'] if f['name']==args.fold))
    registered=register_protocol(Path('reports/research')/(args.output.name+'.registered.json'),
        {'id':'State_500k_vs_full_suite','panels':args.panels,'fold':fold.__dict__,'seed':args.seed,
         'small_max_labels':500000,'full_max_labels':10000000,'feature_version':2,
         'profile':'Public raw data strictly before fit; same profile for both models',
         'model':'LGB 400 iterations / 63 leaves; all other settings identical',
         'selection':'Budget hypothesis fixed; no assessment-driven hyperparameters',
         'formal_State':None,'formal_Physics':None})
    args.output.mkdir(parents=True)
    snapshot=capture_source_snapshot(args.output/'source_snapshot.zip')
    release=Release('data/kaggle_public')
    for panel in args.panels:
        start=time.perf_counter()
        folder=args.output/panel
        folder.mkdir()
        paths=release.files(panel,'train')
        dates=fold.assign([partition_date(release,p) for p in paths])
        parts={key:[p for p in paths if partition_date(release,p) in ds] for key,ds in dates.items()}
        profile=Profile.fit(raw_for(release,panel,parts['profile']))
        x,y,base=training_rows(release,panel,parts['fit'],profile,10000000,args.seed,2)
        small=budget_indices(release,panel,parts['fit'],len(y),500000,10000000,args.seed)
        result={'registered':registered,'source_snapshot':snapshot,'panel':panel,'dates':dates,
                'fitting_labels':{'500k':len(small),'full':len(y)},'selection':{},'assessment':{}}
        models={}
        for name,indices in [('500k',small),('full',None)]:
            model=StateModel(profile,'lightgbm',seed=args.seed,iterations=400,feature_version=2,num_leaves=63)
            model.density_blend=0.
            if indices is None:model.fit(x,y,base)
            else:model.fit(x.iloc[indices],y[indices],base[indices])
            model.save(folder/(name+'.pkl'))
            models[name]=model
        del x,y,base,small
        for name,model in models.items():
            result['selection'][name]=evaluate_state(release,panel,parts['selection'],model,0.,True)
        # Architecture/settings are fixed in the registered budget comparison.
        (folder/'selection_seal.json').write_text(json.dumps(result['selection'],indent=2),encoding='utf-8')
        for name,model in models.items():
            result['assessment'][name]=evaluate_state(release,panel,parts['assessment'],model,0.,True)
        old=result['assessment']['500k'];new=result['assessment']['full']
        result['pooled_State_gain']=new['S_state']-old['S_state']
        result['paired_day_interval']=paired_block_interval(
            [d['daily_State'] for d in old['daily_sufficient_statistics']],
            [d['daily_State'] for d in new['daily_sufficient_statistics']])
        result['seconds']=time.perf_counter()-start
        result['source']=evidence()
        result['formal_State']=None;result['formal_Physics']=None
        result['limitations']='Same-fold public-train development comparison, no latent labels/official flux. Existing assessment periods have been inspected.'
        (folder/'experiment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        output=Path('reports/research')/(args.output.name+'_'+panel+'.json')
        if output.exists():raise FileExistsError('Never overwrite a direction report')
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(panel,old['S_state'],new['S_state'],result['pooled_State_gain'],result['seconds'],flush=True)


if __name__=='__main__':main()
