"""Check whether graph gains are explained by LightGBM's label subsampling."""
import argparse
import json
from pathlib import Path
import time
from trafficflow.data import Release
from trafficflow.experiments import training_rows, partition_date, evaluate_state
from trafficflow.state import StateModel
from trafficflow.runtime import evidence
from trafficflow.research_eval import register_protocol

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=Path('artifacts/research/A01_D12_I5_N_autumn_seed20261013'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-labels',type=int,default=3000000)
    parser.add_argument('--registration',type=Path)
    parser.add_argument('--feature-version',type=int,default=2,choices=[2,3])
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use new budget experiment directory')
    prepared=json.loads((args.cache/'prepared.json').read_text())
    seed=prepared['seed'] if args.seed is None else args.seed
    registration=args.registration or Path('reports/research')/(args.output.name+'.registered.json')
    registered=register_protocol(registration,
        {'id':'A14_LGB_label_budget','panel':prepared['panel'],'fold':prepared['fold'],
        'original_max_labels':300000,'new_max_labels':args.max_labels,'architecture_changed':False,
        'profile_changed':False,'seed_changed':seed!=prepared['seed'],'fit_seed':seed,'all_other_hyperparameters_changed':False,
        'feature_version':args.feature_version,'additional_published_channels':args.feature_version==3,
        'formal_State':None})
    args.output.mkdir(parents=True)
    start=time.perf_counter()
    release=Release('data/kaggle_public')
    old=StateModel.load(args.cache/'lightgbm.pkl')
    paths=release.files(prepared['panel'],'train')
    parts={key:[p for p in paths if partition_date(release,p) in dates]
           for key,dates in prepared['dates'].items()}
    model=StateModel(old.profile,'lightgbm',seed=seed,iterations=400,feature_version=args.feature_version,num_leaves=63)
    model.density_blend=0.
    x,y,base=training_rows(release,prepared['panel'],parts['fit'],old.profile,args.max_labels,seed,args.feature_version)
    n=len(y)
    model.fit(x,y,base)
    model.save(args.output/'lightgbm_large_budget.pkl')
    del x,y,base
    selection=evaluate_state(release,prepared['panel'],parts['selection'],model,0.,True)
    assessment=evaluate_state(release,prepared['panel'],parts['assessment'],model,0.,True)
    result={'registered':registered,'n_fitting_labels':n,'selection':selection,'assessment':assessment,
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_State':None,
        'limits':'Single direction/fold/seed. Development assessment, not a newly untouched final test.'}
    (args.output/'experiment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(n,selection['S_state'],assessment['S_state'],result['seconds'],flush=True)
