"""Fixed zero-shot adaptive graph transfer; target truth is evaluation only."""
import argparse
import json
from pathlib import Path
import time
import pandas as pd
import torch
from trafficflow.adaptive_graph_state import AdaptiveStateModel
from trafficflow.state import StateModel,Profile
from trafficflow.data import Release,VALUES
from trafficflow.experiments import partition_date,evaluate_state
from trafficflow.research_eval import Fold,register_protocol,paired_block_interval
from trafficflow.runtime import sha256_file,evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--target',default='D7_I10_W')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite transfer evidence')
    source=json.loads((args.source/'experiment.json').read_text())
    source_panel=source['prepared']['panel']
    if source_panel.rsplit('_',1)[0]==args.target.rsplit('_',1)[0]:raise ValueError('Transfer requires an excluded family')
    config=source['registered']['protocol'];choice=source['selected']
    checkpoint=args.source/(choice['name']+'.pt')
    if sha256_file(checkpoint)!=source['model_sha256'][checkpoint.name]:raise ValueError('Source checkpoint changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Adaptive_graph_zero_shot_transfer','source_panel':source_panel,'target':args.target,
        'source_choice':choice,'checkpoint_sha256':sha256_file(checkpoint),
        'fold':source['prepared']['fold'],'target_profile':'Geometry-only free speed / .4 capacity',
        'target_fit_labels_used':False,'target_calibration':False,'ensemble_alphas':[0.,.5,1.],
        'assessment_selection':'No target-based choice. The fixed half ensemble is the preregistered comparison.',
        'inputs':'Target released masked State observations and network assets only',
        'formal_State':None,'formal_Physics':None})
    start=time.perf_counter()
    release=Release('data/kaggle_public')
    protocol=json.loads(Path('reports/research/protocol_v2_registered.json').read_text())['protocol']
    fold=Fold(**next(f for f in protocol['folds'] if f['name']==source['prepared']['fold']))
    paths=release.files(args.target,'train')
    dates=fold.assign([partition_date(release,p) for p in paths])
    paths=[p for p in paths if partition_date(release,p) in dates['assessment']]
    net,topo=release.network(args.target),release.topology(args.target)
    empty=pd.DataFrame(columns=VALUES,index=pd.MultiIndex.from_arrays([[],[]],names=['link_id','slot']),dtype=float)
    profile=Profile(empty,pd.DataFrame({'speed_kmh':net.free_speed_kmh,'flow_vph':net.capacity_vph*.4}))
    baseline=StateModel(profile,'interpolation')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    results={}
    for alpha in (0.,.5,1.):
        model=AdaptiveStateModel(baseline,state,topo,alpha,hidden=config['hidden'],attention=choice['name']!='graph_geometry')
        results[alpha]=evaluate_state(release,args.target,paths,model,0.,True)
    old,new=results[0.],results[.5]
    result={'registered':registered,'models':results,'assessment_dates':dates['assessment'],
        'fixed_half_pooled_State_gain':new['S_state']-old['S_state'],
        'fixed_half_paired_day_interval':paired_block_interval(
            [r['daily_State'] for r in old['daily_sufficient_statistics']],
            [r['daily_State'] for r in new['daily_sufficient_statistics']]),
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limitations':'One source/target family. Raw target public-train labels are accessed only by the scorer; no target historical profile or model fitting. Previously inspected development dates, not an untouched final test.'}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({a:r['S_state'] for a,r in results.items()},result['fixed_half_paired_day_interval'],result['seconds'],flush=True)
