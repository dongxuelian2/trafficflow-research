"""Learn public-train density transitions; evaluate only independent time groups."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from trafficflow.data import Release,VALUES
from trafficflow.state import StateFeatures,StateModel
from trafficflow.increment_state import pair_features,IncrementStateModel,PAIR_COLUMNS
from trafficflow.experiments import partition_date,day_template
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats,paired_block_interval
from trafficflow.runtime import evidence
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=Path('artifacts/research/A01_D12_I5_N_autumn_seed20261013'))
    parser.add_argument('--baseline',type=Path,default=Path('artifacts/research/A14_full_label_LGB_control'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-pairs',type=int,default=1500000)
    parser.add_argument('--objective',choices=['regression','regression_l1'],default='regression_l1')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite transition evidence')
    prepared=json.loads((args.cache/'prepared.json').read_text())
    report=Path('reports/research')/(args.output.name+'.json')
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'B02_learned_density_increments','panel':prepared['panel'],'fold':prepared['fold'],
        'fit_label_source':'Public train density increments at valid adjacent pairs touching State targets',
        'features':PAIR_COLUMNS,'max_pairs':args.max_pairs,'objective':args.objective,
        'weights':'link lane count times length, normalized to mean one',
        'strength_grid':[0,.1,1,10,100],
        'State_selection':'Maximum selection State, ties favor lower strength',
        'joint_selection':'Minimum selection target-pair increment relative L1 with State >= baseline -.002',
        'visible_nodes_fixed':True,'extra_forecast_blanks':'Unavailable nodes; excluded',
        'formal_State':None,'formal_Physics':None})
    args.output.mkdir(parents=True)
    start=time.perf_counter()
    release=Release('data/kaggle_public')
    panel=prepared['panel']
    old=StateModel.load(args.baseline/'lightgbm_large_budget.pkl')
    net,topo=release.network(panel),release.topology(panel)
    paths={partition_date(release,p):p for p in release.files(panel,'train')}
    factor=None
    rng=np.random.default_rng(prepared['seed'])
    xs,ys,ws=[],[],[]
    fit=list(sorted((args.cache/'fit').glob('*.npz')))
    per_day=max(1,args.max_pairs//len(fit))
    for path in fit:
        day=dict(np.load(path,allow_pickle=False))
        frame=release.read(paths[path.stem])
        f=StateFeatures(frame,net,topo,old.profile,version=old.feature_version)
        x,valid,_=pair_features(f,day['mask'])
        truth=np.where(day['mask'][...,None],day['y'],day['x'][...,:2])
        density=truth[...,1]*600/np.maximum(truth[...,0]*25,1.)
        delta=np.diff(density,axis=0)/25
        factor=(net.lanes*net.length_km).reindex(day['links']).to_numpy(float)
        weight=np.broadcast_to(factor,valid.shape).ravel()
        indices=np.flatnonzero(valid.ravel())
        if len(indices)>per_day:indices=np.sort(rng.choice(indices,per_day,replace=False))
        xs.append(x.iloc[indices]);ys.append(delta.ravel()[indices]);ws.append(weight[indices])
    x=pd.concat(xs,ignore_index=True)
    y=np.concatenate(ys);weight=np.concatenate(ws);weight/=weight.mean()
    model=LGBMRegressor(n_estimators=400,learning_rate=.035,num_leaves=63,min_child_samples=80,
        reg_lambda=5.,colsample_bytree=.9,random_state=prepared['seed'],n_jobs=4,verbosity=-1,
        deterministic=True,force_col_wise=True,objective=args.objective)
    model.fit(x,y,sample_weight=weight)
    n=len(y)
    del x,y,weight,xs,ys,ws
    combined=IncrementStateModel(old,model)
    combined.save(args.output/'transition_state.pkl')
    template=release.template(1,panel,'train')

    def evaluate(part,strengths):
        state={s:[] for s in strengths}
        increments={s:[] for s in strengths}
        for path in sorted((args.cache/part).glob('*.npz')):
            day=dict(np.load(path,allow_pickle=False))
            frame=release.read(paths[path.stem])
            f=StateFeatures(frame,net,topo,old.profile,version=old.feature_version)
            target=day_template(release,panel,'train',frame,template)
            if not np.array_equal(f.lookup(target),day['target_indices']):raise ValueError('Cached target order changed')
            for strength in strengths:
                combined.strength=strength
                p=combined.predict(f,target)[VALUES].to_numpy(float)
                p[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
                state[strength].append(stats(day,p))
                increments[strength].append(increment_stats(day,p,factor))
        return {s:{**state_from_sufficient_stats(state[s]),
            'density_target_pair_relative_L1':sum(r['error_L1'] for r in increments[s])/sum(r['truth_increment_L1'] for r in increments[s]),
            'daily_State':[r['daily_State'] for r in state[s]]} for s in strengths}

    selection=evaluate('selection',registered['protocol']['strength_grid'])
    chosen=max(selection,key=lambda s:(selection[s]['S_state'],-s))
    eligible=[s for s in selection if selection[s]['S_state']>=selection[0]['S_state']-.002]
    joint=min(eligible,key=lambda s:(selection[s]['density_target_pair_relative_L1'],s))
    seal={'selection':selection,'State_strength':chosen,'joint_strength':joint}
    report.with_suffix('.selection.json').write_text(json.dumps(seal,indent=2),encoding='utf-8')
    assessment=evaluate('assessment',sorted({0,chosen,joint}))
    for name,strength in [('State_selected',chosen),('joint_selected',joint)]:
        combined.strength=strength
        combined.save(args.output/(name+'.pkl'))
    result={'registered':registered,'n_fitting_pairs':n,**seal,'assessment':assessment,
        'joint_paired_day_interval':paired_block_interval(assessment[0]['daily_State'],assessment[joint]['daily_State']),
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_State':None,'formal_Physics':None,
        'limitations':'One fit seed/direction/fold; public-train development. Learned increment labels are observation-derived, no official boundary flux. Geometry bounds can perturb the solved density ratio.'}
    report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(chosen,joint,{s:(r['S_state'],r['density_target_pair_relative_L1']) for s,r in assessment.items()},result['seconds'],flush=True)


if __name__=='__main__':main()
