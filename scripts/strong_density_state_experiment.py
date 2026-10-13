"""Recheck direct density prediction against the full-label State baseline."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from lightgbm import LGBMRegressor
from trafficflow.data import Release,VALUES
from trafficflow.state import StateFeatures,StateModel
from trafficflow.density_state import DensityStateModel
from trafficflow.experiments import partition_date,training_rows,day_template
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats,paired_block_interval
from trafficflow.runtime import evidence
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=Path('artifacts/research/A01_D12_I5_N_autumn_seed20261013'))
    parser.add_argument('--baseline',type=Path,default=Path('artifacts/research/A14_full_label_LGB_control'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite density experiments')
    args.output.mkdir(parents=True)
    prepared=json.loads((args.cache/'prepared.json').read_text())
    budget=json.loads((args.baseline/'experiment.json').read_text())
    report=Path('reports/research')/(args.output.name+'.json')
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'A19_strong_density_expectation','panel':prepared['panel'],'fold':prepared['fold'],
        'fit_labels':budget['n_fitting_labels'],'seed':prepared['seed'],
        'baseline_weights_reused':args.baseline.as_posix(),'density_target':'public train lane flow / max(observed speed,1)',
        'blend_grid':[0,.05,.1,.25,.5,1],
        'State_selection':'maximum selection State; tie favors zero blend',
        'joint_selection':'minimum target-pair density-increment L1 among selection State >= baseline - .002',
        'projection':'geometry-bound baseline first; weighted quadratic projection, then final bounds',
        'formal_State':None,'formal_Physics':None})
    start=time.perf_counter()
    release=Release('data/kaggle_public')
    panel=prepared['panel']
    old=StateModel.load(args.baseline/'lightgbm_large_budget.pkl')
    paths=release.files(panel,'train')
    fit=[p for p in paths if partition_date(release,p) in prepared['dates']['fit']]
    x,y,_=training_rows(release,panel,fit,old.profile,budget['registered']['protocol']['new_max_labels'],prepared['seed'],old.feature_version)
    density=y[:,1]/np.maximum(y[:,0],1.)
    residual=(density-x.density_base.to_numpy(float)*25)/25
    dm=LGBMRegressor(**old.models[0].get_params())
    dm.fit(x,residual)
    n=len(y)
    del x,y,density,residual
    model=DensityStateModel(old,dm)
    model.save(args.output/'density_model.pkl')
    net,topo=release.network(panel),release.topology(panel)
    factor=None
    template=release.template(1,panel,'train')
    by_date={partition_date(release,p):p for p in paths}

    def evaluate(part,blends):
        state={b:[] for b in blends}
        inc={b:[] for b in blends}
        nonlocal factor
        for path in sorted((args.cache/part).glob('*.npz')):
            day=dict(np.load(path,allow_pickle=False))
            if factor is None:factor=(net.lanes*net.length_km).reindex(day['links']).to_numpy(float)
            frame=release.read(by_date[path.stem])
            f=StateFeatures(frame,net,topo,old.profile,version=old.feature_version)
            target=day_template(release,panel,'train',frame,template)
            if not np.array_equal(f.lookup(target),day['target_indices']):raise ValueError('Target order changed')
            for b in blends:
                model.blend=b
                p=model.predict(f,target)[VALUES].to_numpy(float)
                p[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
                state[b].append(stats(day,p))
                inc[b].append(increment_stats(day,p,factor))
        return {b:{**state_from_sufficient_stats(state[b]),
             'density_target_pair_relative_L1':sum(d['error_L1'] for d in inc[b])/sum(d['truth_increment_L1'] for d in inc[b]),
             'daily_State':[d['daily_State'] for d in state[b]]} for b in blends}

    selection=evaluate('selection',registered['protocol']['blend_grid'])
    best=max(selection,key=lambda b:(selection[b]['S_state'],-b))
    eligible=[b for b in selection if selection[b]['S_state']>=selection[0]['S_state']-.002]
    joint=min(eligible,key=lambda b:(selection[b]['density_target_pair_relative_L1'],b))
    seal={'selection':selection,'State_blend':best,'joint_blend':joint}
    report.with_suffix('.selection.json').write_text(json.dumps(seal,indent=2),encoding='utf-8')
    assessment=evaluate('assessment',sorted({0,best,joint}))
    for name,b in [('State_selected',best),('joint_selected',joint)]:
        model.blend=b
        model.save(args.output/(name+'.pkl'))
    result={'registered':registered,'fitting_labels':n,**seal,'assessment':assessment,
            'joint_paired_day_interval':paired_block_interval(assessment[0]['daily_State'],assessment[joint]['daily_State']),
            'seconds':time.perf_counter()-start,'source':evidence(),
            'formal_State':None,'formal_Physics':None,
            'limitation':'One fold/direction/seed. Public-train target-pair density diagnostic, not official conservation scoring; prior joint-LightGBM first-round failure retained.'}
    report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(best,joint,{b:(r['S_state'],r['density_target_pair_relative_L1']) for b,r in assessment.items()},result['seconds'],flush=True)


if __name__=='__main__':main()
