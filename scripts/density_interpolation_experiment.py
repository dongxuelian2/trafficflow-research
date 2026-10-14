"""No-fit masked-density projection controls against strong State weights."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.data import Release, VALUES
from trafficflow.state import StateFeatures, StateModel
from trafficflow.density_state import DensityStateModel, InterpolatedDensityResidual
from trafficflow.experiments import partition_date, day_template
from trafficflow.research_eval import register_protocol, state_from_sufficient_stats, paired_block_interval
from trafficflow.runtime import evidence, capture_source_snapshot, sha256_file
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--baseline-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-state-loss',type=float,default=.005)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite density evidence')
    if not 0<=args.max_state_loss<=.02:raise ValueError('Invalid State tradeoff tolerance')
    info=json.loads((args.cache/'prepared.json').read_text())
    report=Path('reports/research')/(args.output.name+'.json')
    settings={'baseline':('base',0.)}
    settings.update({f'{mode}_{blend}':(mode,blend) for mode in ('base','temporal') for blend in (.1,.25,.5,1.)})
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'Masked_density_interpolation_control','panel':info['panel'],'fold':info['fold'],
        'baseline_file':args.baseline_file.as_posix(),'baseline_sha256':sha256_file(args.baseline_file),
        'hypothesis':'Temporal density anchors can improve accumulation-increment estimates independently of trained density regression.',
        'candidates':settings,'new_fitted_models':False,'inputs':'Published masked Task 1 observations only; offline future visible observations allowed',
        'State_selection':'Maximum calibration State; ties favor baseline',
        'joint_selection':f'Minimum calibration target-pair increment relative L1 with State >= baseline - {args.max_state_loss}',
        'assessment':'Only baseline and calibration-chosen candidates; existing public-train development dates',
        'formal_State':None,'formal_Physics':None,'no_official_flux':True})
    args.output.mkdir(parents=True)
    snapshot=capture_source_snapshot(args.output/'source_snapshot.zip')
    start=time.perf_counter()
    release=Release('data/kaggle_public')
    panel=info['panel']
    baseline=StateModel.load(args.baseline_file)
    net,topo=release.network(panel),release.topology(panel)
    paths={partition_date(release,p):p for p in release.files(panel,'train')}
    template=release.template(1,panel,'train')
    models={key:DensityStateModel(baseline,InterpolatedDensityResidual(mode),blend) for key,(mode,blend) in settings.items()}

    def evaluate(part,names):
        state={n:[] for n in names};increments={n:[] for n in names}
        for path in sorted((args.cache/part).glob('*.npz')):
            day=dict(np.load(path,allow_pickle=False))
            factor=(net.lanes*net.length_km).reindex(day['links']).to_numpy(float)
            frame=release.read(paths[path.stem])
            f=StateFeatures(frame,net,topo,baseline.profile,version=baseline.feature_version)
            target=day_template(release,panel,'train',frame,template)
            if not np.array_equal(f.lookup(target),day['target_indices']):raise ValueError('Cached target order changed')
            reference=baseline.predict(f,target)
            for name in names:
                p=models[name].project(f,target,reference)[VALUES].to_numpy(float)
                p[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
                state[name].append(stats(day,p))
                increments[name].append(increment_stats(day,p,factor))
        return {n:{**state_from_sufficient_stats(state[n]),
            'density_target_pair_relative_L1':sum(r['error_L1'] for r in increments[n])/sum(r['truth_increment_L1'] for r in increments[n]),
            'daily_State':[r['daily_State'] for r in state[n]],'n_target_pairs':sum(r['n_pairs'] for r in increments[n])} for n in names}

    selection=evaluate('selection',list(settings))
    chosen=max(selection,key=lambda n:selection[n]['S_state'])
    allowed=[n for n in selection if selection[n]['S_state']>=selection['baseline']['S_state']-args.max_state_loss]
    joint=min(allowed,key=lambda n:selection[n]['density_target_pair_relative_L1'])
    seal={'selection':selection,'State_selected':chosen,'joint_selected':joint}
    report.with_suffix('.selection.json').write_text(json.dumps(seal,indent=2),encoding='utf-8')
    for name,key in [('State_selected',chosen),('joint_selected',joint)]:models[key].save(args.output/(name+'.pkl'))
    assessment=evaluate('assessment',list(dict.fromkeys(['baseline',chosen,joint])))
    result={'registered':registered,'source_snapshot':snapshot,**seal,'assessment':assessment,
        'joint_paired_day_interval':paired_block_interval(assessment['baseline']['daily_State'],assessment[joint]['daily_State']),
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_State':None,'formal_Physics':None,
        'limitation':'Observed accumulation increments are a supervised diagnostic. They are not organizer boundary-flow residuals, and .005 State tolerance is a preregistered tradeoff rather than a total-score guarantee.'}
    report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.output/'experiment.json').write_text(json.dumps({'completed_report':report.as_posix(),
        'protocol_sha256':registered['protocol_sha256'],'seconds':result['seconds']},indent=2),encoding='utf-8')
    print(panel,'State',chosen,'joint',joint,[(n,r['S_state'],r['density_target_pair_relative_L1']) for n,r in assessment.items()],result['seconds'],flush=True)


if __name__=='__main__':main()
