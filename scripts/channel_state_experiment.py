"""Cal-only per-channel tree/NN ensemble;new calendar fixed before choice."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,VALUES
from trafficflow.state import StateModel,StateFeatures
from trafficflow.channel_state import ChannelStateEnsemble
from trafficflow.experiments import partition_date,day_template,target_truth
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats,paired_block_interval
from trafficflow.runtime import capture_source_snapshot,sha256_file,evidence
from train_graph_experiment import stats
from recurrent_state_experiment import summarize,transition_masks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite channel policy study')
    tree_path=Path('reports/research/A103_I210_W_native_remask_LGB_control.json');tree=json.loads(tree_path.read_text())
    proof_path=Path('reports/research/A110_I210_W_remask_tree_independent_verification.json');proof=json.loads(proof_path.read_text())
    if proof['registered']['protocol']['source_study_sha256']!=sha256_file(tree_path):raise ValueError('Component deployment not verified')
    views=json.loads(Path('artifacts/research/A96_I210_W_native_remask_fit_views/cache.json').read_text())
    original=json.loads(Path(views['registered']['protocol']['source_study']).read_text());old=original['registered']['protocol'];cache=Path(old['cache'])
    source=Path('artifacts/research')/tree_path.stem;nn=tree['registered']['protocol']['NN_primary_export']
    component_paths={'nn_primary':Path(nn['path']),'point_seed_mean':source/'State_point_seed_mean.pkl','queue_seed_mean':source/'State_queue_seed_mean.pkl'}
    if sha256_file(component_paths['nn_primary'])!=nn['sha256']:raise ValueError('Frozen NN changed')
    dates=pd.date_range('2031-02-06','2031-02-19').strftime('%Y-%m-%d').tolist()
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Cal_only_fixed_channel_ensemble','tree_study_sha256':sha256_file(tree_path),'component_verification_sha256':sha256_file(proof_path),
        'panel':'D7_I210_W','cal_evidence_motivation':'Inspected development cal:trees lower flowRMSE but worse speedRMSE;not a blind hypothesis',
        'grid':{'tree_variant':['point','queue'],'speed_tree_weight':[0,.25,.5,.75,1.],'flow_tree_weight':[0,.25,.5,.75,1.]},
        'primary':'Highest cal eligible score;requires>=.0005State gain and no regime cal loss>.001,else keep NN',
        'choice_scope':'Constant per-channel weights,never target/error-specific experts;no model refit or new labels',
        'assessment':'Already-viewed original assessment,read only after channel choice seal;regression only',
        'future_dates':dates,'future_scope':'Fixed before channel choice/new policy evaluation;other past public studies possible,not final/private',
        'components':{n:{'path':p.as_posix(),'sha256':sha256_file(p)} for n,p in component_paths.items()},
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    components={n:StateModel.load(p) for n,p in component_paths.items()};profile=components['nn_primary'].profile;version=components['nn_primary'].feature_version
    release=Release('data/kaggle_public');panel='D7_I210_W';net,topology=release.network(panel),release.topology(panel)
    files={partition_date(release,p):p for p in release.files(panel,'train')};raw_paths={p.name:p for p in release.files(panel,'train',False)};template=release.template(1,panel,'train')
    factor=None;read_order=[];source_hashes={}
    def load(part,seal=None):
        nonlocal factor
        if part=='assessment' and (seal is None or sha256_file(folder/'selection_sealed.json')!=seal):raise ValueError('Assessment before choice seal')
        read_order.append({'part':part,'prior_selection_seal_sha256':seal});days=[];values={n:[] for n in components}
        for path in sorted((cache/part).glob('*.npz')):
            day=dict(np.load(path));day['date']=path.stem;frame=release.read(files[path.stem]);f=StateFeatures(frame,net,topology,profile,version)
            target=day_template(release,panel,'train',frame,template);raw=release.read(raw_paths[files[path.stem].name])
            teacher=target_truth(release,panel,target,raw)[VALUES].to_numpy(float);teacher[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
            if not np.array_equal(teacher,day['station_truth']):raise ValueError('Cached public teacher differs')
            day['features'],day['template']=f,target;day['diagnostic_masks']=transition_masks(day,target,raw);days.append(day)
            pred=source/'predictions'/part/(path.stem+'.npz');source_hashes[pred.as_posix()]=sha256_file(pred);source_hashes[path.as_posix()]=sha256_file(path)
            with np.load(pred) as a:
                for n in values:values[n].append(a[n])
            if factor is None:factor=(net.lanes*net.length_km).reindex(f.links).to_numpy(np.float32)
        return days,values
    def mix(values,variant,speed,flow):
        weight=np.array([speed,flow]);return [(1-weight)*a+weight*b for a,b in zip(values['nn_primary'],values[variant+'_seed_mean'])]
    def simple(days,values):
        rows=[dict(stats(d,p),date=d['date']) for d,p in zip(days,values)]
        return {**state_from_sufficient_stats(rows),'days':rows}
    cal,cal_values=load('selection');baseline=simple(cal,cal_values['nn_primary']);grid=[]
    for variant in ['point','queue']:
        for speed in [0.,.25,.5,.75,1.]:
            for flow in [0.,.25,.5,.75,1.]:
                metrics=simple(cal,mix(cal_values,variant,speed,flow));gain=metrics['S_state']-baseline['S_state']
                minimum=min(a['S_state']-b['S_state'] for a,b in zip(metrics['details'],baseline['details']))
                grid.append({'variant':variant,'speed_tree_weight':speed,'flow_tree_weight':flow,'cal':metrics,'cal_gain':gain,
                    'minimum_regime_cal_gain':minimum,'eligible':gain>=.0005 and minimum>=-.001})
    eligible=[g for g in grid if g['eligible']]
    chosen=max(eligible,key=lambda g:(g['cal']['S_state'],-g['speed_tree_weight']-g['flow_tree_weight'])) if eligible else None
    selection={'choice':chosen,'kept_NN':chosen is None,'baseline_cal':baseline,'cal_grid':grid}
    seal=folder/'selection_sealed.json';seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8');seal_sha=sha256_file(seal)
    if chosen:
        v=chosen['variant'];av,aq=chosen['speed_tree_weight'],chosen['flow_tree_weight']
        model=ChannelStateEnsemble([components['nn_primary'],components[v+'_seed_mean']],[[1-av,1-aq],[av,aq]])
    else:model=components['nn_primary']
    model_path=folder/'State_primary.pkl';model.save(model_path)
    print('channel_PRIMARY_SEALED',None if chosen is None else {k:chosen[k] for k in ['variant','speed_tree_weight','flow_tree_weight','cal_gain']},flush=True)
    assessment,ass_values=load('assessment',seal_sha);all_metrics={};prediction_files=[];direct=[];maximum=0.
    for part,days,values in [('selection',cal,cal_values),('assessment',assessment,ass_values)]:
        selected=values['nn_primary'] if chosen is None else mix(values,chosen['variant'],chosen['speed_tree_weight'],chosen['flow_tree_weight'])
        values['primary_export']=selected;all_metrics[part]={n:summarize(days,p,factor) for n,p in values.items()}
        out=folder/'predictions'/part;out.mkdir(parents=True)
        for i,d in enumerate(days):
            p=model.predict(d['features'],d['template'])[VALUES].to_numpy(float);p[:,1]/=d['target_link_lanes']
            error=float(np.abs(p-selected[i]).max());maximum=max(maximum,error)
            if error>1e-8:raise ValueError('Serialized channel model differs from fixed channel arrays')
            direct.append({'part':part,'date':d['date'],'maximum_lane_unit_error':error})
            path=out/(d['date']+'.npz');np.savez_compressed(path,**{n:p[i] for n,p in values.items()})
            prediction_files.append({'part':part,'date':d['date'],'path':path.as_posix(),'sha256':sha256_file(path)})
    a,b=all_metrics['assessment']['nn_primary'],all_metrics['assessment']['primary_export']
    comparison={'pooled_State_gain':b['S_state']-a['S_state'],
        'paired_day_diagnostic':paired_block_interval([r['daily_State'] for r in a['days']],[r['daily_State'] for r in b['days']])}
    result={'registered':registered,'source_snapshot':snapshot,'selection':selection,'selection_seal_sha256':seal_sha,'metrics':all_metrics,
        'assessment_comparison':comparison,'prediction_files':prediction_files,'source_files_sha256':source_hashes,
        'primary_export':{'path':model_path.as_posix(),'sha256':sha256_file(model_path)},'phase_read_order':read_order,
        'n_direct_selected_predictor_calls':len(direct),'direct_checks':direct,'maximum_lane_unit_prediction_error':maximum,
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Cal used for development/choice;original assessment viewed. Per-channel combination may change q/v dynamics;legacy increment proxy not formal Physics. No hidden-label inference.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('channel_COMPLETE','assessment_gain',comparison['pooled_State_gain'],'calls',len(direct),'maxpred',maximum,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
