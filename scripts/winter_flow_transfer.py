"""Two flow-only tree fits and a channel policy transferred without weight search."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from trafficflow.state import StateModel,StateFeatures
from trafficflow.bounded_state import BoundedState
from trafficflow.state_ensemble import StateEnsemble
from trafficflow.channel_state import ChannelStateEnsemble
from trafficflow.native_remask import native_remask
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.data import Release,VALUES
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol,paired_block_interval
from recurrent_state_experiment import summarize
from flow_transfer_common import compose,PublicDays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Never overwrite flow transfer')
    views_path=Path('artifacts/research/A102_I5_N_native_remask_fit_views/cache.json');views=json.loads(views_path.read_text())
    original_path=Path(views['registered']['protocol']['source_study']);original=json.loads(original_path.read_text());old=original['registered']['protocol']
    nn_path=Path('reports/research/A104_I5_N_native_remask_ablation.json');nn=json.loads(nn_path.read_text());nn_folder=Path('artifacts/research')/nn_path.stem
    proof_path=Path('reports/research/A107_I5_N_native_remask_training_independent_replay.json');proof=json.loads(proof_path.read_text())
    hypothesis_path=Path('reports/research/A112_I210_W_channel_state_ensemble.json');hypothesis=json.loads(hypothesis_path.read_text())
    if proof['registered']['protocol']['source_study_sha256']!=sha256_file(nn_path) or proof['n_role_day_predictor_calls']!=336:raise ValueError('Winter NN not verified')
    if nn['registered']['protocol']['views_report_sha256']!=sha256_file(views_path) or old['panel']!='D12_I5_N':raise ValueError('Matched winter views required')
    h=hypothesis['selection']['choice']
    if (h['variant'],h['speed_tree_weight'],h['flow_tree_weight'])!=('point',0.,1.):raise ValueError('Transfer hypothesis differs')
    baseline=StateModel.load(Path(old['baseline_file']));nn_export=proof['exports']['primary_export']
    if sha256_file(old['baseline_file'])!=old['baseline_sha256'] or sha256_file(nn_export['path'])!=nn_export['sha256']:raise ValueError('Frozen models changed')
    frozen_nn=StateModel.load(Path(nn_export['path']))
    if frozen_nn.feature_version!=baseline.feature_version or not frozen_nn.profile.slots.equals(baseline.profile.slots) or not frozen_nn.profile.fallback.equals(baseline.profile.fallback):raise ValueError('Transfer profiles/features differ')
    params=baseline.models[1].get_params();columns=baseline.models[1].feature_name_;cache=Path(old['cache'])
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'I210_to_I5_fixed_channel_flow_only_transfer','panel':old['panel'],'views':views_path.parent.as_posix(),
        'view_report_sha256':sha256_file(views_path),'source_study':original_path.as_posix(),'source_study_sha256':sha256_file(original_path),
        'NN_study':nn_path.as_posix(),'NN_study_sha256':sha256_file(nn_path),'NN_replay_sha256':sha256_file(proof_path),'NN_primary_export':nn_export,
        'hypothesis_source':hypothesis_path.as_posix(),'hypothesis_source_sha256':sha256_file(hypothesis_path),
        'fixed_transfer':'NN speed,point tree two-seed mean flow;no winter channel-weight search',
        'seeds':[20261013,20261029],'flow_tree_parameters':params,'feature_version':baseline.feature_version,
        'data':'All original fit targets plus4point remasks;only public raw-fit labels;original45day profile fixed',
        'weight':'Half original,half4views;uniform regime/date/row;effective mass original target count,as A103',
        'memory':'Preallocate disk-mapped float32 features,float64 scalar teacher/base/weight,int32row keys;chunk-by-view fill',
        'speed_tree':'Reuse original baseline speed tree unchanged,only as placeholder in flow models;transfer speed comes from frozen NN',
        'cal_gate':'Fixed transfer accepted only if pooled cal gain>=.0005 and no regime cal loss>.001;else keep NN. No other weights tested',
        'assessment':'Old public assessment regression only,read after all fits and cal seal',
        'future_dates':pd.date_range('2031-02-06','2031-02-19').strftime('%Y-%m-%d').tolist(),
        'future_scope':'Fixed before new flow fits;public research dates,prior unrelated studies possible,not final/private',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');panel=old['panel'];net,topology=release.network(panel),release.topology(panel)
    original_count=sum(r['original_target_grid_cells'] for r in views['rows']);counts={r:sum(d['regime']==r for d in views['rows']) for r in ['R1','R2','R3']}
    groups=[];cursor=0
    for date_index,row in enumerate(views['rows']):
        for path,digest in [(row['path'],row['sha256']),(row['original_cache_path'],original['cache_sha256'][row['original_cache_path']])]:
            if sha256_file(path)!=digest:raise ValueError('Public fit/cache changed')
        with np.load(row['path']) as a, np.load(row['original_cache_path']) as b:
            for view in [-1,0,1,2,3]:
                mask=b['mask'] if view<0 else a['mask_point'][view];n=int(mask.sum())
                groups.append({'date':row['date'],'date_index':date_index,'regime':row['regime'],'view':view,'start':cursor,'end':cursor+n});cursor+=n
    data_folder=args.output/'point_flow';data_folder.mkdir();nrows=cursor
    arrays={};shapes={'features':(nrows,len(columns)),'flow_teacher':(nrows,),'flow_base':(nrows,),'weight':(nrows,),'row_keys':(nrows,3)}
    for key,shape in shapes.items():
        arrays[key]=np.lib.format.open_memmap(data_folder/(key+'.npy'),mode='w+',dtype=np.float32 if key=='features' else np.int32 if key=='row_keys' else np.float64,shape=shape)
    build_start=time.perf_counter();source_hashes={}
    for date_index,row in enumerate(views['rows']):
        frame=release.read(row['masked_file']);raw=release.read(row['raw_file'])
        for path in [row['masked_file'],row['raw_file']]:source_hashes[path]=sha256_file(path)
        if 'mask_regime' not in raw:raw['mask_regime']=row['regime']
        orig=dict(np.load(row['original_cache_path']));remasked=dict(np.load(row['path']))
        for group in groups[date_index*5:(date_index+1)*5]:
            view=group['view'];begin,end=group['start'],group['end']
            if view<0:input_frame=frame;mask=orig['mask'];teacher=orig['y'];mass=.5
            else:
                input_frame,_,_=native_remask(raw,panel,row['date'],views['registered']['protocol']['view_seeds'][view],row['regime'],row['public_origins'],False)
                mask=remasked['mask_point'][view];teacher=remasked['y'];mass=.125
            f=StateFeatures(input_frame,net,topology,baseline.profile,baseline.feature_version)
            if f.features.columns.tolist()!=columns or (view>=0 and not np.array_equal(reconstruction_input(f),remasked['x_point'][view])):raise ValueError('Native input/features changed')
            index=np.flatnonzero(mask.ravel())
            arrays['features'][begin:end]=f.features.iloc[index].to_numpy(np.float32)
            arrays['flow_teacher'][begin:end]=teacher.reshape(-1,2)[index,1].astype(float)*600
            arrays['flow_base'][begin:end]=f.base.reshape(-1,2)[index,1]
            arrays['weight'][begin:end]=mass/(3*counts[row['regime']]*len(index))*original_count
            arrays['row_keys'][begin:end]=np.c_[np.full(len(index),date_index),np.full(len(index),view),index]
        if (date_index+1)%15==0:print('flow_table',date_index+1,'of',len(views['rows']),'peakMiB',evidence()['process_peak_working_set_MiB'],flush=True)
    for a in arrays.values():a.flush()
    dataset={'rows':nrows,'columns':columns,'groups':groups,'n_original_rows':original_count,'effective_weight_sum':float(arrays['weight'].sum()),
        'files':{k:{'path':(data_folder/(k+'.npy')).as_posix(),'sha256':sha256_file(data_folder/(k+'.npy'))} for k in arrays},
        'source_files_sha256':source_hashes,'preparation_seconds':time.perf_counter()-build_start}
    (args.output/'dataset.json').write_text(json.dumps(dataset,indent=2,allow_nan=False),encoding='utf-8')
    x=pd.DataFrame(arrays['features'],columns=columns,copy=False);residual=(arrays['flow_teacher']-arrays['flow_base'])/600
    models={'old_lgb':BoundedState(baseline)};training={}
    for seed in [20261013,20261029]:
        tic=time.perf_counter();kwargs=params.copy();kwargs.update(random_state=seed,n_jobs=4)
        tree=LGBMRegressor(**kwargs);tree.fit(x,residual,sample_weight=arrays['weight'])
        model=StateModel(baseline.profile,'lightgbm',seed,baseline.iterations,4,baseline.feature_version,baseline.num_leaves)
        model.models=[baseline.models[0],tree];role=f'flow_seed{seed}';bounded=BoundedState(model);path=args.output/('State_'+role+'.pkl');bounded.save(path);models[role]=bounded
        training[role]={'seconds':time.perf_counter()-tic,'new_channels_fit':1,'speed_tree_reused':True,'model_path':path.as_posix(),
            'model_sha256':sha256_file(path),'process_peak_working_set_MiB':evidence()['process_peak_working_set_MiB']}
        (args.output/'progress.json').write_text(json.dumps({'training':training},indent=2,allow_nan=False),encoding='utf-8')
        print('flow_fit_complete',role,'seconds',training[role]['seconds'],'peakMiB',training[role]['process_peak_working_set_MiB'],flush=True)
    del x,residual,arrays;gc.collect()
    models['nn_primary']=frozen_nn;models['flow_seed_mean']=StateEnsemble([models[f'flow_seed{s}'] for s in [20261013,20261029]],[.5,.5])
    for role,flow in [('channel_seed20261013','flow_seed20261013'),('channel_seed20261029','flow_seed20261029'),('transfer_fixed','flow_seed_mean')]:
        models[role]=ChannelStateEnsemble([models['nn_primary'],models[flow]],[[1,0],[0,1]])
    evaluator=PublicDays(panel,baseline.profile,baseline.feature_version);read_order=[];cache_hashes={}
    def evaluate(part,primary,seal=None):
        if part=='assessment' and (seal is None or sha256_file(args.output/'selection_sealed.json')!=seal):raise ValueError('Assessment before seal')
        read_order.append({'part':part,'prior_selection_seal_sha256':seal});days=[];predictions={r:[] for r in models};records=[]
        out=args.output/'predictions'/part;out.mkdir(parents=True)
        for path in sorted((cache/part).glob('*.npz')):
            if sha256_file(path)!=original['cache_sha256'][path.as_posix()]:raise ValueError('Public evaluator cache changed')
            day=evaluator.load(path.stem,path);days.append(day);cache_hashes[path.as_posix()]=sha256_file(path)
            source=nn_folder/'predictions'/part/(path.stem+'.npz');cache_hashes[source.as_posix()]=sha256_file(source)
            with np.load(source) as a:values={'nn_primary':a[nn['selection']['primary']],'old_lgb':a['lgb']}
            for role in ['flow_seed20261013','flow_seed20261029']:
                p=models[role].predict(day['features'],day['template'])[VALUES].to_numpy(float);p[:,1]/=day['target_link_lanes'];values[role]=p
            values=compose(values,primary)
            for role,p in values.items():predictions[role].append(p)
            target=out/(path.stem+'.npz');np.savez_compressed(target,**values);records.append({'part':part,'date':path.stem,'path':target.as_posix(),'sha256':sha256_file(target)})
        return {r:summarize(days,p,evaluator.factor) for r,p in predictions.items()},records
    cal,cal_files=evaluate('selection',None);gain=cal['transfer_fixed']['S_state']-cal['nn_primary']['S_state']
    minimum=min(a['S_state']-b['S_state'] for a,b in zip(cal['transfer_fixed']['details'],cal['nn_primary']['details']))
    primary='transfer_fixed' if gain>=.0005 and minimum>=-.001 else 'nn_primary'
    selection={'primary':primary,'fixed_transfer_cal_gain':gain,'minimum_regime_cal_gain':minimum,'cal_metrics':cal}
    seal=args.output/'selection_sealed.json';seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8');seal_sha=sha256_file(seal)
    print('flow_primary_SEALED',primary,'fixed_transfer_cal_gain',gain,flush=True)
    models['primary_export']=models[primary];exports={}
    for role,model in models.items():
        path=args.output/('State_'+role+'.pkl')
        if not path.exists():model.save(path)
        exports[role]={'path':path.as_posix(),'sha256':sha256_file(path)}
    assessment,ass_files=evaluate('assessment',primary,seal_sha)
    comparisons={r:{'pooled_State_gain':m['S_state']-assessment['nn_primary']['S_state'],
        'paired_day_diagnostic':paired_block_interval([d['daily_State'] for d in assessment['nn_primary']['days']],[d['daily_State'] for d in m['days']])} for r,m in assessment.items()}
    result={'registered':registered,'source_snapshot':snapshot,'dataset':dataset,'training':training,'selection':selection,'selection_seal_sha256':seal_sha,
        'assessment':assessment,'comparisons':comparisons,'prediction_files':cal_files+ass_files,'exports':exports,'phase_read_order':read_order,
        'cache_sha256':cache_hashes,'source_files_sha256':evaluator.hashes,'seconds':time.perf_counter()-start,'source':evidence(gpu_used=False),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Only flow channels newly fitted;NN/control reuse explicit. More labels and weighted MSE differ from NN. Public regression dates;no formal Physics/Total.'}
    text=json.dumps(result,indent=2,allow_nan=False);report.write_text(text,encoding='utf-8');(args.output/'experiment.json').write_text(text,encoding='utf-8')
    print('flow_transfer_COMPLETE',primary,'NN',assessment['nn_primary']['S_state'],'fixed',assessment['transfer_fixed']['S_state'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
