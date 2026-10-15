"""Strong same-view LightGBM reference;cal-selected hybrid kept separate."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from trafficflow.data import Release,VALUES
from trafficflow.state import StateFeatures,StateModel
from trafficflow.state_ensemble import StateEnsemble
from trafficflow.bounded_state import BoundedState
from trafficflow.native_remask import native_remask
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.experiments import partition_date,day_template,target_truth
from trafficflow.research_eval import register_protocol,paired_block_interval
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from recurrent_state_experiment import summarize,transition_masks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--views',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Never overwrite remask tree study')
    views_path=args.views/'cache.json';views=json.loads(views_path.read_text());original_path=Path(views['registered']['protocol']['source_study'])
    original=json.loads(original_path.read_text());config=original['registered']['protocol'];cache=Path(config['cache'])
    nn_path=Path('reports/research/A97_I210_W_native_remask_ablation.json');nn=json.loads(nn_path.read_text());nn_folder=Path('artifacts/research')/nn_path.stem
    nn_replay_path=Path('reports/research/A100_I210_W_native_remask_training_independent_replay.json');nn_replay=json.loads(nn_replay_path.read_text())
    if config['panel']!='D7_I210_W' or nn['registered']['protocol']['views_report_sha256']!=sha256_file(views_path):raise ValueError('Matched native view pool required')
    primary_nn=nn_replay['exports']['primary_export']
    if sha256_file(primary_nn['path'])!=primary_nn['sha256']:raise ValueError('Frozen NN primary changed')
    baseline=StateModel.load(Path(config['baseline_file']));params=baseline.models[0].get_params()
    dates=pd.date_range('2031-01-23','2031-02-05').strftime('%Y-%m-%d').tolist()
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'Strong_LightGBM_on_matched_native_remask_views','source_study_sha256':sha256_file(original_path),
        'view_report_sha256':sha256_file(views_path),'native_NN_study_sha256':sha256_file(nn_path),
        'NN_primary_export':primary_nn,'panel':config['panel'],'seeds':[20261013,20261029],'variants':['point','queue'],
        'tree_parameters_from_original':params,'feature_version':baseline.feature_version,
        'data':'All original fit grid targets plus all4remask target sets pervariant,same float32 link/time teacher pool as NN',
        'sample_weight':'Half original,half four new views;uniform regime/date/target inside each group;total effective mass original target count',
        'budget':'Original400trees/63leaves and4CPUthreads;more rows and different loss/weighting than NN,not equal optimization budget',
        'primary':'Cal-only max oldLGB,previousNNprimary and each2seed tree mean blended with NN at tree weight0/.25/.5/.75/1',
        'assessment':'Original viewed assessment only after selection seal;all negative results retained',
        'future_dates':dates,'future_status':'Fixed before tree fit;public research calendar,other historical studies may have viewed dates',
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');panel=config['panel'];net,topology=release.network(panel),release.topology(panel)
    masked={partition_date(release,p):p for p in release.files(panel,'train')};raw_paths={p.name:p for p in release.files(panel,'train',False)};template=release.template(1,panel,'train')
    factor=None;read_order=[];cache_hashes={}
    def load(part,seal=None):
        nonlocal factor
        if part=='assessment' and (seal is None or sha256_file(args.output/'selection_sealed.json')!=seal):raise ValueError('Assessment before seal')
        read_order.append({'part':part,'prior_selection_seal_sha256':seal});result=[]
        for path in sorted((cache/part).glob('*.npz')):
            if sha256_file(path)!=original['cache_sha256'][path.as_posix()]:raise ValueError('Reference cache changed')
            day=dict(np.load(path));day['date']=path.stem;cache_hashes[path.as_posix()]=sha256_file(path)
            frame=release.read(masked[path.stem]);f=StateFeatures(frame,net,topology,baseline.profile,baseline.feature_version)
            if not np.array_equal(reconstruction_input(f),day['x']):raise ValueError('Original input differs')
            target=day_template(release,panel,'train',frame,template);raw=release.read(raw_paths[masked[path.stem].name])
            truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float);truth[:,1]/=target.link_id.map(net.lanes).to_numpy(float)
            if not np.array_equal(truth,day['station_truth']):raise ValueError('Raw evaluator differs')
            day['features'],day['template']=f,target;day['diagnostic_masks']=transition_masks(day,target,raw)
            with np.load(nn_folder/'predictions'/part/(path.stem+'.npz')) as a:
                day['nn_primary']=a[nn['selection']['primary']];day['old_lgb']=a['lgb']
            if factor is None:factor=(net.lanes*net.length_km).reindex(f.links).to_numpy(np.float32)
            result.append(day)
        return result
    cal=load('selection');training={};models={};datasets={};original_count=sum(r['original_target_grid_cells'] for r in views['rows'])
    group_counts={r:sum(row['regime']==r for row in views['rows']) for r in ['R1','R2','R3']}
    for variant in ['point','queue']:
        blocks=[];labels=[];bases=[];weights=[];keys=[];groups=[];cursor=0;prep_start=time.perf_counter()
        for day_index,row in enumerate(views['rows']):
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Matched views changed')
            frame=release.read(row['masked_file']);raw=release.read(row['raw_file'])
            if 'mask_regime' not in raw:raw['mask_regime']=row['regime']
            original_day=dict(np.load(row['original_cache_path']));arrays=dict(np.load(row['path']))
            for view in [-1,0,1,2,3]:
                if view<0:input_frame=frame;mask=original_day['mask'];teacher=original_day['y'];mass=.5
                else:
                    input_frame,_,_=native_remask(raw,panel,row['date'],views['registered']['protocol']['view_seeds'][view],row['regime'],row['public_origins'],variant=='queue')
                    mask=arrays['mask_'+variant][view];teacher=arrays['y'];mass=.5/4
                f=StateFeatures(input_frame,net,topology,baseline.profile,baseline.feature_version)
                if view>=0 and not np.array_equal(reconstruction_input(f),arrays['x_'+variant][view]):raise ValueError('Tree input differs from NN view')
                index=np.flatnonzero(mask.ravel());n=len(index)
                blocks.append(f.features.iloc[index]);labels.append(teacher.reshape(-1,2)[index]*np.array([25.,600.]))
                bases.append(f.base.reshape(-1,2)[index]);weights.append(np.full(n,mass/(3*group_counts[row['regime']]*n)*original_count))
                keys.append(np.c_[np.full(n,day_index),np.full(n,view),index].astype(np.int32))
                groups.append({'date':row['date'],'view':view,'start':cursor,'end':cursor+n,'regime':row['regime']});cursor+=n
            if (day_index+1)%20==0:print('remask_tree_prepare',variant,day_index+1,'dates',flush=True)
        x=pd.concat(blocks,ignore_index=True);y=np.concatenate(labels);base=np.concatenate(bases);weight=np.concatenate(weights);row_keys=np.concatenate(keys)
        del blocks,labels,bases,weights,keys
        if x.columns.tolist()!=baseline.models[0].feature_name_:raise ValueError('Feature columns differ from original strong tree')
        folder=args.output/variant;folder.mkdir();x_path=folder/'features.parquet';array_path=folder/'labels_weights_keys.npz'
        x.to_parquet(x_path,index=False,compression='zstd');np.savez_compressed(array_path,y=y,base=base,weight=weight,row_keys=row_keys)
        datasets[variant]={'rows':len(x),'features':len(x.columns),'features_path':x_path.as_posix(),'features_sha256':sha256_file(x_path),
            'arrays_path':array_path.as_posix(),'arrays_sha256':sha256_file(array_path),'effective_weight_sum':float(weight.sum()),
            'n_original_rows':int((row_keys[:,1]<0).sum()),'n_remasked_rows':int((row_keys[:,1]>=0).sum()),'groups':groups,'preparation_seconds':time.perf_counter()-prep_start}
        residual=(y-base)/np.array([25.,600.])
        for seed in [20261013,20261029]:
            name=f'{variant}_seed{seed}';tic=time.perf_counter();model=StateModel(baseline.profile,'lightgbm',seed,baseline.iterations,4,baseline.feature_version,baseline.num_leaves)
            for channel in range(2):
                kwargs=params.copy();kwargs['random_state']=seed;kwargs['n_jobs']=4
                tree=LGBMRegressor(**kwargs);tree.fit(x,residual[:,channel],sample_weight=weight);model.models.append(tree)
            bounded=BoundedState(model);path=args.output/('State_'+name+'.pkl');bounded.save(path);models[name]=bounded
            training[name]={'seconds':time.perf_counter()-tic,'model_path':path.as_posix(),'model_sha256':sha256_file(path),'trees_per_channel':model.iterations}
            print('remask_tree_fit_complete',name,'seconds',training[name]['seconds'],flush=True)
        members=[models[f'{variant}_seed{s}'] for s in [20261013,20261029]];models[variant+'_seed_mean']=StateEnsemble(members,[.5,.5])
        models[variant+'_seed_mean'].save(args.output/('State_'+variant+'_seed_mean.pkl'))
        del x,y,base,weight,row_keys,residual
        (args.output/'progress.json').write_text(json.dumps({'datasets':datasets,'training':training},indent=2,allow_nan=False),encoding='utf-8')
    def evaluate(days):
        values={'old_lgb':[d['old_lgb'] for d in days],'nn_primary':[d['nn_primary'] for d in days]}
        for role,model in models.items():
            output=[]
            for day in days:
                p=model.predict(day['features'],day['template'])[VALUES].to_numpy(float);p[:,1]/=day['target_link_lanes'];output.append(p)
            values[role]=output
        for variant in ['point','queue']:
            for alpha in [.25,.5,.75]:
                values[f'{variant}_hybrid_{alpha}']=[alpha*t+(1-alpha)*n for t,n in zip(values[variant+'_seed_mean'],values['nn_primary'])]
        return values,{n:summarize(days,p,factor) for n,p in values.items()}
    cal_values,cal_metrics=evaluate(cal)
    eligible=['old_lgb','nn_primary','point_seed_mean','queue_seed_mean']+[f'{v}_hybrid_{a}' for v in ['point','queue'] for a in [.25,.5,.75]]
    primary=max(eligible,key=lambda n:cal_metrics[n]['S_state']);selection={'primary':primary,'cal_metrics':cal_metrics}
    seal=args.output/'selection_sealed.json';seal.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8');seal_sha=sha256_file(seal)
    print('remask_tree_PRIMARY_SEALED',primary,flush=True)
    assessment=load('assessment',seal_sha);ass_values,ass_metrics=evaluate(assessment)
    for phase,days,values in [('selection',cal,cal_values),('assessment',assessment,ass_values)]:
        folder=args.output/'predictions'/phase;folder.mkdir(parents=True)
        for i,day in enumerate(days):np.savez_compressed(folder/(day['date']+'.npz'),**{n:p[i] for n,p in values.items()})
    if primary=='nn_primary':selected=StateModel.load(Path(primary_nn['path']))
    elif primary=='old_lgb':
        selected=BoundedState(baseline)
    elif primary.endswith('seed_mean'):selected=models[primary]
    else:
        variant=primary.split('_hybrid_')[0];alpha=float(primary.split('_hybrid_')[1]);selected=StateEnsemble([models[variant+'_seed_mean'],StateModel.load(Path(primary_nn['path']))],[alpha,1-alpha])
    selected.save(args.output/'State_primary.pkl')
    comparisons={n:{'pooled_State_gain_vs_NN':m['S_state']-ass_metrics['nn_primary']['S_state'],
        'paired_day_diagnostic':paired_block_interval([r['daily_State'] for r in ass_metrics['nn_primary']['days']],[r['daily_State'] for r in m['days']])} for n,m in ass_metrics.items()}
    result={'registered':registered,'source_snapshot':snapshot,'datasets':datasets,'training':training,'selection':selection,'selection_seal_sha256':seal_sha,
        'assessment':ass_metrics,'comparisons':comparisons,'phase_read_order':read_order,'cache_sha256':cache_hashes,
        'primary_export':{'path':(args.output/'State_primary.pkl').as_posix(),'sha256':sha256_file(args.output/'State_primary.pkl')},
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=False),'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Same fit view pools as NN,but squared residual loss,tabular features and full weighted row pass differ from neural training. '
        'Old assessment viewed. Hybrid arrays use frozen saved NN predictions;no GPU work in this training process. No formal score inferred.'}
    text=json.dumps(result,indent=2,allow_nan=False);report.write_text(text,encoding='utf-8');(args.output/'experiment.json').write_text(text,encoding='utf-8')
    print('remask_tree_COMPLETE',primary,'assessment',ass_metrics[primary]['S_state'],'NN',ass_metrics['nn_primary']['S_state'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
