"""Fit-only transient supervision,base-input errors,and contextual label diagnostics."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release,eligible
from trafficflow.state import neighbors
from trafficflow.transition_diagnostics import transition_diagnostics
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol


def aggregate(records):
    result={}
    for row in records:
        key=row['population'];out=result.setdefault(key,{})
        for name,value in row['statistics'].items():out[name]=out.get(name,0.)+value
    for out in result.values():
        for key,denominator in [('drop20','pair_targets'),('recover20','pair_targets'),('flow150','pair_targets'),
            ('speed_reversal5','speed_event_future1'),('speed_persistent10','speed_event_future2'),
            ('neighbor_same_sign10','speed_event_neighbor_available')]:
            out[key+'_fraction']=out[key]/out[denominator] if out.get(denominator,0)>0 else None
        for key in ['drop20','recover20','flow150','speed_reversal5','speed_persistent10']:
            out[key+'_base_speed_sse_share']=out.get(key+'_speed_sse',0)/out['base_speed_sse'] if out.get('base_speed_sse',0)>0 else None
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite fit diagnostic')
    sources=[('artifacts/research/A96_I210_W_native_remask_fit_views/cache.json','reports/research/A97_I210_W_native_remask_ablation.json'),
        ('artifacts/research/A102_I5_N_native_remask_fit_views/cache.json','reports/research/A104_I5_N_native_remask_ablation.json')]
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Fit_only_transient_population_and_input_base_audit','sources':[{'views':v,'views_sha256':sha256_file(v),'NN_study':n,'NN_study_sha256':sha256_file(n)} for v,n in sources],
        'data_scope':'Only registered fit raw/cache/exposure files;no cal/assessment/forward teacher or prediction arrays',
        'thresholds':{'speed_change_kmh':20,'speed_return_to_previous_kmh':5,'speed_persistence_2future_steps_kmh':10,
            'flow_change_per_lane_vph':150,'flow_return_per_lane_vph':50,'flow_persistence_per_lane_vph':75,'neighbor_change_kmh':10,'neighbor_window_steps':2},
        'semantics':'Returns/persistence/neighbors are raw-fit observational diagnostics,not latent noise or physical-event truth;no midnight wrapping',
        'base_errors':'Cached input base channels6/7 against fit raw grid means;not neural errors or official State',
        'exposure':'Actual new-NN per-cell counts,not inferred adjacent-pair joint exposure;four remask views not independent events',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public')
    panels={};outputs=[];maximum_teacher=0.;source_hashes={}
    for view_path,nn_path in sources:
        views=json.loads(Path(view_path).read_text());nn=json.loads(Path(nn_path).read_text());original=json.loads(Path(views['registered']['protocol']['source_study']).read_text())
        panel=original['registered']['protocol']['panel'];net,topology=release.network(panel),release.topology(panel);records=[];exposures={};archives={}
        for path in sorted((release.root/'corridors'/panel/'network').glob('*.csv')):source_hashes[path.as_posix()]=sha256_file(path)
        out_folder=folder/panel;out_folder.mkdir()
        for name,row in nn['training'].items():
            path=Path('artifacts/research')/Path(nn_path).stem/(name+'_exposure.npz')
            if sha256_file(path)!=row['exposure_sha256']:raise ValueError('Actual label exposures changed')
            source_hashes[path.as_posix()]=sha256_file(path);archives[name]=np.load(path);exposures[name]=[]
        for i,row in enumerate(views['rows']):
            if row['date'] not in original['prepared']['dates']['fit']:raise ValueError('Non-fit date requested')
            for key,digest in [('raw_file','raw_file_sha256'),('path','sha256'),('original_cache_path','original_cache_sha256')]:
                if sha256_file(row[key])!=row[digest]:raise ValueError('Fit source changed')
                source_hashes[row[key]]=row[digest]
            raw=release.read(row['raw_file']);orig=dict(np.load(row['original_cache_path']));aug=np.load(row['path']);links=orig['links'].tolist();times=orig['times'].tolist()
            index=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id']);good=raw.loc[eligible(raw)];locations=index.get_indexer(pd.MultiIndex.from_frame(good[['timestamp','link_id']]))
            number=np.bincount(locations,minlength=len(index));teacher=np.zeros((len(index),2),float)
            for c,values in enumerate([good.speed_kmh.to_numpy(float),good.flow_vph.to_numpy(float)/good.link_id.map(net.lanes).to_numpy(float)]):
                teacher[:,c]=np.divide(np.bincount(locations,weights=values,minlength=len(index)),number,out=np.zeros(len(index)),where=number>0)
            shape=orig['mask'].shape;teacher=teacher.reshape(*shape,2);valid=number.reshape(shape)>0
            error=float(np.abs(teacher/np.array([25,600])-aug['y']).max());maximum_teacher=max(maximum_teacher,error)
            if error>1e-6:raise ValueError('Independent raw teacher differs')
            d=transition_diagnostics(teacher,valid,neighbors(topology,links,'incoming'),neighbors(topology,links,'outgoing'))
            def statistics(target,base=None,weight=None):
                w=target.astype(float) if weight is None else np.asarray(weight,float)*target
                speed_event=d['drop20']|d['recover20'];stats={'targets':float(w.sum()),'pair_targets':float(w[d['pair_valid']].sum()),
                    'speed_event_future1':float(w[speed_event&d['future1_eligible']].sum()),'speed_event_future2':float(w[speed_event&d['future2_eligible']].sum()),
                    'speed_event_neighbor_available':float(w[speed_event&d['neighbor_available']].sum()),
                    'neighbor_same_sign10':float(w[speed_event&d['neighbor_same_sign10']].sum())}
                if weight is None:stats['adjacent_supervised_pairs']=float((target[1:]&target[:-1]&d['pair_valid'][1:]).sum())
                for key in ['drop20','recover20','flow150','speed_reversal5','speed_persistent10','flow_reversal50','flow_persistent75']:
                    stats[key]=float(w[d[key]].sum())
                if base is not None:
                    squared=(base-teacher)**2;stats['base_speed_sse']=float((squared[...,0]*w).sum());stats['base_flow_sse']=float((squared[...,1]*w).sum())
                    for key in ['drop20','recover20','flow150','speed_reversal5','speed_persistent10']:
                        stats[key+'_speed_sse']=float((squared[...,0]*w*d[key]).sum());stats[key+'_flow_sse']=float((squared[...,1]*w*d[key]).sum())
                return stats
            records.append({'date':row['date'],'population':'all_raw_fit','statistics':statistics(valid)})
            records.append({'date':row['date'],'population':'original_mask','statistics':statistics(orig['mask'],orig['x'][...,6:8]*np.array([25,600]))})
            for variant in ['point','queue']:
                masks=aug['mask_'+variant];x=aug['x_'+variant]
                for view in range(4):records.append({'date':row['date'],'population':variant+'_4views','view':view,'statistics':statistics(masks[view],x[view,...,6:8]*np.array([25,600]))})
            for name,archive in archives.items():
                actual=archive[row['date']];exposures[name].append({'date':row['date'],'population':name,'statistics':statistics(valid,weight=actual)})
            output=out_folder/(row['date']+'.npz');np.savez_compressed(output,raw_valid=valid,**d)
            outputs.append({'panel':panel,'date':row['date'],'path':output.as_posix(),'sha256':sha256_file(output)})
            aug.close()
            if (i+1)%25==0:print('fit_transition',panel,i+1,'dates',flush=True)
        for archive in archives.values():archive.close()
        panels[panel]={'n_fit_days':len(views['rows']),'population_statistics':aggregate(records),'actual_exposure_statistics':{k:aggregate(v)[k] for k,v in exposures.items()},
            'day_population_records':records,'day_actual_exposure_records':exposures}
    result={'registered':registered,'source_snapshot':snapshot,'panels':panels,'files':outputs,'source_files_sha256':source_hashes,
        'maximum_raw_teacher_difference_normalized_units':maximum_teacher,'seconds':time.perf_counter()-start,'source':evidence(gpu_used=False),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'limits':'Fit diagnostics only. No model validation,latent noise identification,accident truth or official physics score. Exposure counts do not identify joint patch-pair supervision.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'audit.json').write_text(text,encoding='utf-8')
    print('fit_transition_COMPLETE',len(outputs),'days','teacher_max',maximum_teacher,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
