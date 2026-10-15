"""Rebuild fit labels and diagnostics without the transition helper."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.data import Release,eligible
from trafficflow.state import neighbors
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol


def independent_masks(y,valid,incoming,outgoing):
    previous=np.concatenate([np.full_like(y[:1],np.nan),y[:-1]])
    following=np.concatenate([y[1:],np.full_like(y[:1],np.nan)])
    second=np.concatenate([y[2:],np.full_like(y[:2],np.nan)])
    prev_valid=np.concatenate([np.zeros_like(valid[:1]),valid[:-1]])
    next_valid=np.concatenate([valid[1:],np.zeros_like(valid[:1])]);second_valid=np.concatenate([valid[2:],np.zeros_like(valid[:2])])
    pair=valid&prev_valid;delta=np.where(pair[...,None],y-previous,np.nan);f1=pair&next_valid;f2=f1&second_valid
    result={'pair_valid':pair,'delta':delta,'drop20':delta[...,0]<=-20,'recover20':delta[...,0]>=20,'flow150':np.abs(delta[...,1])>=150,
        'future1_eligible':f1,'future2_eligible':f2}
    for c,large,width,persist,ret_name,persist_name in [(0,20,5,10,'speed_reversal5','speed_persistent10'),(1,150,50,75,'flow_reversal50','flow_persistent75')]:
        selected=np.abs(delta[...,c])>=large
        result[ret_name]=selected&f1&(np.abs(following[...,c]-previous[...,c])<=width)
        result[persist_name]=selected&f2&(np.sign(delta[...,c])*(following[...,c]-previous[...,c])>=persist)&(np.sign(delta[...,c])*(second[...,c]-previous[...,c])>=persist)
    windows=np.lib.stride_tricks.sliding_window_view(np.pad(delta[...,0],((2,2),(0,0)),constant_values=np.nan),5,axis=0)
    availability=np.zeros_like(valid);support=np.zeros_like(valid)
    for j in range(y.shape[1]):
        ns=sorted(set(incoming[j]+outgoing[j]))
        if not ns:continue
        w=windows[:,ns,:];finite=np.isfinite(w);availability[:,j]=finite.any(axis=(1,2))
        largest=np.where(finite,w,-np.inf).max(axis=(1,2));smallest=np.where(finite,w,np.inf).min(axis=(1,2))
        support[:,j]=((delta[:,j,0]>0)&(largest>=10))|((delta[:,j,0]<0)&(smallest<=-10))
    result['neighbor_available']=availability;result['neighbor_same_sign10']=support
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite fit audit verification')
    source=json.loads(args.study.read_text());registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Independent_fit_transition_labels_masks_and_statistics','source_study_sha256':sha256_file(args.study),
        'scope':'All245raw fit teacher grids,all diagnostic arrays,all population/exposure count and SSE fields;no model fitting or validation',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public')
    for path,digest in source['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Raw/cache/network/exposure changed')
    checks=[];n_stats=n_arrays=0;maximum=0.
    for item in source['registered']['protocol']['sources']:
        views=json.loads(Path(item['views']).read_text());nn=json.loads(Path(item['NN_study']).read_text());old=json.loads(Path(views['registered']['protocol']['source_study']).read_text())
        panel=old['registered']['protocol']['panel'];net,topology=release.network(panel),release.topology(panel);observed=source['panels'][panel]
        sources={(r['date'],r['population'],r.get('view',-1)):r['statistics'] for r in observed['day_population_records']}
        archives={}
        for name in nn['training']:archives[name]=np.load(Path('artifacts/research')/Path(item['NN_study']).stem/(name+'_exposure.npz'))
        exposure_records={(name,r['date']):r['statistics'] for name,rows in observed['day_actual_exposure_records'].items() for r in rows}
        accumulators={};actual_accumulators={}
        def compare(stat,wanted,accumulator):
            nonlocal n_stats,maximum
            if set(stat)!=set(wanted):raise ValueError('Statistic fields differ')
            for k,v in stat.items():
                error=abs(v-wanted[k]);maximum=max(maximum,error)
                if not np.isclose(v,wanted[k],rtol=1e-12,atol=1e-6):raise ValueError('Count/SSE differs: '+k)
                accumulator[k]=accumulator.get(k,0.)+v
            n_stats+=1
        for row in views['rows']:
            raw=release.read(row['raw_file']);a=dict(np.load(row['original_cache_path']));b=np.load(row['path']);links=a['links'].tolist();times=a['times'].tolist()
            good=raw.loc[eligible(raw)];loc=good.timestamp.map({v:i for i,v in enumerate(times)}).to_numpy(int)*len(links)+good.link_id.map({v:i for i,v in enumerate(links)}).to_numpy(int)
            count=np.bincount(loc,minlength=len(times)*len(links));y=np.zeros((len(count),2))
            for c,v in enumerate([good.speed_kmh.to_numpy(float),good.flow_vph.to_numpy(float)/good.link_id.map(net.lanes).to_numpy(float)]):
                y[:,c]=np.divide(np.bincount(loc,weights=v,minlength=len(count)),count,out=np.zeros(len(count)),where=count>0)
            y=y.reshape(*a['mask'].shape,2);valid=count.reshape(a['mask'].shape)>0
            d=independent_masks(y,valid,neighbors(topology,links,'incoming'),neighbors(topology,links,'outgoing'))
            record=next(r for r in source['files'] if r['panel']==panel and r['date']==row['date'])
            if sha256_file(record['path'])!=record['sha256']:raise ValueError('Diagnostic arrays changed')
            with np.load(record['path']) as stored:
                if not np.array_equal(valid,stored['raw_valid']):raise ValueError('Raw eligibility differs')
                for key,value in d.items():
                    if not np.array_equal(value,stored[key],equal_nan=True):raise ValueError('Independent event/neighbor/eligibility differs: '+key)
                    n_arrays+=1
            def statistics(target,base=None,weights=None):
                w=target.astype(float) if weights is None else target*weights
                event=d['drop20']|d['recover20'];out={'targets':float(w.sum()),'pair_targets':float(w[d['pair_valid']].sum()),
                    'speed_event_future1':float(w[event&d['future1_eligible']].sum()),'speed_event_future2':float(w[event&d['future2_eligible']].sum()),
                    'speed_event_neighbor_available':float(w[event&d['neighbor_available']].sum()),'neighbor_same_sign10':float(w[event&d['neighbor_same_sign10']].sum())}
                classes=['drop20','recover20','flow150','speed_reversal5','speed_persistent10','flow_reversal50','flow_persistent75']
                out.update({k:float(w[d[k]].sum()) for k in classes})
                if weights is None:out['adjacent_supervised_pairs']=float(np.count_nonzero(target[1:]&target[:-1]&d['pair_valid'][1:]))
                if base is not None:
                    squared=(y-base)**2
                    for c,label in enumerate(['speed','flow']):
                        out['base_'+label+'_sse']=float((squared[...,c]*w).sum())
                        for k in classes[:5]:out[k+'_'+label+'_sse']=float((squared[...,c]*w)[d[k]].sum())
                return out
            rows=[('all_raw_fit',-1,valid,None),('original_mask',-1,a['mask'],a['x'][...,6:8]*np.array([25,600]))]
            for variant in ['point','queue']:
                masks=b['mask_'+variant];x=b['x_'+variant]
                rows.extend((variant+'_4views',j,masks[j],x[j,...,6:8]*np.array([25,600])) for j in range(4))
            for name,view,mask,base in rows:
                stat=statistics(mask,base);compare(stat,sources[(row['date'],name,view)],accumulators.setdefault(name,{}))
            for name,archive in archives.items():compare(statistics(valid,weights=archive[row['date']]),exposure_records[(name,row['date'])],actual_accumulators.setdefault(name,{}))
            b.close();checks.append({'panel':panel,'date':row['date'],'all_masks_counts_exposures_and_base_SSE_verified':True})
        def finish(stats):
            for k,denom in [('drop20','pair_targets'),('recover20','pair_targets'),('flow150','pair_targets'),('speed_reversal5','speed_event_future1'),('speed_persistent10','speed_event_future2'),('neighbor_same_sign10','speed_event_neighbor_available')]:stats[k+'_fraction']=stats[k]/stats[denom] if stats.get(denom,0)>0 else None
            for k in ['drop20','recover20','flow150','speed_reversal5','speed_persistent10']:stats[k+'_base_speed_sse_share']=stats.get(k+'_speed_sse',0)/stats['base_speed_sse'] if stats.get('base_speed_sse',0)>0 else None
            return stats
        for name,stats in accumulators.items():
            actual=finish(stats);wanted=observed['population_statistics'][name]
            if set(actual)!=set(wanted) or any((v is None)!=(wanted[k] is None) or (v is not None and not np.isclose(v,wanted[k],rtol=1e-12,atol=1e-6)) for k,v in actual.items()):raise ValueError('Pooled population/SSE share differs')
        for name,stats in actual_accumulators.items():
            actual=finish(stats);wanted=observed['actual_exposure_statistics'][name]
            if any((v is None)!=(wanted[k] is None) or (v is not None and not np.isclose(v,wanted[k],rtol=1e-12,atol=1e-6)) for k,v in actual.items()):raise ValueError('Actual exposure pooling differs')
        for archive in archives.values():archive.close()
    result={'registered':registered,'source_snapshot':snapshot,'n_raw_fit_days':len(checks),'n_diagnostic_arrays_verified':n_arrays,'n_day_population_or_exposure_records_verified':n_stats,
        'maximum_absolute_statistic_difference':maximum,'checks':checks,'seconds':time.perf_counter()-start,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'Full implementation/count verification only. Same raw numerical teacher definition;independent shift/window masks and rollups. Not model or latent-event validation.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('fit_transition_verified',len(checks),'days',n_stats,'records','maxstat',maximum,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
