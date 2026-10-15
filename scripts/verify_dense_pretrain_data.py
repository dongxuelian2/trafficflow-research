"""Independent all-day indexed-history and teacher reconstruction for C87."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.dense_queue_data import dense_window,dense_origins
from trafficflow.queue_graph_geometry import travel_time_neighborhoods
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol


def independent_day(raw_path,masked_path,links,net):
    raw=pd.read_parquet(raw_path);masked=pd.read_parquet(masked_path);times=pd.date_range(pd.Timestamp(str(raw.timestamp.iloc[0])[:10],tz='UTC'),periods=288,freq='5min');tokens=times.strftime('%Y-%m-%dT%H:%M:%SZ');lanes=net.lanes.reindex(links).to_numpy(float);cut=net.v_cut.reindex(links).to_numpy(float)
    def channels(frame):
        frame=frame.copy();frame['timestamp']=frame.timestamp.astype(str);frame['link_id']=frame.link_id.astype(str);good=pd.to_numeric(frame.pct_observed,errors='coerce').ge(75)&np.isfinite(pd.to_numeric(frame.speed_kmh,errors='coerce'))&np.isfinite(pd.to_numeric(frame.flow_vph,errors='coerce'))
        grids={}
        for name in ['speed_kmh','flow_vph','occupancy']:
            frame['_value']=pd.to_numeric(frame[name],errors='coerce').where(good);grids[name]=frame.groupby(['timestamp','link_id'])['_value'].mean().unstack('link_id').reindex(index=tokens,columns=links).to_numpy(float)
        v,q,o=[grids[k] for k in ['speed_kmh','flow_vph','occupancy']];result=np.empty((288,len(links),6),np.float32);result[:,:,0]=np.nan_to_num(v/25.);result[:,:,1]=np.nan_to_num(q/lanes/600.);result[:,:,2]=np.isfinite(v);result[:,:,3]=np.isfinite(q);result[:,:,4]=np.nan_to_num(o/.25);result[:,:,5]=np.isfinite(o)
        return result,v
    r,v=channels(raw);m,_=channels(masked);phase=2*np.pi*(np.array(times.hour)*60+np.array(times.minute))/1440.;time=np.column_stack([np.sin(phase),np.cos(phase)]).astype(np.float32)
    return {'raw_channels':r,'masked_channels':m,'time_channels':time,'truth_speed':r[:,:,0].copy(),'truth_queue':(v<=cut).astype(np.float32),'available':np.isfinite(v),'timestamps':np.array(tokens),'links':np.array(links)}


def independent_window(pack,at,ongoing=None):
    n=len(pack['links']);x=np.empty((13,n,9),np.float32);x[:12,:,:6]=pack['raw_channels'][at-12:at];x[12,:,:6]=pack['masked_channels'][at];x[:,:,6:8]=pack['time_channels'][at-12:at+1,None,:];condition=bool((pack['truth_queue'][at-12:at].sum(axis=0)>=2).any()) if ongoing is None else bool(ongoing);x[:,:,8]=condition
    return x,pack['truth_queue'][at+1:at+7],pack['truth_speed'][at+1:at+7],pack['available'][at+1:at+7]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-folder',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite dense neural data audit')
    seal=args.study_folder/'all_datasets_sealed.json';datasets=json.loads(seal.read_text());registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_dense_pretrain_input_teacher_audit','source_dataset_seal_sha256':sha256_file(seal),
        'scope':'Independent raw eligibility/groupby/scales/time arrays for ALL fit days,all legal origin histories,all fit/checkpoint event targets,actual future/rawT poison;old neural cache x was not covered by prior tree-only audits. No network training or hidden labels.', 'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public');old_proof=json.loads(Path('reports/research/C72b_rolling_onset_threshold_verification.json').read_text());source_hashes={};cases=[];totals={'fit_day_packs':0,'dense_origins':0,'event_cache_windows':0,'actual_future_rawT_poison_checks':0,'event_cache_records':0}
    for data in datasets['datasets']:
        panel=data['panel'];net,topology=release.network(panel),release.topology(panel);raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};masked_paths={partition_date(release,p):p for p in release.files(panel,'train')};first=np.load(data['day_records'][0]['path']);links=first['links'].tolist();origins=[]
        for p in (release.root/'corridors'/panel/'network').glob('*.csv'):
            digest=sha256_file(p);source_hashes[p.as_posix()]=digest
            if old_proof['source_files_sha256'].get(p.as_posix())!=digest:raise ValueError('Frozen network metadata changed')
        for path,digest in data['source_files_sha256'].items():
            if sha256_file(path)!=digest:raise ValueError('Source raw/masked data changed')
            source_hashes[path]=digest
        for day_index,r in enumerate(data['day_records']):
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Fit day pack changed')
            pack=independent_day(Path(r['raw']['path']),Path(r['masked']['path']),links,net);saved=dict(np.load(r['path']))
            if pack.keys()!=saved.keys() or any(not np.array_equal(pack[k],saved[k]) for k in pack):raise ValueError('Independent daily normalized channels/teacher differs')
            at=np.arange(12,282);history=at[:,None]+np.arange(-12,0);valid=at[pack['available'][history].mean(axis=(1,2))>=.7]
            if not np.array_equal(valid,dense_origins(saved)):raise ValueError('History-only origin pool differs')
            for slot in valid:
                expected=independent_window(pack,int(slot));actual=dense_window(saved,int(slot))
                if any(not np.array_equal(v,actual[k]) for v,k in zip(expected,['x','labels','truth_speed','available'])):raise ValueError('Independent dense legal window differs')
                origins.append((day_index,int(slot)));totals['dense_origins']+=1
            for slot in [int(valid[0]),int(valid[len(valid)//2]),int(valid[-1])]:
                before=dense_window(saved,slot)['x'];bad={k:v.copy() for k,v in saved.items()};bad['raw_channels'][slot:]=777;bad['masked_channels'][slot+1:]=-777;bad['truth_queue'][slot:]=1-bad['truth_queue'][slot:];bad['truth_speed'][slot+1:]=999;bad['available'][slot+1:]=~bad['available'][slot+1:]
                if not np.array_equal(before,dense_window(bad,slot)['x']):raise ValueError('Actual future/rawT poison changed input')
                totals['actual_future_rawT_poison_checks']+=1
            totals['fit_day_packs']+=1
        if sha256_file(data['origin_record']['path'])!=data['origin_record']['sha256']:raise ValueError('Origin index changed')
        with np.load(data['origin_record']['path']) as saved:
            if not np.array_equal(np.array(origins,np.int32),saved['indices']):raise ValueError('Complete dense origin index differs')
        @lru_cache(maxsize=8)
        def day(date):return independent_day(raw_paths[date],masked_paths[date],links,net)
        for r in data['event_records']:
            if sha256_file(r['path'])!=r['sha256'] or sha256_file(r['source_record']['path'])!=r['source_record']['sha256']:raise ValueError('Event cache/source changed')
            a=dict(np.load(r['path']));legacy=dict(np.load(r['source_record']['path']))
            for i,text in enumerate(a['origins']):
                origin=pd.Timestamp(str(text));date=origin.date().isoformat();pack=day(date);at=int((origin-pd.Timestamp(date,tz='UTC'))/pd.Timedelta(minutes=5));expected=independent_window(pack,at,bool(a['ongoing'][i]))
                if any(not np.array_equal(v,a[k][i]) for v,k in zip(expected,['x','labels','truth_speed','available'])):raise ValueError('Independent fit/checkpoint history or teacher differs')
                if 'x' in legacy and not np.array_equal(expected[0],legacy['x'][i]):raise ValueError('Previously unused old neural input is inconsistent')
                totals['event_cache_windows']+=1
            totals['event_cache_records']+=1
        if sha256_file(data['geometry']['path'])!=data['geometry']['sha256']:raise ValueError('Geometry cache changed')
        with np.load(data['geometry']['path']) as g:
            incoming,outgoing=travel_time_neighborhoods(links,net,topology,30.);lane=net.lanes.reindex(links).to_numpy(float);static=np.stack([net.free_speed_kmh.reindex(links),net.capacity_vph.reindex(links)/lane,net.k_jam.reindex(links)/lane,net.v_cut.reindex(links)]).astype(np.float32)
            if not np.array_equal(incoming,g['incoming']) or not np.array_equal(outgoing,g['outgoing']) or not np.array_equal(static,g['geometry']):raise ValueError('Existing geometry recipe/fields differ')
        cases.append({'name':data['name'],'all_fit_channels_origins_event_teachers_and_poison_verified':True,'dense_origins':len(origins)});print('dense_pretrain_data_verified_case',data['name'],flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'totals':totals,'source_files_sha256':source_hashes,'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,
        'scope':'Fresh full fit input/teacher/origin audit,including old neural x and selected checkpoint events. Static graph recipe reused/rederived,not a new geometry theory or independent neural training. No model performance claim.'};raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8');print('dense_pretrain_data_verification_COMPLETE',totals,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
