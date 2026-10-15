"""Independent raw joint teachers and masked State context audit."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.joint_queue_data import joint_window,state_window
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol
from verify_dense_pretrain_data import independent_day,independent_window


def reference_day(raw,masked,links,net):
    p=independent_day(raw,masked,links,net);m=pd.read_parquet(masked);regimes=m.groupby(['timestamp','link_id']).mask_regime.first().unstack('link_id').reindex(index=p['timestamps'],columns=links).to_numpy(str);p['mask_regime']=regimes;p['v_cut_scaled']=np.asarray(net.v_cut.reindex(links),np.float32)/25.;return p


def reference_state(p,at):
    n=len(p['links']);x=np.zeros((13,n,9),np.float32);x[:,:,:6]=p['masked_channels'][at-12:at+1];x[:,:,6]=p['time_channels'][at-12:at+1,0,None];x[:,:,7]=p['time_channels'][at-12:at+1,1,None]
    congested=(x[:12,:,0]<=p['v_cut_scaled'][None])&(x[:12,:,2]>.5);x[:,:,8]=bool((np.count_nonzero(congested,axis=0)>=2).any());truth=np.stack([p['raw_channels'][at,:,0],p['raw_channels'][at,:,1]]);domain=np.stack([p['available'][at]&~(x[-1,:,2]>.5),p['available'][at]&~(x[-1,:,3]>.5)])
    return {'state_x':x,'truth_current':truth,'current_domain':domain,'current_available':p['available'][at].copy(),'current_regime':p['mask_regime'][at].copy()}


def reference_joint(p,at,ongoing):
    x,y,v,mask=independent_window(p,at,ongoing);return {'x':x,'labels':y,'truth_speed':v,'available':mask,'truth_flow':p['raw_channels'][at+1:at+7,:,1].copy(),'ongoing':ongoing,**reference_state(p,at)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-folder',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite joint data verification')
    seal=args.study_folder/'all_datasets_sealed.json';datasets=json.loads(seal.read_text());registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_joint_flow_masked_State_inputs_and_teachers','dataset_seal_sha256':sha256_file(seal),'scope':'Independent public raw grouped arrays,allfit maskedState contexts,oldQueueinputs/newflow/currentteachers,actual rawpast/futurepoison;no model performance claim. Reuse C88 rawchannel reference implementation explicitly.'})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');tic=time.perf_counter();release=Release('data/kaggle_public');source_files={};totals={'fit_day_packs':0,'fit_masked_state_contexts':0,'fit_legal_Queue_origins':0,'fit_CP_event_windows':0,'actual_poison_checks':0};cases=[]
    for data in datasets['datasets']:
        panel=data['panel'];net=release.network(panel);links=np.load(data['day_records'][0]['path'])['links'].tolist();raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)};masked_paths={partition_date(release,p):p for p in release.files(panel,'train')}
        for path,d in data['source_files_sha256'].items():
            if sha256_file(path)!=d:raise ValueError('Public source changed')
            source_files[path]=d
        for path in (release.root/'corridors'/panel/'network').glob('*.csv'):source_files[path.as_posix()]=sha256_file(path)
        @lru_cache(maxsize=8)
        def day(date):return reference_day(raw_paths[date],masked_paths[date],links,net)
        origin_rows=[]
        for di,r in enumerate(data['day_records']):
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Joint day archive changed')
            expected=day(r['date']);saved=dict(np.load(r['path']))
            if expected.keys()!=saved.keys() or any(not np.array_equal(expected[k],saved[k]) for k in expected):raise ValueError('Joint independent day differs')
            for at in range(12,288):
                s=reference_state(expected,at);actual=state_window(saved,at)
                if any(not np.array_equal(s[k],actual[k]) for k in s):raise ValueError('Masked current State context or teacher differs')
                totals['fit_masked_state_contexts']+=1
                if at<282 and expected['available'][at-12:at].mean()>=.7:
                    inferred=bool((expected['truth_queue'][at-12:at].sum(axis=0)>=2).any());j=reference_joint(expected,at,inferred);actual=joint_window(saved,at)
                    if any(not np.array_equal(j[k],actual[k]) for k in j):raise ValueError('Dense joint teacher differs')
                    origin_rows.append((di,at));totals['fit_legal_Queue_origins']+=1
            for at in [12,150,281]:
                clean=joint_window(saved,at);bad={k:v.copy() for k,v in saved.items()};bad['raw_channels'][:]=999;bad['truth_queue'][:]=1;bad['available'][:]=False;bad['masked_channels'][at+1:]=999
                if not np.array_equal(state_window(bad,at)['state_x'],clean['state_x']):raise ValueError('Rawpast/label/future contaminated State inputs')
                bad={k:v.copy() for k,v in saved.items()};bad['raw_channels'][at:]=999;bad['truth_queue'][at:]=1;bad['masked_channels'][at+1:]=999;bad['available'][at+1:]=False
                if not np.array_equal(joint_window(bad,at)['x'],clean['x']):raise ValueError('Future/rawT contaminated Queue inputs')
                totals['actual_poison_checks']+=2
            totals['fit_day_packs']+=1
        if sha256_file(data['origin_record']['path'])!=data['origin_record']['sha256'] or not np.array_equal(np.array(origin_rows,np.int32),np.load(data['origin_record']['path'])['indices']):raise ValueError('Complete history-only Queue pool differs')
        for r in data['event_records']:
            if sha256_file(r['path'])!=r['sha256']:raise ValueError('Joint fit/CP event changed')
            a=dict(np.load(r['path']))
            for i,text in enumerate(a['origins']):
                t=pd.Timestamp(str(text));at=int((t-pd.Timestamp(t.date().isoformat(),tz='UTC'))/pd.Timedelta(minutes=5));expected=reference_joint(day(t.date().isoformat()),at,bool(a['ongoing'][i]))
                if any(not np.array_equal(v,a[k][i]) for k,v in expected.items()):raise ValueError('Freshfit/CP event input or newteacher differs')
                totals['fit_CP_event_windows']+=1
        cases.append({'name':data['name'],'verified':True});print('joint_data_verified',data['name'],flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'totals':totals,'source_files_sha256':source_files,'seconds':time.perf_counter()-tic,'formal_State':None,'formal_Queue':None,'formal_Total':None,'scope':'Fullfit data/CP inputs and teacher audit;not neural training,prediction,modelselection,official score or latent truth.'};raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'verification.json').write_text(raw,encoding='utf-8');print('joint_data_verification_COMPLETE',totals,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
