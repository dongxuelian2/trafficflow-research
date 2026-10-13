"""Small artificial integration fixture, NEVER official benchmark evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import STATE_KEY,QUEUE_KEY,OD_KEY

def make_fixture(root:Path,days=42,seed=2026,panels=None):
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Fixture output must be an empty directory')
    root.mkdir(parents=True,exist_ok=True)
    (root/'FIXTURE_ONLY.json').write_text(json.dumps({'kind':'artificial_test_fixture','seed':seed,'official':False,'days':days}))
    (root/'config').mkdir()
    panels=panels or ['D7_I10_E','D12_I5_N']
    manifest={'panels':[{'corridor_id':p,'family_id':p.rsplit('_',1)[0]} for p in panels]}
    (root/'config'/'corridors.json').write_text(json.dumps(manifest))
    master=[];rng=np.random.default_rng(seed)
    for panel_no,panel in enumerate(panels):
        links=[f'{panel}_L{i:02d}' for i in range(8)]
        stations=[f'{panel}_S{i:02d}' for i in range(8)]
        folder=root/'corridors'/panel/'network';folder.mkdir(parents=True)
        lanes=3+panel_no;vf=105.-panel_no*5;cap=2100.*lanes;jam=100.*lanes
        net=pd.DataFrame({'link_id':links,'length_km':1.1+panel_no*.2,'lanes':lanes,
                          'free_speed_kmh':vf,'capacity_vph':cap,'milepost':np.arange(8)})
        net.to_csv(folder/'links.csv',index=False)
        fd=net.copy();fd['station_id']=stations;fd['k_jam']=jam;fd['critical_density']=cap/vf;fd['v_cut']=.6*vf
        fd.to_csv(folder/'fd_parameters.csv',index=False)
        topo=pd.DataFrame({'link_id':links,'incoming_link_ids':['']+links[:-1],
                           'outgoing_link_ids':links[1:]+[''],'on_ramp_link_ids':'','off_ramp_link_ids':''})
        topo.to_csv(folder/'lwr_mainline_topology.csv',index=False)
        pd.DataFrame(columns=['ramp_link_id','nearest_mainline_link_id']).to_csv(folder/'ramp_attachment_map.csv',index=False)
        paths=[];inc=[]
        for a in range(8):
            for b in range(a,8):
                pid=f'{panel}_P{a}_{b}'
                paths.append({'path_id':pid,'origin_zone':f'O{a}','destination_zone':f'D{b}'})
                inc.extend({'path_id':pid,'link_id':links[j]} for j in range(a,b+1))
        pd.DataFrame(paths).to_csv(folder/'path_set.csv',index=False)
        pd.DataFrame(inc).to_csv(folder/'path_link_incidence.csv',index=False)
        for split,start,n_days in [('train','2030-06-01',days),('validation','2031-03-01',3),('private','2031-04-01',3)]:
            target_parts=[];window_parts=[];history_parts=[];queue_parts=[]
            for day in range(n_days):
                date=pd.Timestamp(start,tz='UTC')+pd.Timedelta(days=day)
                times=pd.date_range(date,periods=288,freq='5min')
                tt=np.arange(288)*5.
                # Artificial daily congestion with a backward-moving disturbance and variable amplitude.
                center=470.+rng.normal(0,12)+panel_no*15
                centers=center+(7-np.arange(8))*5.
                dip=np.exp(-.5*((tt[:,None]-centers[None,:])/48.)**2)
                incident=(day%7==3)
                if incident:
                    dip=np.maximum(dip,np.exp(-.5*((tt[:,None]-(960.+np.arange(8)*8))/35.)**2)*.8)
                speed=vf-(vf*.68+rng.normal(0,1))*dip+rng.normal(0,1.4,(288,8))
                speed=np.clip(speed,15,vf*1.04)
                demand=(.35+.3*np.exp(-.5*((tt-450)/80)**2)+.2*np.exp(-.5*((tt-990)/80)**2))[:,None]
                flow=cap*np.broadcast_to(demand,(288,8))*(1-.25*dip)+rng.normal(0,70,(288,8))
                flow*=1+.04*np.sin(np.arange(8))[None,:]
                pct=rng.choice([50,75,100],(288,8),p=[.1,.1,.8])
                unavailable=rng.random((288,8))<.025
                speed[unavailable]=np.nan;flow[unavailable]=np.nan
                time_text=times.strftime('%Y-%m-%dT%H:%M:%SZ')
                frame=pd.DataFrame({'panel':panel,'date':date.strftime('%Y-%m-%d'),
                         'timestamp':np.repeat(time_text,8),'station_id':np.tile(stations,288),
                         'link_id':np.tile(links,288),'milepost':np.tile(np.arange(8),288),
                         'speed_kmh':speed.ravel(),'flow_vph':flow.ravel(),'pct_observed':pct.ravel()})
                frame['is_score_eligible']=(frame.pct_observed>=75)&frame.speed_kmh.notna()&frame.flow_vph.notna()
                regime=['R1','R2','R3'][day%3];frame['mask_regime']=regime
                visible_queue=(speed<=.6*vf)&(pct>=75)&np.isfinite(flow)
                first_queue=int(np.flatnonzero(visible_queue.any(axis=1))[0])
                onset_at=max(12,first_queue-2)
                ongoing_at=min(280,first_queue+7)
                window_origins=[onset_at,ongoing_at]
                protected_times=set()
                for at in window_origins:
                    protected_times.update(time_text[at-12:at+19])
                removed=(rng.random(len(frame))<{'R1':.2,'R2':.3,'R3':.5}[regime])&frame.is_score_eligible
                # Task 2 history is republished and its horizon/buffer are excluded from Task 1 targets.
                removed &= ~frame.timestamp.isin(protected_times)
                masked=frame.copy();masked.loc[removed,['speed_kmh','flow_vph']]=np.nan
                filename='synthetic_mainline_'+date.strftime('%Y_%m_%d')+'.parquet'
                md=root/'corridors'/panel/split/'mainline_states_masked'/f'mask_regime={regime}';md.mkdir(parents=True,exist_ok=True)
                if split=='train':
                    rawdir=root/'corridors'/panel/split/'mainline_states'/f'year_month={date.strftime("%Y-%m")}';rawdir.mkdir(parents=True,exist_ok=True)
                    frame.to_parquet(rawdir/filename,index=False)
                target=frame.loc[removed,STATE_KEY].copy();target[['speed_kmh','flow_vph']]=0.
                target_parts.append(target)
                # Two public histories; no queue labels are written anywhere in the fixture release.
                for i,at in enumerate(window_origins):
                    wid=f'{panel}_{split}_{day}_{i}'
                    condition='queue_onset' if i==0 else 'queue_ongoing'
                    origin=time_text[at]
                    window_parts.append({'window_id':wid,'panel':panel,'family_id':panel.rsplit('_',1)[0],
                        'split':split,'condition':condition,'forecast_origin':origin})
                    h=frame[frame.timestamp.isin(time_text[at-12:at])].copy();h['window_id']=wid
                    history_parts.append(h)
                    qtarget=pd.MultiIndex.from_product([[wid],time_text[at+1:at+7],links],names=QUEUE_KEY).to_frame(index=False)
                    qtarget['queue_pred']=0;queue_parts.append(qtarget)
                    # An extra blank horizon cell is deliberately not a Task 1 target.
                    buffer=masked.timestamp.isin(time_text[at:at+19])
                    drop=buffer&~removed
                    masked.loc[drop,['speed_kmh','flow_vph']]=np.nan
                masked.to_parquet(md/filename,index=False)
            for task in [1,2,4]:
                (root/f'task{task}'/panel/split).mkdir(parents=True,exist_ok=True)
            state=pd.concat(target_parts,ignore_index=True);state.to_csv(root/'task1'/panel/split/'sample_submission_state.csv',index=False)
            for part in [state.assign(task='state',split=split)]:
                master.append(part.drop(columns=['speed_kmh','flow_vph']))
            q=pd.concat(queue_parts,ignore_index=True);q.to_csv(root/'task2'/panel/split/'sample_submission_queue.csv',index=False)
            pd.DataFrame(window_parts).to_csv(root/'task2'/panel/split/'window_index.csv',index=False)
            pd.concat(history_parts,ignore_index=True).to_parquet(root/'task2'/panel/split/'window_history.parquet',index=False)
            master.append(q.drop(columns='queue_pred').assign(task='queue',split=split,panel=panel))
            fstar=rng.uniform(50,200,len(paths))
            prior=fstar*np.exp(rng.normal(0,.25,len(paths)))
            A=np.zeros((8,len(paths)))
            for j,path in enumerate(paths):
                a,b=map(int,path['path_id'].rsplit('P',1)[1].split('_'))
                A[a:b+1,j]=1
            pd.DataFrame({'link_id':links,'count':A@fstar}).to_csv(root/'task4'/panel/split/'synthetic_link_counts.csv',index=False)
            token=f'FIXTURE-{split.upper()}-PM'
            od=pd.DataFrame(paths).assign(panel=panel,departure_time=token,path_flow=prior)
            od[OD_KEY+['path_flow']].to_csv(root/'task4'/panel/split/'synthetic_weak_prior.csv',index=False)
            od[OD_KEY].assign(path_flow=0.).to_csv(root/'task4'/panel/split/'sample_submission_path_flow.csv',index=False)
            master.append(od[OD_KEY].assign(task='odme',split=split))
    key=pd.concat(master,ignore_index=True).fillna('')
    key.insert(0,'submission_id',np.arange(len(key)))
    key.to_csv(root/'submission_key.csv',index=False)
    sample=key[['submission_id','task']].copy()
    for c in ['speed_kmh','flow_vph','queue_pred','path_flow']:
        sample[c]=0.
    sample.to_csv(root/'sample_submission.csv',index=False)
    return {'root':str(root.resolve()),'panels':panels,'master_rows':len(key),'official':False}
