"""Independent coordinate masks,raw teachers and rebuilt augmented inputs."""
import argparse
import json
from pathlib import Path
import time
import zlib
import numpy as np
import pandas as pd
from trafficflow.data import Release,eligible
from trafficflow.state import StateFeatures,StateModel
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.outage_stress import MEASUREMENTS
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot,sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite remask verification')
    source=json.loads(args.study.read_text());config=source['registered']['protocol']
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Independent_native_remask_raw_teacher_and_full_input_replay','source_study_sha256':sha256_file(args.study),
        'scope':'All736views,independent mask/removal implementation;16actual loss-target poison checks;all raw teacher grids',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    original=json.loads(Path(config['source_study']).read_text());panel=original['registered']['protocol']['panel']
    release=Release('data/kaggle_public');net,topology=release.network(panel),release.topology(panel)
    if sha256_file(config['baseline_file'])!=config['baseline_sha256']:raise ValueError('Frozen profile changed')
    baseline=StateModel.load(Path(config['baseline_file']));checks=[];maximum_teacher=0.;poison_checks=0
    for index,row in enumerate(source['rows']):
        for name in ['path','masked_file','raw_file','original_cache_path']:
            digest_key={'path':'sha256','original_cache_path':'original_cache_sha256'}.get(name,name+'_sha256')
            if sha256_file(row[name])!=row[digest_key]:raise ValueError('Input/cache bytes changed')
        raw=release.read(row['raw_file']);masked=release.read(row['masked_file'])
        if 'mask_regime' not in raw:raw['mask_regime']=row['regime']
        original_features=StateFeatures(masked,net,topology,baseline.profile,baseline.feature_version)
        times=sorted(raw.timestamp.unique());links=sorted(raw.link_id.unique());time_map={t:i for i,t in enumerate(times)};link_map={link:i for i,link in enumerate(links)}
        ti=raw.timestamp.map(time_map).to_numpy(int);li=raw.link_id.map(link_map).to_numpy(int)
        timestamp=pd.to_datetime(raw.timestamp,utc=True);good=eligible(raw);indices=original_features.lookup(raw.loc[good])
        counts=np.bincount(indices,minlength=288*len(links));teacher=np.zeros((288*len(links),2),np.float64)
        values=np.c_[raw.loc[good,'speed_kmh'].to_numpy(float)/25,
            raw.loc[good,'flow_vph'].to_numpy(float)/raw.loc[good,'link_id'].map(net.lanes).to_numpy(float)/600]
        for c in range(2):teacher[:,c]=np.divide(np.bincount(indices,weights=values[:,c],minlength=len(teacher)),counts,out=np.zeros(len(teacher)),where=counts>0)
        with np.load(row['path']) as arrays:
            difference=float(np.abs(teacher.reshape(arrays['y'].shape)-arrays['y']).max());maximum_teacher=max(maximum_teacher,difference)
            if difference>1e-6:raise ValueError('Raw teacher grid differs')
            for variant in ['point','queue']:
                for view,seed in enumerate(config['view_seeds']):
                    rng=np.random.default_rng(np.random.SeedSequence([seed,int(row['date'].replace('-','')),zlib.crc32(panel.encode())]))
                    selected=(rng.random((288,len(links)))<config['rates'][row['regime']])[ti,li]
                    origins=list(row['public_origins'])
                    if variant=='queue':
                        queue_rng=np.random.default_rng(np.random.SeedSequence([seed,int(row['date'].replace('-','')),zlib.crc32(panel.encode()),991]))
                        origins.append(times[int(queue_rng.integers(12,270))])
                    history=np.zeros(len(raw),bool);guard=np.zeros(len(raw),bool)
                    for text in origins:
                        origin=pd.Timestamp(text);delta=(timestamp-origin).dt.total_seconds().to_numpy()/60
                        history|=(delta>=-60)&(delta<=0);guard|=(delta>=5)&(delta<=90)
                    hidden=(selected&~history)|guard;targets=selected&~history&~guard&good
                    def independently_rebuild(poison=False):
                        frame=raw.copy(deep=True)
                        if poison:frame.loc[targets,[c for c in MEASUREMENTS if c!='pct_observed']]=1e8
                        for c in MEASUREMENTS:
                            if c in frame:frame.loc[hidden,c]=np.nan
                        for c,value in [('is_observed',0),('is_imputed',0),('is_score_eligible',0),('is_missing',1)]:
                            if c in frame:frame.loc[hidden,c]=value
                        return StateFeatures(frame,net,topology,baseline.profile,baseline.feature_version)
                    features=independently_rebuild();x=reconstruction_input(features);mask=np.zeros(x.shape[:2],bool)
                    mask.ravel()[features.lookup(raw.loc[targets])]=True;region=np.zeros_like(mask);region.ravel()[features.lookup(raw.loc[hidden])]=True
                    if not np.array_equal(x,arrays['x_'+variant][view]) or not np.array_equal(mask,arrays['mask_'+variant][view]) or not np.array_equal(region,arrays['hidden_'+variant][view]):
                        raise ValueError('Native remask input/loss/hidden region differs')
                    if x[...,2:4][mask].any() or x[...,9][mask].any() or targets[history|guard].any():raise ValueError('Target leakage or forbidden supervision')
                    if index in [0,len(source['rows'])-1]:
                        if not np.array_equal(reconstruction_input(independently_rebuild(True)),x):raise ValueError('Loss target values leaked into input')
                        poison_checks+=1
                    checks.append({'date':row['date'],'variant':variant,'view_seed':seed,'input_and_loss_mask_exact':True,'n_loss_grid_cells':int(mask.sum())})
        if (index+1)%20==0:print('native_cache_verify',index+1,'dates',flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'n_fit_dates':len(source['rows']),'n_views_independently_rebuilt':len(checks),
        'n_actual_target_poison_checks':poison_checks,'maximum_teacher_grid_difference_normalized_units':maximum_teacher,
        'checks':checks,'seconds':time.perf_counter()-start,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'scope':'Implementation/leakage verification using allowed public fit observations;not a new predictive validation or official mask selector reconstruction'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('native_cache_verified',len(checks),'poison',poison_checks,'teacher_max',maximum_teacher,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
