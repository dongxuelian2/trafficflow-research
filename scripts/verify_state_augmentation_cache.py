"""Independent cache reconstruction without the augmentation removal helper."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.data import Release
from trafficflow.state import StateFeatures,StateModel
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.outage_stress import MEASUREMENTS
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Use a new cache verification')
    study=json.loads(args.study.read_text());config=study['registered']['protocol'];source=Path(config['source_cache'])
    if sha256_file(source/'prepared.json')!=config['prepared_sha256'] or sha256_file(config['baseline_file'])!=config['baseline_sha256']:
        raise ValueError('Original profile/cache provenance changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Full_fit_augmentation_cache_independent_measurement_removal_replay','source_study_sha256':sha256_file(args.study),
        'scope':'Every stored input view and source/region;independent pandas removal,not rebuild_after_outage',
        'poison_checks':'First and last fit date,each4views;all removed measurement fields replaced by extremes before independent deletion',
        'raw_teachers_opened':False,'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');panel=json.loads((source/'prepared.json').read_text())['panel']
    net,topology=release.network(panel),release.topology(panel);baseline=StateModel.load(Path(config['baseline_file']))
    rows=[];poison_count=0
    def independently_hide(frame,selected,poison=False):
        result=frame.copy(deep=True)
        if poison:result.loc[selected,[k for k in MEASUREMENTS if k in result]]=1e12
        for column in MEASUREMENTS:
            if column in result:result.loc[selected,column]=np.nan
        for column,value in [('is_observed',0),('is_imputed',0),('is_score_eligible',0),('is_missing',1)]:
            if column in result:result.loc[selected,column]=value
        return StateFeatures(result,net,topology,baseline.profile,version=baseline.feature_version)
    for index,row in enumerate(study['rows']):
        if sha256_file(row['path'])!=row['sha256'] or sha256_file(row['masked_file'])!=row['masked_file_sha256']:
            raise ValueError('Augmented or public masked input changed')
        cached=source/'fit'/(row['date']+'.npz')
        if sha256_file(cached)!=row['source_cache_sha256']:raise ValueError('Original fit cache changed')
        frame=release.read(row['masked_file']);original=StateFeatures(frame,net,topology,baseline.profile,version=baseline.feature_version)
        with np.load(cached) as a:
            if not np.array_equal(reconstruction_input(original),a['x']):raise ValueError('Original strict12 input differs')
        with np.load(row['path']) as views:
            for view,geometry in enumerate(row['geometries']):
                selected=(frame.timestamp.between(geometry['start_timestamp'],geometry['last_removed_timestamp'])&frame.link_id.isin(geometry['selected_links'])).to_numpy()
                features=independently_hide(frame,selected);x=reconstruction_input(features)
                time_region=np.array([(geometry['start_timestamp']<=t<=geometry['last_removed_timestamp']) for t in features.times])
                link_region=np.array([link in geometry['selected_links'] for link in features.links]);region=time_region[:,None]&link_region[None]
                if not np.array_equal(x,views['x'][view]) or not np.array_equal(region,views['regions'][view]):raise ValueError('Rebuilt input/region differs')
                if x[...,2:4][region].any() or x[...,9][region].any():raise ValueError('Measurement remained visible')
                if index in [0,len(study['rows'])-1]:
                    poison=reconstruction_input(independently_hide(frame,selected,poison=True))
                    if not np.array_equal(poison,x):raise ValueError('Removed measurement leaked through features')
                    poison_count+=1
                rows.append({'date':row['date'],'view':view,'inputs_exact':True,'region_exact':True,
                    'removed_visible_input_channels_all_zero':True,'selected_frame_rows':int(selected.sum())})
        if (index+1)%20==0:print('augmentation_cache_replay',index+1,'dates',flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'n_fit_dates':len(study['rows']),'n_views_rebuilt':len(rows),
        'n_actual_public_measurement_poison_checks':poison_count,'all_inputs_and_regions_exact':True,'checks':rows,
        'seconds':time.perf_counter()-start,'raw_teachers_opened':False,'formal_State':None,'formal_Physics':None,'formal_Total':None}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('augmentation_cache_replay_complete',len(rows),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
