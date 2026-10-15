"""Secondary equal-alpha diagnostic on already-viewed public research arrays."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.state import StateModel,StateFeatures
from trafficflow.data import Release,VALUES,STATE_KEY,eligible
from trafficflow.experiments import partition_date,day_template
from trafficflow.outage_stress import outage_geometry,remove_measurements
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats,paired_block_interval
from trafficflow.runtime import capture_source_snapshot,sha256_file
from public_outage_stress import metric


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite equal-blend evidence')
    source_path=Path('reports/research/A88_I210_W_augmentation_future_outages.json');source=json.loads(source_path.read_text())
    train_path=Path(source['registered']['protocol']['source_study']);trained=json.loads(train_path.read_text());choices=trained['selection']['choices']
    names=list(choices);roles=names+['control_seed_mean','augmented_seed_mean'];dates=source['registered']['protocol']['dates']
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Secondary_equal_alpha_augmentation_diagnostic',
        'source_sha256':sha256_file(source_path),'training_sha256':sha256_file(train_path),'alpha':.5,
        'scope':'Already-viewed A88 teachers;all4perseed predictors directly re-inferred at alpha.5,no retraining/no reselection',
        'direct_replay':'Every ordinary date and stress case x4perseed models,set alpha.5 on fresh loaded copies',
        'status':'Post-outcome explanatory diagnostic,not fresh validation or a selected production policy',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    rows=[];stress=[];sources={};maximum=0.;direct=[]
    def directly_predict(features,target):
        values={name:model.predict(features,target)[VALUES].to_numpy(float,copy=True) for name,model in models.items()}
        for arm in ['control','augmented']:
            values[arm+'_seed_mean']=np.mean([values[f'seed{s}_{arm}'] for s in trained['registered']['protocol']['seeds']],axis=0)
        return values
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    models={n:StateModel.load(Path(source['registered']['protocol']['exports'][n]['path'])) for n in names}
    for m in models.values():m.alpha=.5
    release=Release('data/kaggle_public');panel='D7_I210_W';net,topology=release.network(panel),release.topology(panel)
    files={partition_date(release,p):p for p in release.files(panel,'train')};template=release.template(1,panel,'train')
    profile=models[names[0]].profile;version=models[names[0]].feature_version
    def record_roundoff(a,values,population):
        nonlocal maximum
        for name in names:
            inverted=a['lgb']+(.5/choices[name]['alpha'])*(a[name]-a['lgb'])
            error=float(np.abs(values[name]-inverted).max());maximum=max(maximum,error)
            direct.append({'population':population,'role':name,'maximum_error_of_discarded_linear_inversion_total_units':error})
    for record in source['ordinary_files']:
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Ordinary source changed')
        sources[record['path']]=record['sha256']
        frame=release.read(files[record['date']]);features=StateFeatures(frame,net,topology,profile,version)
        target=day_template(release,panel,'train',frame,template)
        with np.load(record['path']) as a:
            values=directly_predict(features,target);record_roundoff(a,values,'ordinary_'+record['date'])
            for role,p in values.items():
                m=metric(a['truth_total_units'],p,a['station_lanes']);old=next(r for r in source['ordinary_daily_stats'] if r['date']==record['date'] and r['role']==role)
                rows.append({'date':record['date'],'role':role,'regime':old['regime'],'n':m['n'],'speed_sse':m['speed_sse'],
                    'lane_flow_sse':m['lane_flow_sse'],'daily_State':m['synthetic_reconstruction_score']})
    frame_cache={d:release.read(files[d]) for d in dates}
    from trafficflow.experiments import raw_for
    raw_cache={d:next(raw_for(release,panel,[files[d]])) for d in dates}
    for record in source['case_records']:
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Stress source changed')
        sources[record['path']]=record['sha256']
        frame=frame_cache[record['date']];raw=raw_cache[record['date']]
        selected,geometry=outage_geometry(frame,panel,record['date'],record['mask_seed'],record['duration_steps'],record['spatial_fraction'])
        features=StateFeatures(remove_measurements(frame,selected),net,topology,profile,version)
        inside=raw.timestamp.between(geometry['start_timestamp'],geometry['last_removed_timestamp'])&raw.link_id.isin(geometry['selected_links'])
        target=raw.loc[inside&eligible(raw),['timestamp','station_id','link_id']].copy();target['panel']=panel;target['mask_regime']=frame.mask_regime.iloc[0]
        with np.load(record['path']) as a:
            if not np.array_equal(features.lookup(target),a['target_grid_indices']):raise ValueError('Saved teacher ordering differs')
            values=directly_predict(features,target[STATE_KEY]);record_roundoff(a,values,record['key'])
            for role,p in values.items():stress.append({k:record[k] for k in ['date','mask_seed','duration_steps','spatial_fraction','key']}|
                {'role':role,**metric(a['truth_total_units'],p,a['station_lanes'])})
        if len(stress)%(12*6)==0:print('equal_alpha_direct',record['date'],'complete',flush=True)
    ordinary={r:state_from_sufficient_stats([v for v in rows if v['role']==r]) for r in roles}
    ordinary_gain=ordinary['augmented_seed_mean']['S_state']-ordinary['control_seed_mean']['S_state']
    summary={}
    for duration,fraction in source['registered']['protocol']['cases']:
        subset=[r for r in stress if r['duration_steps']==duration and r['spatial_fraction']==fraction]
        scores={r:float(np.mean([v['synthetic_reconstruction_score'] for v in subset if v['role']==r])) for r in roles}
        paired=paired_block_interval([np.mean([v['synthetic_reconstruction_score'] for v in subset if v['date']==d and v['role']=='control_seed_mean']) for d in dates],
            [np.mean([v['synthetic_reconstruction_score'] for v in subset if v['date']==d and v['role']=='augmented_seed_mean']) for d in dates])
        summary[f'{duration*5}min_{int(fraction*100)}pct']={'scores':scores,'augmented_minus_control':scores['augmented_seed_mean']-scores['control_seed_mean'],'paired_day_diagnostic':paired}
    result={'registered':registered,'source_snapshot':snapshot,'ordinary':ordinary,'ordinary_augmented_minus_control':ordinary_gain,
        'ordinary_daily_stats':rows,'stress_summary':summary,'stress_rows':stress,'sources_sha256':sources,
        'n_direct_fixed_alpha_predictor_calls':len(direct),'direct_replays':direct,'maximum_error_of_discarded_linear_inversion_total_units':maximum,
        'seconds':time.perf_counter()-start,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Secondary analysis of viewed data;two optimizer seeds share calendars. Checkpoints held fixed,alpha.5 diagnostic not chosen from outcomes.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'diagnostic.json').write_text(text,encoding='utf-8')
    print('equal_alpha_complete',ordinary_gain,{k:v['augmented_minus_control'] for k,v in summary.items()},'direct',len(direct),'discarded_inversion_maxerr',maximum,flush=True)


if __name__=='__main__':main()
