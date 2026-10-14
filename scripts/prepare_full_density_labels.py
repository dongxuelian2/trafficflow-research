"""Prepare full public-train evaluator anchors separately from State inputs.

No hidden flux is read. Mode A public ramp/topology eligibility is for a
supervised observed-increment loss, not an official Physics score.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release, VALUES, eligible
from trafficflow.experiments import partition_date
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():
        raise FileExistsError('Never overwrite an input or label cache')
    info = json.loads((args.parent/'prepared.json').read_text())
    release = Release('data/kaggle_public')
    panel = info['panel']
    contract = Path('vendor/official/src/task3/score_task3.py')
    mode_path = Path('vendor/official/config/task3_lwr_modes.json')
    mode = json.loads(mode_path.read_text())['panels'][panel]['mode']
    if mode != 'A':
        raise ValueError('This preparation explicitly requires public Mode A')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id':'Public_train_full_observation_density_loss_labels',
        'parent_prepared_sha256':sha256_file(args.parent/'prepared.json'),
        'panel':panel,'fold':info['fold'],'coverage_mode':mode,
        'inputs':'Every parent input/target/label array copied unchanged; full truth exists only in label arrays',
        'labels':'Normalized raw eligible public TRAIN v/q, q per network lane; non-target evaluator anchors include released-input buffer blanks',
        'pair_coverage':'Eligible adjacent <=5.1min endpoints, observable internal topology, Mode A public on/off ramp validity at interval start',
        'boundary_flux_accessed':False,'formal_Physics':None,'formal_Total':None})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    specification = importlib.util.spec_from_file_location('published_density_contract', contract)
    official = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(official)
    panel_dir = release.root/'corridors'/panel
    topology = pd.read_csv(panel_dir/'network/lwr_mainline_topology.csv',dtype=str).fillna('').set_index('link_id')
    ramps = official.load_ramp_flows(panel_dir,'train')
    ramps['timestamp']=pd.to_datetime(ramps.timestamp,utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)}
    net=release.network(panel)
    records, summary=[],{}
    for part in ['fit','selection','assessment']:
        (args.output/part).mkdir()
        counts=dict(days=0,eligible_full_cells=0,non_target_eligible_input_blanks=0,
                    all_valid_pairs=0,target_touching_valid_pairs=0,
                    valid_target_pairs_excluded_by_old_visibility=0,
                    old_visible_target_pairs_excluded_by_public_coverage=0)
        for source in sorted((args.parent/part).glob('*.npz')):
            day=dict(np.load(source,allow_pickle=False))
            truth=release.read(raw_paths[source.stem])
            truth=truth.loc[eligible(truth)]
            times,links=day['times'].tolist(),day['links'].tolist()
            full=np.stack([truth.pivot_table(index='timestamp',columns='link_id',values=c,aggfunc='mean').reindex(index=times,columns=links).to_numpy(float) for c in VALUES],axis=-1)
            full[...,1]/=net.lanes.reindex(links).to_numpy(float)
            full/=np.array([25.,600.])
            valid=np.isfinite(full).all(axis=-1)
            labels=np.where(valid[...,None],full,0.).astype(np.float32)
            if not np.array_equal(labels[day['mask']],day['y'][day['mask']]):
                raise ValueError('Original scored target labels changed: '+source.stem)
            topo_ids,measured=set(topology.index),set(links)
            observable=np.array([link in topology.index and all(
                ({v.strip() for v in str(topology.loc[link,field]).split(';') if v.strip()}&topo_ids).issubset(measured)
                for field in ['incoming_link_ids','outgoing_link_ids']) for link in links])
            current=ramps[ramps.timestamp.isin(times)]
            flags=[]
            for kind in ['on','off']:
                attached={link for link in links if link in topology.index and str(topology.loc[link,kind+'_ramp_link_ids']).strip()}
                flag=current.pivot_table(index='timestamp',columns='link_id',values=kind+'_ramp_valid',aggfunc='max').reindex(index=times,columns=links).fillna(0).to_numpy(float)>=.5
                flag[:,[j for j,link in enumerate(links) if link not in attached]]=True
                flags.append(flag)
            dt=np.diff(pd.to_datetime(times,utc=True).as_unit('ns').asi8)/60e9
            pairs=valid[:-1]&valid[1:]&observable[None]&flags[0][:-1]&flags[1][:-1]&(dt[:,None]<=5.1)&(dt[:,None]>0)
            inputs=day.get('x_control',day['x'])
            visible=np.all(inputs[...,2:4]>.5,axis=-1)
            old_valid=visible|day['mask']
            touching=day['mask'][:-1]|day['mask'][1:]
            old_pairs=old_valid[:-1]&old_valid[1:]&touching
            item=dict(days=1,eligible_full_cells=int(valid.sum()),
                non_target_eligible_input_blanks=int((valid&~day['mask']&~visible).sum()),
                all_valid_pairs=int(pairs.sum()),target_touching_valid_pairs=int((pairs&touching).sum()),
                valid_target_pairs_excluded_by_old_visibility=int((pairs&touching&~old_pairs).sum()),
                old_visible_target_pairs_excluded_by_public_coverage=int((old_pairs&~pairs).sum()))
            for key,value in item.items():counts[key]+=value
            day.update(y_full=labels,eligible_full=valid,lwr_pair_valid=pairs)
            destination=args.output/part/source.name
            np.savez_compressed(destination,**day)
            # Verify parent data retained in the serialized output, including
            # NaNs and float32 targets; full labels must never replace x.
            loaded=dict(np.load(destination,allow_pickle=False))
            original=dict(np.load(source,allow_pickle=False))
            for key,array in original.items():
                try:equal=np.array_equal(array,loaded[key],equal_nan=True)
                except TypeError:equal=np.array_equal(array,loaded[key])
                if not equal:raise ValueError('Parent array changed: '+key)
            records.append({'part':part,'date':source.stem,'parent_day_sha256':sha256_file(source),
                'raw_TRAIN_label_source_sha256':sha256_file(raw_paths[source.stem]),
                'full_label_cache_sha256':sha256_file(destination),**item})
        summary[part]=counts
        print(part,counts,flush=True)
    (args.output/'graph.npz').write_bytes((args.parent/'graph.npz').read_bytes())
    result={'registered':registered,'source_snapshot':snapshot,'parent_cache':args.parent.as_posix(),
        'panel':panel,'fold':info['fold'],'dates':info['dates'],'records':records,'coverage_summary':summary,
        'graph_sha256':sha256_file(args.output/'graph.npz'),'parent_prepared_sha256':sha256_file(args.parent/'prepared.json'),
        'seconds':time.perf_counter()-start,'source':evidence(),
        'boundary_flux_accessed':False,'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limitations':'Training/scoring label cache only, not a trained model or measured gain. New full labels must not enter inference channels; raw train unavailable outside train split. Mode A public coverage still does not expose actual organizer flux.'}
    (args.output/'prepared.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    report.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
