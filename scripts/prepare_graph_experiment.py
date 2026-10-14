"""Prepare public-train rolling folds and a same-split LightGBM comparator."""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release, VALUES, eligible
from trafficflow.experiments import partition_date, raw_for, day_template, target_truth, training_rows
from trafficflow.state import Profile, StateModel, StateFeatures, neighbors
from trafficflow.research_eval import Fold, canonical_hash
from trafficflow.runtime import evidence,capture_source_snapshot,sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', default='D12_I5_N')
    parser.add_argument('--fold', default='rolling_autumn')
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new cache/version')
    started = time.perf_counter()
    release = Release('data/kaggle_public')
    protocol = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())
    fold = Fold(**next(f for f in protocol['protocol']['folds'] if f['name']==args.fold))
    paths = release.files(args.panel, 'train')
    dates = fold.assign([partition_date(release, p) for p in paths])
    parts = {key:[p for p in paths if partition_date(release, p) in days] for key, days in dates.items()}
    args.output.mkdir(parents=True)
    snapshot=capture_source_snapshot(args.output/'source_snapshot.zip')
    profile = Profile.fit(raw_for(release, args.panel, parts['profile']))
    model = StateModel(profile, 'lightgbm', seed=args.seed, iterations=400, feature_version=2, num_leaves=63)
    model.density_blend = 0.
    model.fit(*training_rows(release, args.panel, parts['fit'], profile, 300000, args.seed, 2))
    model.save(args.output/'lightgbm.pkl')
    net, topo = release.network(args.panel), release.topology(args.panel)
    template = release.template(1, args.panel, 'train')
    sources = []
    reference_links = None
    for part in ('fit', 'selection', 'assessment'):
        for path, raw in zip(parts[part], raw_for(release, args.panel, parts[part])):
            frame = release.read(path)
            features = StateFeatures(frame, net, topo, profile, version=2)
            times, links = features.times, features.links
            if reference_links is None:
                reference_links = links
            elif links != reference_links:
                raise ValueError('Graph node ordering changed between dates')
            shape = (len(times), len(links))
            scale = np.array([25., 600.])
            lanes = net.lanes.reindex(links).to_numpy(float)
            observed = features.raw / scale
            available = np.isfinite(observed)
            known = frame.copy()
            known.loc[~eligible(frame), 'occupancy'] = np.nan
            occupancy = known.pivot_table(index='timestamp', columns='link_id', values='occupancy', aggfunc='mean').reindex(index=times, columns=links).to_numpy(float)
            dt = pd.to_datetime(times, utc=True)
            angle = 2*np.pi*(dt.hour*60+dt.minute)/1440
            sin = np.broadcast_to(np.asarray(np.sin(angle))[:,None,None], (*shape,1))
            cos = np.broadcast_to(np.asarray(np.cos(angle))[:,None,None], (*shape,1))
            x = np.concatenate([np.nan_to_num(observed), available.astype(float), features.prior/scale,
                features.base/scale, np.nan_to_num(occupancy[...,None])/.25,
                np.isfinite(occupancy[...,None]), sin, cos], axis=-1).astype(np.float32)
            target = day_template(release, args.panel, 'train', frame, template)
            truth = target_truth(release, args.panel, target, raw)
            indices = features.lookup(target)
            y = np.zeros((*shape,2), np.float32)
            mask = np.zeros(shape, bool)
            truth_lane = truth[VALUES].to_numpy(float,copy=True)
            truth_lane[:,1] /= target.link_id.map(net.lanes).to_numpy(float)
            # Multiple stations on a link share the grid estimate; retain all
            # station labels separately for exact public-template evaluation.
            aggregate = pd.DataFrame({'i': indices, 'v': truth_lane[:,0], 'q': truth_lane[:,1]}).groupby('i')[['v','q']].mean()
            y.reshape(-1,2)[aggregate.index] = aggregate.to_numpy()/scale
            mask.ravel()[indices] = True
            comparator = model.predict(features, target)[VALUES].to_numpy(float,copy=True)
            comparator[:,0] = np.minimum(comparator[:,0], target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
            comparator[:,1] = np.minimum(comparator[:,1], target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)
            comparator[:,1] /= target.link_id.map(net.lanes).to_numpy(float)
            output = args.output/part/(times[0][:10]+'.npz')
            output.parent.mkdir(exist_ok=True)
            np.savez_compressed(output, x=x, y=y, mask=mask, base=(features.base/scale).astype(np.float32),
                target_indices=indices, station_truth=truth_lane.astype(np.float64),
                comparator=comparator, station_lanes=release.station_lanes(args.panel,target),
                target_link_lanes=target.link_id.map(net.lanes).to_numpy(float),
                regime=target.mask_regime.iloc[0], links=np.array(links), times=np.array(times))
            sources.append({'part':part,'path':path.as_posix(),'cache':output.as_posix()})
        print(args.panel, part, len(parts[part]), flush=True)
    adjacency = []
    for direction in ('incoming', 'outgoing'):
        matrix = np.zeros((len(links),len(links)), np.float32)
        for j, connected in enumerate(neighbors(topo, links, direction)):
            if connected:
                matrix[j, connected] = 1 / len(connected)
        adjacency.append(matrix)
    fd = np.stack([net.free_speed_kmh.reindex(links), net.capacity_vph.reindex(links)/lanes,
                   net.k_jam.reindex(links)/lanes], axis=0).astype(np.float32)
    np.savez(args.output/'graph.npz', incoming=adjacency[0], outgoing=adjacency[1], fd=fd)
    result = {'experiment':'A01_masked_graph_state', 'panel':args.panel,'fold':args.fold,'seed':args.seed,
        'protocol_sha256':protocol['protocol_sha256'],'dates':dates,'source':evidence(),
        'channels':['speed/25','lane_flow/600','speed_available','flow_available','profile_speed/25',
        'profile_lane_flow/600','base_speed/25','base_lane_flow/600','occupancy/.25','occupancy_available','sin_tod','cos_tod'],
        'sources':sources,'seconds':time.perf_counter()-started,'formal_score':None,
        'source_snapshot':snapshot,'LGB_model_sha256':sha256_file(args.output/'lightgbm.pkl'),
        'graph_sha256':sha256_file(args.output/'graph.npz')}
    (args.output/'prepared.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({'prepared_seconds':result['seconds'],'links':len(links)},flush=True)


if __name__ == '__main__':
    main()
