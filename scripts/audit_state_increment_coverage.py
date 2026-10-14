"""Public-train observation increment diagnostic with Mode-A-like coverage.

Actual organizer boundary flows remain unavailable. RHS here is the difference
of public-train observation-derived accumulation, not the official flux RHS.
Neither the ratio nor its complement is a formal Physics component score.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from trafficflow.data import Release, VALUES
from trafficflow.state import StateModel, StateFeatures
from trafficflow.experiments import partition_date, day_template
from trafficflow.runtime import sha256_file, evidence, capture_source_snapshot
from trafficflow.research_eval import paired_block_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--baseline-file', type=Path, required=True)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory=Path('artifacts/research')/args.output.stem
    if args.output.exists() or directory.exists():
        raise FileExistsError('Never overwrite a coverage diagnostic')
    directory.mkdir(parents=True)
    snapshot=capture_source_snapshot(directory/'source_snapshot.zip')
    prepared = json.loads((args.cache/'prepared.json').read_text())
    release = Release('data/kaggle_public')
    panel = prepared['panel']
    panel_dir = release.root/'corridors'/panel
    official_path = Path('vendor/official/src/task3/score_task3.py')
    specification = importlib.util.spec_from_file_location('public_physics_contract', official_path)
    public = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(public)
    # Only published network parameters and ramp observations are reused.
    params = public.network_parameters(panel_dir).set_index('link_id')
    ramps = public.load_ramp_flows(panel_dir, 'train')
    ramps['timestamp'] = pd.to_datetime(ramps.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    topology = pd.read_csv(panel_dir/'network/lwr_mainline_topology.csv', dtype=str).fillna('').set_index('link_id')
    mode = json.loads(Path('vendor/official/config/task3_lwr_modes.json').read_text())['panels'][panel]['mode']
    if mode != 'A':
        raise ValueError('This diagnostic explicitly assumes the published Mode A')
    models = {'baseline': StateModel.load(args.baseline_file)}
    models.update({p.parent.name: StateModel.load(p) for p in args.model})
    masked = {partition_date(release, p): p for p in release.files(panel, 'train')}
    raw = {partition_date(release, p): p for p in release.files(panel, 'train', False)}
    net, topo = release.network(panel), release.topology(panel)
    template = release.template(1, panel, 'train')
    rows = []
    for source in sorted((args.cache/'assessment').glob('*.npz')):
        day = dict(np.load(source, allow_pickle=False))
        links, times = day['links'].tolist(), day['times'].tolist()
        truth = release.read(raw[source.stem])
        good = truth.is_score_eligible.astype(bool).to_numpy() & np.isfinite(truth[VALUES].to_numpy(float)).all(axis=1)
        truth = truth.loc[good]
        def grid(value):
            return truth.pivot_table(index='timestamp', columns='link_id', values=value, aggfunc='mean').reindex(index=times, columns=links).to_numpy(float)
        speed, flow = grid('speed_kmh'), grid('flow_vph')
        length = params.length_km.reindex(links).to_numpy(float)
        true_accumulation = flow/np.maximum(speed, 1.)*length[None]
        valid = np.isfinite(true_accumulation)
        topo_ids, measured = set(topology.index), set(links)
        observable = np.array([link in topology.index and all(
            ({v.strip() for v in str(topology.loc[link, field]).split(';') if v.strip()} & topo_ids).issubset(measured)
            for field in ['incoming_link_ids', 'outgoing_link_ids']) for link in links])
        current_ramps = ramps[ramps.timestamp.isin(times)]
        flags = []
        for ramp_type in ['on', 'off']:
            attached = {link for link in links if link in topology.index and str(topology.loc[link, ramp_type+'_ramp_link_ids']).strip()}
            flag = current_ramps.pivot_table(index='timestamp', columns='link_id', values=ramp_type+'_ramp_valid', aggfunc='max').reindex(index=times, columns=links).fillna(0).to_numpy(float) >= .5
            flag[:, [j for j, link in enumerate(links) if link not in attached]] = True
            flags.append(flag)
        eligible_pairs = valid[:-1] & valid[1:] & observable[None] & flags[0][:-1] & flags[1][:-1]
        touching = day['mask'][:-1] | day['mask'][1:]
        frame = release.read(masked[source.stem])
        features = StateFeatures(frame, net, topo, models['baseline'].profile, version=models['baseline'].feature_version)
        target = day_template(release, panel, 'train', frame, template)
        flat_indices = features.lookup(target)
        for name, model in models.items():
            prediction = model.predict(features, target)[VALUES].to_numpy(float,copy=True)
            unbounded=prediction.copy()
            prediction[:,0]=np.clip(prediction[:,0],0,target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
            prediction[:,1]=np.clip(prediction[:,1],0,target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)
            estimate_v, estimate_q = speed.copy().reshape(-1), flow.copy().reshape(-1)
            count = np.zeros(len(estimate_v))
            total = np.zeros((len(estimate_v), 2))
            np.add.at(total, flat_indices, prediction)
            np.add.at(count, flat_indices, 1)
            target_grid = count > 0
            estimate_v[target_grid], estimate_q[target_grid] = (total[target_grid]/count[target_grid, None]).T
            estimate_N = (estimate_q.reshape(speed.shape)/np.maximum(estimate_v.reshape(speed.shape), 1.))*length[None]
            predicted_delta, true_delta = np.diff(estimate_N, axis=0), np.diff(true_accumulation, axis=0)
            error = np.abs(predicted_delta-true_delta)
            denominator = np.abs(true_delta)
            all_error, all_denominator = float(error[eligible_pairs].sum()), float(denominator[eligible_pairs].sum())
            pair_target = eligible_pairs & touching
            rows.append({'date': source.stem, 'regime': str(day['regime']), 'model': name,
                'n_predictions_changed_by_common_bounds': int(np.any(prediction!=unbounded,axis=1).sum()),
                'n_observable_links': int(observable.sum()), 'n_all_valid_pairs': int(eligible_pairs.sum()),
                'n_target_touching_pairs': int(pair_target.sum()), 'error_L1': all_error,
                'observation_increment_denominator_L1': all_denominator,
                'projected_increment_relative_L1': all_error/max(all_denominator, 1e-9),
                'target_touching_denominator_L1': float(denominator[pair_target].sum()),
                'n_public_raw_non_target_cells': int((valid & ~day['mask']).sum()),
                'formal_Physics': None})
    summaries = {}
    for name in models:
        selected = [r for r in rows if r['model'] == name]
        details = {regime: sum(r['error_L1'] for r in selected if r['regime'] == regime)/max(sum(r['observation_increment_denominator_L1'] for r in selected if r['regime'] == regime), 1e-9)
            for regime in ['R1', 'R2', 'R3']}
        summaries[name] = {'regime_projected_increment_L1': details, 'macro_projected_increment_L1': float(np.mean(list(details.values())))}
    base = [r['projected_increment_relative_L1'] for r in rows if r['model'] == 'baseline']
    intervals = {name: paired_block_interval([-v for v in base], [-r['projected_increment_relative_L1'] for r in rows if r['model'] == name]) for name in models if name != 'baseline'}
    result = {'panel': panel, 'source_snapshot':snapshot,'rows': rows, 'summary': summaries, 'paired_day_error_reduction_intervals': intervals,
        'prediction_bounds':'Same 0..1.05 public free speed/capacity on all models, including full LGB',
        'models_sha256': {p.as_posix(): sha256_file(p) for p in [args.baseline_file, *args.model]},
        'public_contract_source_sha256': sha256_file(official_path), 'coverage_mode': mode,
        'boundary_flux_accessed': False, 'formal_State': None, 'formal_Physics': None, 'formal_Total': None,
        'source': evidence(gpu_used=any(getattr(m, '_network', None) is not None and next(m._network.parameters()).is_cuda for m in models.values())), 'limitations': 'Supervised train-only observed-increment diagnostic. Assumed observation projection, not actual organizer boundary flux; no official Physics score. Models/parameters are already sealed. Non-target raw labels are used as evaluator anchors only, never model inputs.'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(summaries, flush=True)


if __name__ == '__main__':
    main()
