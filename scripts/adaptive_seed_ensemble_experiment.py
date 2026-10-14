"""Fixed equal-seed State mixture vs the matching equal-seed LGB mixture.

Calibration choices of each member remain sealed. Equal weights are registered
before assessment loading; seed duplicates are never independent day samples.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import Release, VALUES
from trafficflow.experiments import partition_date, day_template
from trafficflow.state import StateModel, StateFeatures
from trafficflow.state_ensemble import StateEnsemble
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.research_eval import register_protocol, state_from_sufficient_stats, paired_block_interval
from trafficflow.runtime import capture_source_snapshot, sha256_file, evidence
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.study) < 2:
        raise ValueError('At least two studies and a fresh output required')
    original = [json.loads((p/'experiment.json').read_text()) for p in args.study]
    configurations = [r['registered']['protocol'] for r in original]
    if len({c['panel'] for c in configurations}) != 1 or len({c['fold'] for c in configurations}) != 1 or len({c['seed'] for c in configurations}) != len(configurations):
        raise ValueError('Seed ensemble needs distinct fit seeds of the same direction/fold')
    members = []
    baselines = []
    records = []
    for directory, r, config in zip(args.study, original, configurations):
        replay = json.loads((directory/'deployment_verification.json').read_text())
        if r['selected'] != json.loads((directory/'selection_sealed.json').read_text()):
            raise ValueError('Member calibration selection changed')
        model_path = directory/'State_selected.pkl'
        baseline_path = Path(config['baseline_file'])
        if sha256_file(model_path) != replay['model_sha256'] or sha256_file(baseline_path) != config['baseline_sha256']:
            raise ValueError('Sealed ensemble member changed')
        members.append(StateModel.load(model_path))
        baselines.append(StateModel.load(baseline_path))
        records.append({'study': directory.as_posix(), 'seed': config['seed'], 'selected': r['selected'],
            'model_sha256': replay['model_sha256'], 'baseline_sha256': config['baseline_sha256']})
    candidate = StateEnsemble(members, np.ones(len(members)))
    comparator = StateEnsemble(baselines, np.ones(len(baselines)))
    if not comparator.profile.slots.equals(candidate.profile.slots) or not comparator.profile.fallback.equals(candidate.profile.fallback):
        raise ValueError('Candidate and LGB ensemble profiles differ')
    report = Path('reports/research')/(args.output.name+'.json')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'Fixed_equal_seed_adaptive_State_ensemble', 'members': records,
        'weights': candidate.weights.tolist(), 'baseline_weights': comparator.weights.tolist(),
        'selection': 'Member checkpoints/architectures/alpha fixed on original calibration; equal weights predetermined here, no assessment selection',
        'baseline': 'Mean predictions of the same full-label LGB fit seeds, not a single seed',
        'seed_uncertainty': 'Average member daily statistics before block resampling; shared dates are not independent samples',
        'assessment_status': 'Previously inspected public-train development dates; not final blinded data',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    release = Release('data/kaggle_public')
    panel = configurations[0]['panel']
    net, topo = release.network(panel), release.topology(panel)
    paths = {partition_date(release, p): p for p in release.files(panel, 'train')}
    template = release.template(1, panel, 'train')
    cache = Path(configurations[0]['cache'])
    result = {'registered': registered, 'source_snapshot': snapshot, 'partitions': {}, 'member_stats': {}}
    for partition in ('selection', 'assessment'):
        if partition == 'assessment':
            (args.output/'ensemble_sealed.json').write_text(json.dumps(registered, indent=2), encoding='utf-8')
        rows = {'LGB_equal_seed': [], 'adaptive_equal_seed': []}
        increments = {name: [] for name in rows}
        seed_rows = {c['seed']: {'baseline': [], 'candidate': []} for c in configurations}
        for path in sorted((cache/partition).glob('*.npz')):
            day = dict(np.load(path, allow_pickle=False))
            f = StateFeatures(release.read(paths[path.stem]), net, topo, candidate.profile, version=candidate.feature_version)
            if not np.array_equal(reconstruction_input(f), day['x']):
                raise ValueError('Independent ensemble inputs differ from audited cache')
            target = day_template(release, panel, 'train', f.frame, template)
            if not np.array_equal(f.lookup(target), day['target_indices']):
                raise ValueError('Ensemble target order changed')
            lane = target.link_id.map(net.lanes).to_numpy(float)
            factor = (net.lanes*net.length_km).reindex(day['links']).to_numpy(np.float32)
            baseline_predictions, candidate_predictions = [], []
            for config, baseline, member in zip(configurations, baselines, members):
                base = baseline.predict(f, target)[VALUES].to_numpy(float, copy=True)
                base[:, 0] = np.minimum(base[:, 0], target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
                base[:, 1] = np.minimum(base[:, 1], target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/lane
                pred = member.predict(f, target)[VALUES].to_numpy(float, copy=True)
                pred[:, 1] /= lane
                baseline_predictions.append(base)
                candidate_predictions.append(pred)
                seed_rows[config['seed']]['baseline'].append(stats(day, base))
                seed_rows[config['seed']]['candidate'].append(stats(day, pred))
            for name, predictions in (('LGB_equal_seed', baseline_predictions), ('adaptive_equal_seed', candidate_predictions)):
                prediction = np.mean(predictions, axis=0)
                rows[name].append(stats(day, prediction))
                increments[name].append(increment_stats(day, prediction, factor))
        metrics = {}
        for name in rows:
            metrics[name] = {**state_from_sufficient_stats(rows[name]), 'days': rows[name],
                'density_target_pair_relative_L1': sum(d['error_L1'] for d in increments[name])/sum(d['truth_increment_L1'] for d in increments[name])}
        result['partitions'][partition] = metrics
        result['member_stats'][partition] = seed_rows
        for r, config in zip(original, configurations):
            score = state_from_sufficient_stats(seed_rows[config['seed']]['candidate'])['S_state']
            expected = r['selected_assessment']['S_state'] if partition == 'assessment' else r['selected']['selection_State']
            if abs(score-expected) > 1e-7:
                raise ValueError('An ensemble member did not independently reproduce')
    assessment = result['partitions']['assessment']
    result['paired_daily_interval'] = paired_block_interval(
        [r['daily_State'] for r in assessment['LGB_equal_seed']['days']],
        [r['daily_State'] for r in assessment['adaptive_equal_seed']['days']])
    seed_stats = result['member_stats']['assessment']
    seed_base = np.mean([[d['daily_State'] for d in row['baseline']] for row in seed_stats.values()], axis=0)
    seed_candidate = np.mean([[d['daily_State'] for d in row['candidate']] for row in seed_stats.values()], axis=0)
    result['mean_seed_daily_gain_interval'] = paired_block_interval(seed_base, seed_candidate)
    wrapper_prediction = candidate.predict(f, target)[VALUES].to_numpy(float, copy=True)
    wrapper_prediction[:, 1] /= lane
    np.testing.assert_allclose(wrapper_prediction, prediction, rtol=0, atol=1e-6)
    candidate.save(args.output/'State_equal_seed.pkl')
    reloaded = StateModel.load(args.output/'State_equal_seed.pkl')
    replay = reloaded.predict(f, target)[VALUES].to_numpy(float, copy=True)
    replay[:, 1] /= lane
    np.testing.assert_allclose(replay, prediction, rtol=0, atol=1e-6)
    result.update(seconds=time.perf_counter()-start, model_sha256=sha256_file(args.output/'State_equal_seed.pkl'),
        wrapper_prediction_and_reload_verified_on_last_assessment_day=True,
        wrapper_replay_scope='One full assessment day for the wrapper/load path; all member predictions replayed on all selection/assessment days.',
        source=evidence(gpu_used=torch.cuda.is_available()), formal_State=None, formal_Physics=None, formal_Total=None)
    (args.output/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(assessment['LGB_equal_seed']['S_state'], assessment['adaptive_equal_seed']['S_state'],
        result['paired_daily_interval'], result['seconds'], flush=True)
