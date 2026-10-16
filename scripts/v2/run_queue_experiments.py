"""Preregister and run a small conditional observed-event Queue falsification.

Execution requires an existing first-stage State report.  Preparation reads old
metadata only; experiment labels are opened in the five preserved time roles.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import time
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from trafficflow.data import Release
from trafficflow.full_queue_calibration import full_queue_decisions, tune_full_probabilities
from trafficflow.queue_calibration import RefinedQueueModel
from trafficflow.queue_domain import score_full_forecast
from trafficflow.queue_phase_split import validate_phase_origins
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.v2.event_posterior import (ConditionalEventPosterior, signed_graph_distance,
    expected_iou_decode, independent_trajectories, stable_origin_seed)

CASES = [
    'C55_I5_S_autumn_five_phase_balanced_v2',
    'C56_I5_S_winter_five_phase_balanced_v2',
    'C57_I10_W_autumn_five_phase_balanced_v2',
    'C58_I10_W_winter_five_phase_balanced_v2',
]
SEEDS = [20261016, 20261029]
ARMS = ['old_probability_old_decoder', 'old_probability_new_decoder',
        'event_probability_old_decoder', 'event_probability_new_decoder']


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Never overwrite Queue research evidence: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def protocol():
    return {
        'id': 'V2_conditional_observed_event_mixture_v1',
        'cases': CASES, 'scenario_seeds': SEEDS, 'draws': 256,
        'source_cache': 'Preserved C55-C58 balanced_v2 five-phase histories, targets and frozen strong trees.',
        'direction_deviation': 'I5_N has C30/C33 three-phase caches only; use I5_S and I10_W two seasons to preserve truly separate five roles.',
        'fit': 'Fit-only condition-separated empirical joint trajectories, Gaussian threshold/observation smoothing; legal thirteen-slot history has fixed 8-bin summaries.',
        'checkpoint': 'Choose nearest-neighbor count from [8,24] by equal-condition/equal-window Brier, ties prefer24. No other event hyperparameter search.',
        'decision': 'Freeze old incumbent/tree/thresholds from original decision seal; calibrate event old-decoder with exactly the original 31-threshold/condition/force grid. New decoder has zero tuned parameters.',
        'selection': 'Primary event+new-decoder versus strong old+old, gain>=.02 and day-block CI lower>0. No retuning; seal before assessment.',
        'assessment': 'Report every fixed four-way arm and both MC seeds, even failed arms. Main research gate requires overall gain>=.02, lower95CI>0, positive sign in all four direction-season cases and >=30 unique onset origins.',
        'decoder': 'Same finite candidate rule/scenario budget for both information arms: empty, 5 thresholds,7 count prefixes,16 sampled trajectories; joint six-step average scenario IoU. Old p uses IID scenarios, event p uses shared fit template and temporally correlated observation noise.',
        'information_control': 'Old p+new decoder retains exact old marginal p; event p+old decoder uses same old decision algorithm/calibration grid. Four-way experiment separates new conditional marginals and scenario decision effects, but does not separately identify every dependence assumption.',
        'relationship_to_C79': 'Both use empirical joint scenarios. C79 fixed tree marginals and fit-only unconditional empirical dependence; this model predicts new marginals and trajectory weights from H,E. No claim of novel generic kNN or MC-IoU algorithm.',
        'noise_assumption': 'sigma=clip(robust five-minute speed difference scale/sqrt(2),1.5,8)km/h; traffic innovations confound actual sensor noise. Shared Gaussian threshold noise correlation .5 fixed.',
        'prediction_domain': 'Full public node x6 domain; onset first5 empty from announced condition. Future availability,quality,observations are training targets or evaluator-only, never prediction-set inputs.',
        'sample_bias': 'Preserved proxy sampler selected future observed positives. Its population is not the unknown official latent selector; this bias is not solved.',
        'date_provenance': 'All public calendars previously used for development; these are not blind tests.',
        'uncertainty': '4000 paired calendar-day circular moving-block draws,7-day blocks; origins deduplicated across event sampler seeds. Same calendar draws shared across corridors; separate seasons sampled separately. MC seeds are not independent traffic events.',
        'interpretation': 'Observed noisy Queue proxy only. Bottleneck denotes an observed trajectory location tag, not identified causal capacity drop; graph-incomplete direction/front summaries stay qualified. No latent true Queue labels available.',
        'stop': 'Do not expand architecture or threshold/search grid if initial gate fails.',
        'formal_Queue': None, 'formal_Total': None, 'uploaded': False,
    }


class PreservedCache:
    def __init__(self, name, output):
        self.directory = Path('artifacts/research')/name
        self.report_path = self.directory/'experiment.json'
        self.report = json.loads(self.report_path.read_text(encoding='utf-8'))
        self.output = output
        self.records = self.report['cache_records']
        self.reads = []
        self.dates = self.report['dates']
        self.panel = self.report['registered']['protocol']['panel']
        decision = json.loads((self.directory/'decision_sealed.json').read_text(encoding='utf-8'))
        self.incumbent = decision['incumbent']
        self.choice = decision['choices'][self.incumbent]
        if self.choice['members']:
            raise ValueError('Expected the preserved strong tree incumbent')
        self.tree_path = self.directory/('tree_'+self.choice['baseline_arm']+'.pkl')
        self.tree = RefinedQueueModel.load(self.tree_path)

    def load(self, phase, seed):
        required = {'decision': 'checkpoint_sealed.json', 'selection': 'calibration_sealed.json',
                    'assessment': 'selection_sealed.json'}.get(phase)
        if required and not (self.output/required).is_file():
            raise ValueError('Earlier choice must be sealed before '+phase)
        record = next(r for r in self.records if r['phase'] == phase and r['seed'] == seed)
        path = Path(record['path'])
        if sha256_file(path) != record['sha256']:
            raise ValueError('Preserved Queue cache changed: '+str(path))
        arrays = dict(np.load(path, allow_pickle=False))
        validate_phase_origins(phase, arrays['origins'].tolist(), self.dates)
        if arrays['x'].shape[1:] != (13, len(arrays['links']), 9):
            raise ValueError('Thirteen-slot causal input cache required')
        if not np.array_equal(arrays['x'][:, 0, 0, 8].astype(bool), arrays['ongoing']):
            raise ValueError('Public event condition/history disagree')
        arm = self.choice['baseline_arm']
        record_x = next(p for p in record['feature_files'] if p['path'].endswith('_'+arm+'.parquet'))
        feature_path = Path(record_x['path'])
        if sha256_file(feature_path) != record_x['sha256']:
            raise ValueError('Frozen tree features changed')
        features = pd.read_parquet(feature_path)
        self.reads.append({'phase': phase, 'seed': seed, 'path': path.as_posix(),
                           'sha256': record['sha256'], 'features_sha256': record_x['sha256'],
                           'prior_seal_sha256': sha256_file(self.output/required) if required else None})
        return arrays, features


def condition_brier(probability, arrays):
    # Evaluation-only mask, after all probabilities have been produced.
    valid = arrays['available'].copy()
    valid[~arrays['ongoing'], :5] = False
    count = valid.sum((1, 2))
    error = (((probability-arrays['labels'])**2)*valid).sum((1, 2))/count.clip(min=1)
    return float(np.mean([error[arrays['ongoing'] == c].mean() for c in [False, True]]))


def old_decode(probability, arrays, setting):
    return full_queue_decisions(probability.ravel(), arrays['window'], arrays['condition'],
        arrays['lead_minutes'], setting['thresholds'], setting['require_nonempty'], setting['timing']).reshape(probability.shape)


def predict_four(model, old_probability, histories, origins, ongoing, old_setting, event_setting, metadata, seed):
    """Prediction deliberately receives no labels, availability, or quality."""
    event_probability = model.probability(histories)
    result = {
        ARMS[0]: old_decode(old_probability, metadata, old_setting),
        ARMS[2]: old_decode(event_probability, metadata, event_setting),
        ARMS[1]: np.empty(old_probability.shape, np.int8),
        ARMS[3]: np.empty(old_probability.shape, np.int8),
    }
    summaries = []
    for i, origin in enumerate(origins):
        local_seed = stable_origin_seed(origin, seed)
        old_scenarios = independent_trajectories(old_probability[i], 256, local_seed)
        event_scenarios = model.sample(histories[i:i+1], 256, local_seed)[0]
        old_action, old_detail = expected_iou_decode(old_probability[i], old_scenarios, bool(ongoing[i]))
        event_action, event_detail = expected_iou_decode(event_probability[i], event_scenarios, bool(ongoing[i]))
        result[ARMS[1]][i], result[ARMS[3]][i] = old_action, event_action
        summaries.append({'origin': str(origin), 'old': old_detail, 'event': event_detail})
    return result, event_probability, summaries


def paired_comparison(reports, candidate=ARMS[3], baseline=ARMS[0], seed=20261016, resamples=4000):
    """Synchronous date bootstrap, equal cases then equal public conditions."""
    rng = np.random.default_rng(seed)
    calendars, all_scores, counts = {}, [], []
    for report in reports:
        dates = tuple(report['dates'])
        if dates not in calendars:
            n = len(dates)
            starts = rng.integers(n, size=(resamples, int(np.ceil(n/7))))
            calendars[dates] = ((starts[..., None]+np.arange(7)) % n).reshape(resamples, -1)[:, :n]
        rows = []
        for run in report['runs']:
            left = run['models'][baseline]['windows']
            right = {r['window'].rsplit(':', 1)[0]: r for r in run['models'][candidate]['windows']}
            for value in left:
                origin = value['window'].rsplit(':', 1)[0]
                other = right[origin]
                rows.append({'origin': origin, 'date': pd.Timestamp(origin.split(':', 1)[1]).date().isoformat(),
                             'condition': value['condition'], 'left': value['IoU'], 'right': other['IoU']})
        frame = pd.DataFrame(rows)
        for _, group in frame.groupby('origin'):
            if group.condition.nunique() != 1 or not np.allclose(group[['left', 'right']], group[['left', 'right']].iloc[0], atol=1e-12, rtol=0):
                raise ValueError('Repeated physical origins disagree')
        frame = frame.drop_duplicates('origin')
        delta, count = np.zeros((len(dates), 2)), np.zeros((len(dates), 2))
        for row in frame.itertuples():
            d, c = dates.index(row.date), int(row.condition == 'queue_ongoing')
            delta[d, c] += row.right-row.left
            count[d, c] += 1
        index = calendars[dates]
        total_count, total_delta = count[index].sum(axis=1), delta[index].sum(axis=1)
        sample = np.where((total_count > 0).all(axis=1),
                          np.mean(total_delta/total_count.clip(min=1), axis=1), np.nan)
        gain = frame.groupby('condition')[['left', 'right']].mean().mean()
        case_gain = float(gain.right-gain.left)
        all_scores.append((sample, case_gain))
        counts.append({'case': report['case'], 'unique_origins': len(frame),
                       'unique_onset': int((frame.condition == 'queue_onset').sum()),
                       'sampled_windows': len(rows), 'days_with_events': int(frame.date.nunique()),
                       'paired_macro_gain': case_gain})
    samples = np.mean(np.stack([r[0] for r in all_scores]), axis=0)
    valid = np.isfinite(samples)
    gain = float(np.mean([r[1] for r in all_scores]))
    interval = np.quantile(samples[valid], [.025, .975]).tolist() if valid.any() else None
    return {'baseline': baseline, 'candidate': candidate, 'paired_macro_gain': gain, 'ci95': interval,
            'cases': counts, 'valid_resamples': int(valid.sum()), 'resamples': resamples,
            'research_gate_pass': bool(gain >= .02 and interval is not None and interval[0] > 0
                and all(r['paired_macro_gain'] > 0 for r in counts)
                and sum(r['unique_onset'] for r in counts) >= 30),
            'proxy_only': True}


def evaluate(cache, model, phase, output, event_setting):
    by_seed = {str(seed): [] for seed in SEEDS}
    for event_seed in [2909, 3253, 4001]:
        arrays, features = cache.load(phase, event_seed)
        old_probability = cache.tree.probability(features).reshape(arrays['labels'].shape)
        metadata = {k: arrays[k] for k in ['window', 'condition', 'lead_minutes']}
        for seed in SEEDS:
            pred, probability, details = predict_four(model, old_probability, arrays['x'], arrays['origins'],
                arrays['ongoing'], cache.choice['setting'], event_setting, metadata, seed)
            metrics = {name: score_full_forecast(arrays['labels'].ravel(), value.ravel(), arrays['window'],
                       arrays['condition'], arrays['available'].ravel()) for name, value in pred.items()}
            run = {'event_seed': event_seed, 'scenario_seed': seed, 'models': metrics,
                   'old_Brier': condition_brier(old_probability, arrays),
                   'event_Brier': condition_brier(probability, arrays), 'decoder': details}
            by_seed[str(seed)].append(run)
            summary = model.event_summary(arrays['x'])
            np.savez_compressed(output/(phase+'_'+str(event_seed)+'_'+str(seed)+'_predictions.npz'),
                old_probability=old_probability, event_probability=probability, **pred,
                origins=arrays['origins'], **summary)
        print(cache.panel, phase, event_seed, 'windows', len(arrays['origins']), flush=True)
    return by_seed


def run_case(name, output):
    start = time.perf_counter()
    output.mkdir()
    cache = PreservedCache(name, output)
    fit, _ = cache.load('fit', 20261013)
    release = Release('data/kaggle_public')
    network, topology = release.network(cache.panel), release.topology(cache.panel)
    distance = signed_graph_distance(fit['links'], network, topology)
    model = ConditionalEventPosterior.fit(fit['x'], fit['truth_speed'], fit['available'],
        network.v_cut.reindex(fit['links']).to_numpy(float), distance)
    checkpoint, _ = cache.load('checkpoint', 4129)
    scores = {}
    for k in [8, 24]:
        model.neighbors = k
        scores[str(k)] = condition_brier(model.probability(checkpoint['x']), checkpoint)
    model.neighbors = min([8, 24], key=lambda k: (scores[str(k)], k != 24))
    model_path = output/'model.pkl'
    model_path.write_bytes(pickle.dumps(model, protocol=5))
    write_json(output/'checkpoint_sealed.json', {'checkpoint_Brier': scores,
        'selected_neighbors': model.neighbors, 'sigma_speed_kmh': model.sigma_speed_kmh,
        'model_sha256': sha256_file(model_path), 'fitting_randomness': 'deterministic, no optimization seed'})
    decision, _ = cache.load('decision', 4133)
    dp = model.probability(decision['x'])
    setting = tune_full_probabilities(dp.ravel(), decision['window'], decision['condition'], decision['lead_minutes'],
                                      decision['labels'].ravel(), decision['available'].ravel())['selected']
    write_json(output/'calibration_sealed.json', {'event_old_decoder': setting,
        'old_decoder': cache.choice['setting'], 'fixed_primary': ARMS[3],
        'new_decoder_tuned_parameters': 0, 'model_sha256': sha256_file(model_path)})
    selection = evaluate(cache, model, 'selection', output, setting)
    selection_report = {'case': name, 'dates': cache.dates['selection'], 'runs': selection[str(SEEDS[0])]}
    gate = paired_comparison([selection_report])
    write_json(output/'selection_sealed.json', {'gate': gate, 'candidate_accepted': gate['research_gate_pass'],
        'selected': ARMS[3] if gate['research_gate_pass'] else ARMS[0],
        'assessment_all_fixed_arms': True})
    assessment = evaluate(cache, model, 'assessment', output, setting)
    result = {'case': name, 'panel': cache.panel, 'dates': cache.dates,
        'selection': selection, 'assessment': assessment, 'selection_gate': gate,
        'model_sha256': sha256_file(model_path), 'frozen_tree_sha256': sha256_file(cache.tree_path),
        'source_experiment_sha256': sha256_file(cache.report_path), 'phase_reads': cache.reads,
        'checkpoint_Brier': scores, 'neighbors': model.neighbors, 'sigma_speed_kmh': model.sigma_speed_kmh,
        'graph_resolved_pair_fraction': float(np.isfinite(distance).mean()),
        'seconds': time.perf_counter()-start, 'runtime': evidence(False),
        'formal_Queue': None, 'formal_Total': None, 'uploaded': False}
    write_json(output/'case_completed.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('reports/v2/QUEUE_CONDITIONAL_RUN'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--stage1-report', type=Path)
    args = parser.parse_args()
    if not args.output.name.startswith('QUEUE') or args.output.parent != Path('reports/v2'):
        raise ValueError('Queue evidence must live in a distinct reports/v2/QUEUE* directory')
    preregister = args.output.with_suffix('.registered.json')
    if not args.execute:
        write_json(preregister, protocol())
        print('Prepared only; no Queue training or new assessment labels read. Await first-stage release.', flush=True)
        return
    if args.stage1_report is None or not args.stage1_report.is_file():
        raise ValueError('Complete first-stage State work and provide its report before executing Queue')
    if not preregister.is_file() or json.loads(preregister.read_text(encoding='utf-8')) != protocol():
        raise ValueError('Prepare and freeze this exact protocol before execution')
    if args.output.exists():
        raise FileExistsError('Never overwrite Queue experiments')
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    write_json(args.output/'start.json', {'code_commit': commit, 'source_snapshot': snapshot,
        'preregistration_sha256': sha256_file(preregister), 'stage1_report': args.stage1_report.as_posix(),
        'stage1_report_sha256': sha256_file(args.stage1_report), 'scenario_seeds': SEEDS})
    try:
        with threadpool_limits(limits=4):
            cases = [run_case(case, args.output/case) for case in CASES]
            comparisons = {}
            for seed in SEEDS:
                reports = [{'case': r['case'], 'dates': r['dates']['assessment'],
                            'runs': r['assessment'][str(seed)]} for r in cases]
                comparisons[str(seed)] = {arm: paired_comparison(reports, arm) for arm in ARMS[1:]}
        primary = comparisons[str(SEEDS[0])][ARMS[3]]
        result = {'protocol': protocol(), 'code_commit': commit, 'source_snapshot_sha256': snapshot['archive_sha256'],
            'cases': [{k: v for k, v in r.items() if k not in ['selection', 'assessment']} for r in cases],
            'assessment_comparisons': comparisons, 'primary_gate': primary,
            'decision': 'Research gate passed; requires further observation/latent robustness evidence' if primary['research_gate_pass'] else 'Initial conditional event model rejected; stop expansion',
            'seconds': time.perf_counter()-start, 'runtime': evidence(False),
            'formal_Queue': None, 'formal_Total': None, 'uploaded': False}
        write_json(args.output/'experiment.json', result)
        print(json.dumps({'decision': result['decision'], 'gain': primary['paired_macro_gain'],
                          'ci95': primary['ci95'], 'seconds': result['seconds']}), flush=True)
    except Exception as error:
        write_json(args.output/'failure.json', {'exception': repr(error), 'traceback': traceback.format_exc(),
            'seconds': time.perf_counter()-start, 'runtime': evidence(False), 'source_snapshot_sha256': snapshot['archive_sha256']})
        raise


if __name__ == '__main__':
    main()
