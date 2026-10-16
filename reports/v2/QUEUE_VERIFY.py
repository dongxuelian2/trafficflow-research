"""Independent scalar readback of frozen V2 Queue research predictions."""
import importlib.util
import json
from pathlib import Path
import pickle
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from trafficflow.runtime import sha256_file, evidence

ROOT = Path('reports/v2/QUEUE_CONDITIONAL_RUN_V2')
OUTPUT = Path('reports/v2/QUEUE_VERIFICATION.json')


def run():
    started = time.perf_counter()
    main = json.loads((ROOT/'experiment.json').read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('v2_runner_verify', Path('scripts/v2/run_queue_experiments.py'))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    scores = defaultdict(dict)
    counts = defaultdict(int)
    maximum_score_error = 0.
    maximum_probability_error = 0.
    frozen_source = main['source_snapshot_sha256']
    assert sha256_file(ROOT/'source_snapshot.zip') == frozen_source
    for case in main['cases']:
        name = case['case']
        folder = ROOT/name
        original = json.loads((Path('artifacts/research')/name/'experiment.json').read_text(encoding='utf-8'))
        report = json.loads((folder/'case_completed.json').read_text(encoding='utf-8'))
        assert sha256_file(folder/'model.pkl') == case['model_sha256']
        assert sha256_file(Path('artifacts/research')/name/'experiment.json') == case['source_experiment_sha256']
        model = pickle.loads((folder/'model.pkl').read_bytes())
        calibration = json.loads((folder/'calibration_sealed.json').read_text(encoding='utf-8'))
        checkpoint = json.loads((folder/'checkpoint_sealed.json').read_text(encoding='utf-8'))
        assert checkpoint['model_sha256'] == calibration['model_sha256'] == case['model_sha256']
        assert checkpoint['selected_neighbors'] == min([8, 24], key=lambda k: (checkpoint['checkpoint_Brier'][str(k)], k != 24))
        expected_phases = ['fit', 'checkpoint', 'decision']+['selection']*3+['assessment']*3
        assert [r['phase'] for r in case['phase_reads']] == expected_phases
        for record in case['phase_reads']:
            assert sha256_file(record['path']) == record['sha256']
            if record['phase'] in ['decision', 'selection', 'assessment']:
                seal = {'decision': 'checkpoint_sealed.json', 'selection': 'calibration_sealed.json',
                        'assessment': 'selection_sealed.json'}[record['phase']]
                assert sha256_file(folder/seal) == record['prior_seal_sha256']
            counts['hashed_phase_caches'] += 1
        source_decision = json.loads((Path('artifacts/research')/name/'decision_sealed.json').read_text(encoding='utf-8'))
        incumbent = source_decision['incumbent']
        old_choice = source_decision['choices'][incumbent]
        assert old_choice['setting'] == calibration['old_decoder']
        assert sha256_file(Path('artifacts/research')/name/('tree_'+old_choice['baseline_arm']+'.pkl')) == case['frozen_tree_sha256']
        tree = runner.RefinedQueueModel.load(Path('artifacts/research')/name/('tree_'+old_choice['baseline_arm']+'.pkl'))
        for phase in ['selection', 'assessment']:
            for event_seed in [2909, 3253, 4001]:
                record = next(r for r in original['cache_records'] if r['phase'] == phase and r['seed'] == event_seed)
                arrays = dict(np.load(record['path'], allow_pickle=False))
                count = len(arrays['origins'])
                width = arrays['labels'][0].size
                windows = arrays['window'].reshape(count, width)[:, 0]
                for seed in runner.SEEDS:
                    archive = np.load(folder/(phase+'_'+str(event_seed)+'_'+str(seed)+'_predictions.npz'))
                    run_report = next(r for r in report[phase][str(seed)] if r['event_seed'] == event_seed)
                    assert np.array_equal(archive['origins'], arrays['origins'])
                    assert archive['old_probability'].shape == archive['event_probability'].shape == arrays['labels'].shape
                    for arm in runner.ARMS:
                        pred = archive[arm]
                        assert pred.shape == arrays['labels'].shape
                        assert np.isin(pred, [0, 1]).all()
                        assert not pred[~arrays['ongoing'], :5].any()
                        saved = {r['window']: r for r in run_report['models'][arm]['windows']}
                        condition_values = defaultdict(list)
                        for i, window in enumerate(windows):
                            valid = arrays['available'][i].ravel()
                            predicted = set(np.flatnonzero(pred[i].ravel() & valid))
                            truth = set(np.flatnonzero(arrays['labels'][i].ravel().astype(bool) & valid))
                            value = len(predicted & truth)/len(predicted | truth) if predicted | truth else (1. if valid.any() else 0.)
                            maximum_score_error = max(maximum_score_error, abs(saved[window]['IoU']-value))
                            assert saved[window]['IoU'] == value
                            condition = 'queue_ongoing' if arrays['ongoing'][i] else 'queue_onset'
                            condition_values[condition].append(value)
                            origin = str(window).rsplit(':', 1)[0]
                            key = (name, phase, seed, arm)
                            if origin in scores[key]:
                                assert scores[key][origin] == (condition, value)
                            scores[key][origin] = (condition, value)
                            counts['scalar_window_arm_checks'] += 1
                        macro = np.mean([np.mean(v) for v in condition_values.values()])
                        maximum_score_error = max(maximum_score_error, abs(run_report['models'][arm]['proxy_IoU']-macro))
                        if phase == 'assessment' and arm == runner.ARMS[0]:
                            source_run = next(r for r in original['assessment'] if r['seed'] == event_seed)
                            assert source_run['models'][incumbent]['windows'] == run_report['models'][arm]['windows']
                            counts['original_strong_baseline_exact_matches'] += 1
                    # Recompute every frozen old probability with its original fit tree.
                    feature_record = next(r for r in record['feature_files'] if r['path'].endswith('_'+old_choice['baseline_arm']+'.parquet'))
                    assert sha256_file(feature_record['path']) == feature_record['sha256']
                    expected_old = tree.probability(pd.read_parquet(feature_record['path'])).reshape(arrays['labels'].shape)
                    assert np.array_equal(expected_old, archive['old_probability'])
                    counts['full_tree_probability_recomputations'] += 1
                    # Independent query batching and reversed order retain same actions.
                    chosen = np.array(sorted(set([0, count-1, int(np.flatnonzero(arrays['ongoing'])[0])])), dtype=int)[::-1]
                    metadata = {k: arrays[k].reshape(count, width)[chosen].ravel() for k in ['window', 'condition', 'lead_minutes']}
                    fresh, probabilities, _ = runner.predict_four(model, expected_old[chosen], arrays['x'][chosen],
                        arrays['origins'][chosen], arrays['ongoing'][chosen], calibration['old_decoder'],
                        calibration['event_old_decoder'], metadata, seed)
                    maximum_probability_error = max(maximum_probability_error,
                        float(np.max(np.abs(probabilities-archive['event_probability'][chosen]))))
                    assert np.allclose(probabilities, archive['event_probability'][chosen], atol=1e-12, rtol=0)
                    for arm in runner.ARMS:
                        assert np.array_equal(fresh[arm], archive[arm][chosen])
                        counts['reordered_window_arm_reproductions'] += len(chosen)
                    assert np.allclose(archive['bottleneck_probability'].sum(axis=1), 1.)
                    assert np.allclose(archive['direction_probability_upstream_stationary_downstream_unknown'].sum(axis=1), 1.)
                    assert np.allclose(archive['direction_probability_upstream_stationary_downstream_unknown'][~arrays['ongoing'], 3], 1.)
                    counts['prediction_archives'] += 1
    # Fresh calendar sampling and scalar-score based pooled gains/intervals.
    comparisons, case_scores = {}, {}
    for seed in runner.SEEDS:
        comparisons[str(seed)] = {}
        for arm in runner.ARMS:
            rng = np.random.default_rng(20261016)
            calendars, case_draws, case_gains = {}, [], []
            for case in main['cases']:
                name = case['case']
                dates = tuple(case['dates']['assessment'])
                if dates not in calendars:
                    n = len(dates)
                    start = rng.integers(0, n, (4000, int(np.ceil(n/7))))
                    calendars[dates] = np.array([[(int(s)+offset) % n for s in starts for offset in range(7)][:n] for starts in start])
                left = scores[(name, 'assessment', seed, runner.ARMS[0])]
                right = scores[(name, 'assessment', seed, arm)]
                assert left.keys() == right.keys()
                sums, numbers = np.zeros((len(dates), 2)), np.zeros((len(dates), 2))
                categories, left_categories = defaultdict(list), defaultdict(list)
                for origin in left:
                    c, y = right[origin]
                    c0, x = left[origin]
                    assert c == c0
                    day = origin.split(':', 1)[1][:10]
                    d, j = dates.index(day), int(c == 'queue_ongoing')
                    sums[d, j] += y-x
                    numbers[d, j] += 1
                    categories[c].append(y)
                    left_categories[c].append(x)
                macro = np.mean([np.mean(v) for v in categories.values()])
                base = np.mean([np.mean(v) for v in left_categories.values()])
                case_scores.setdefault(str(seed), {}).setdefault(name, {})[arm] = float(macro)
                case_gains.append(macro-base)
                index = calendars[dates]
                numerator, denominator = sums[index].sum(axis=1), numbers[index].sum(axis=1)
                sample = (numerator/denominator.clip(min=1)).mean(axis=1)
                sample[(denominator == 0).any(axis=1)] = np.nan
                case_draws.append(sample)
            samples = np.mean(np.stack(case_draws), axis=0)
            samples = samples[np.isfinite(samples)]
            gain, interval = float(np.mean(case_gains)), np.quantile(samples, [.025, .975]).tolist()
            comparisons[str(seed)][arm] = {'gain': gain, 'ci95': interval, 'valid_resamples': len(samples)}
            if arm != runner.ARMS[0]:
                saved = main['assessment_comparisons'][str(seed)][arm]
                assert abs(saved['paired_macro_gain']-gain) < 1e-12
                assert np.allclose(saved['ci95'], interval, atol=1e-12, rtol=0)
                assert saved['valid_resamples'] == len(samples)
    result = {'verified': True, 'source_experiment_sha256': sha256_file(ROOT/'experiment.json'),
        'counts': dict(counts), 'maximum_scalar_score_error': maximum_score_error,
        'maximum_reordered_probability_error': maximum_probability_error,
        'independent_case_condition_macro_IoU': case_scores, 'independent_comparisons': comparisons,
        'seconds': time.perf_counter()-started, 'runtime': evidence(False),
        'limitations': 'Reuses preserved raw history audit and source caches; does not reconstruct all raw observations. Scalar set scores and synchronized date intervals independently recomputed. No latent or formal Queue measurement.'}
    if OUTPUT.exists():
        raise FileExistsError('Do not overwrite verification evidence')
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'verified': True, 'counts': result['counts'], 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    run()
