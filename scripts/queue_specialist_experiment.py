"""Matched public-train onset-final-step experts and fixed expected-IoU sets."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.experiments import partition_date, raw_for
from trafficflow.queue_events import sample_events
from trafficflow.queue_calibration import RefinedQueueModel, public_onset_timing
from trafficflow.queue_specialist import OnsetSpecialistQueueModel, ExpectedIoUQueueModel
from trafficflow.causal_queue import released_origin_rows
from trafficflow.research_eval import Fold, register_protocol
from trafficflow.runtime import evidence, capture_source_snapshot
from trafficflow.metrics import grouped_iou
from compare_queue_events import comparison


def probability(model, data):
    if isinstance(model, (OnsetSpecialistQueueModel, ExpectedIoUQueueModel)):
        p = model.probability(data.x, data.condition)
    else:
        p = model.probability(data.x)
    return public_onset_timing(p, data.x.lead_minutes, data.condition)


def evaluate(model, data):
    p = probability(model, data)
    prediction = model.classify(data.x, data.window, data.condition)
    frame = pd.DataFrame({'window': data.window, 'condition': data.condition, 'squared': (p-data.y)**2})
    event_brier = frame.groupby(['window', 'condition']).squared.mean().reset_index()
    return {**grouped_iou(data.y, prediction, data.window, data.condition),
        'timing_constrained_observation_proxy_Brier': float(np.mean((p-data.y)**2)),
        'condition_macro_event_Brier': float(event_brier.groupby('condition').squared.mean().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', required=True)
    parser.add_argument('--fold', default='rolling_autumn')
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Never overwrite Queue evidence')
    model_dir = Path('artifacts/research') / args.output.stem
    if model_dir.exists(): raise FileExistsError('Never overwrite research models')
    cfg = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())['protocol']
    original = Fold(**next(f for f in cfg['folds'] if f['name'] == args.fold))
    selection_end = (pd.Timestamp(original.fit_end)+pd.Timedelta(days=45)).date().isoformat()
    assessment_end = (pd.Timestamp(selection_end)+pd.Timedelta(days=30)).date().isoformat()
    fold = Fold(original.name+'_long_cal', original.profile_end, original.fit_end, selection_end, assessment_end)
    registered = register_protocol(args.output.with_suffix('.registered.json'), {
        'id': 'Queue_onset_final_step_specialist', 'panel': args.panel, 'fold': fold.__dict__,
        'fit_seed': args.seed, 'fit_per_condition': 500, 'fit_spacing_minutes': 30,
        'selection_seed': 411, 'assessment_seeds': cfg['simulator_assessment_seeds'],
        'selection_assessment_per_condition': 30, 'evaluation_spacing_minutes': 360,
        'model': 'LGB 600 iterations / 63 leaves; same causal features and public-train events',
        'candidates': ['shared_threshold', 'experts_threshold', 'shared_expected_IoU', 'experts_expected_IoU'],
        'hypotheses': ['Removing known-empty early onset cells from expert fitting improves final-step location recovery.',
            'Independent-Bernoulli expected IoU can choose a better predicted set size than fixed probability thresholds.'],
        'onset_timing_source': 'https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350',
        'inputs': 'Only legal history and released T rows; provided condition routes experts; no >T numeric inputs',
        'label_source': 'Noisy eligible public-train observations; not underlying queue truth',
        'selection': 'Calibration only; freeze before assessment loading. No assessment-driven model choice.',
        'promotion': 'Require >=.01 origin-macro gain and positive day-block CI in at least two prespecified folds/directions; no assessed panel loss >.01; multiple fit seeds required.',
        'final_test': 'These public-train assessment periods have already been inspected for other architectures.',
        'formal_Queue': None, 'official_Physics': None})
    model_dir.mkdir(parents=True)
    snapshot = capture_source_snapshot(model_dir/'source_snapshot.zip')
    start = time.perf_counter()
    release = Release('data/kaggle_public')
    paths = release.files(args.panel, 'train')
    dates = fold.assign([partition_date(release, p) for p in paths])
    parts = {key: [p for p in paths if partition_date(release, p) in days] for key, days in dates.items()}
    net, topo = release.network(args.panel), release.topology(args.panel)
    cache = {}
    def origin_reader(origin):
        token = str(origin)
        if token not in cache: cache[token] = released_origin_rows(release, args.panel, 'train', origin)
        return cache[token]
    def sample(part, seed, quota, spacing):
        return sample_events(raw_for(release, args.panel, parts[part]), net, topo, args.panel,
            seed, per_condition=quota, spacing_minutes=spacing, origin_reader=origin_reader,
            onset_last_step=True)
    fit, fit_meta = sample('fit', args.seed, 500, 30)
    cal, cal_meta = sample('selection', 411, 30, 360)
    models = {}
    for name, cls in [('shared_threshold', RefinedQueueModel), ('experts_threshold', OnsetSpecialistQueueModel)]:
        model = cls(iterations=600, num_leaves=63, seed=args.seed).fit(fit)
        model.uses_origin_timestamp = True
        model.latest_valid_queue_bookkeeping = True
        model.uses_occupancy_features = False
        model.uses_published_condition = False
        model.uses_public_onset_timing = True
        models[name] = model
    models['shared_expected_IoU'] = ExpectedIoUQueueModel(models['shared_threshold'])
    models['experts_expected_IoU'] = ExpectedIoUQueueModel(models['experts_threshold'])
    result = {'registered': registered, 'source_snapshot': snapshot, 'dates': dates,
        'samplers': {'fit': fit_meta, 'selection': cal_meta}, 'selection': {}, 'assessment': [],
        'expert_training_counts': models['experts_threshold'].training_counts,
        'formal_Queue': None, 'limitations': 'No latent truth. Approximate published sampling with noisy observations; duplicate origins across seeds are not independent.'}
    for name, model in models.items():
        tuning = model.tune(cal)
        result['selection'][name] = {'metrics': evaluate(model, cal), 'tuning': tuning}
        model.save(model_dir/(name+'.pkl'))
    result['selected'] = max(models, key=lambda n: result['selection'][n]['metrics']['proxy_IoU'])
    args.output.with_suffix('.selection.json').write_text(json.dumps({
        'protocol_sha256': registered['protocol_sha256'], 'selected': result['selected'],
        'selection': result['selection'], 'note': 'Sealed before assessment loading'}, indent=2), encoding='utf-8')
    for seed in cfg['simulator_assessment_seeds']:
        data, meta = sample('assessment', seed, 30, 360)
        result['assessment'].append({'seed': seed, 'sampler': meta,
            'models': {n: evaluate(m, data) for n, m in models.items()}})
    result['comparisons'] = {n: comparison(result, 'shared_threshold', n) for n in models if n != 'shared_threshold'}
    result['selected_comparison'] = comparison(result, 'shared_threshold', result['selected'])
    result['seconds'] = time.perf_counter()-start
    result['source'] = evidence()
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    (model_dir/'experiment.json').write_text(json.dumps({'completed_report': args.output.as_posix(),
        'seconds': result['seconds'], 'selected': result['selected'], 'protocol_sha256': registered['protocol_sha256']}, indent=2), encoding='utf-8')
    print(args.panel, args.fold, 'selected', result['selected'], 'seconds', result['seconds'], flush=True)
    print(result['selected_comparison'], flush=True)


if __name__ == '__main__': main()
