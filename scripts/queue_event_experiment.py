"""Controlled joint-vs-event Queue comparison under the new public sampler."""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
from trafficflow.data import Release
from trafficflow.experiments import partition_date, raw_for
from trafficflow.research_eval import Fold
from trafficflow.queue import QueueModel, proxy_windows
from trafficflow.queue_events import EventQueueModel, sample_events
from trafficflow.metrics import grouped_iou
from trafficflow.runtime import evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', default='D12_I5_N')
    parser.add_argument('--fold', default='rolling_autumn')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Experiment evidence must not be overwritten')
    start = time.perf_counter()
    protocol = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())
    cfg = protocol['protocol']
    fold = Fold(**next(f for f in cfg['folds'] if f['name']==args.fold))
    release = Release('data/kaggle_public')
    paths = release.files(args.panel, 'train')
    dates = fold.assign([partition_date(release, path) for path in paths])
    parts = {key:[p for p in paths if partition_date(release, p) in days] for key, days in dates.items()}
    net, topo = release.network(args.panel), release.topology(args.panel)
    train, train_meta = sample_events(raw_for(release, args.panel, parts['fit']), net, topo, args.panel,
                                    cfg['fit_seeds'][0], per_condition=80)
    old_train = proxy_windows(raw_for(release, args.panel, parts['fit']), net, topo, args.panel)
    cal, cal_meta = sample_events(raw_for(release, args.panel, parts['selection']), net, topo, args.panel,
                                 cfg['simulator_development_seeds'][0], per_condition=5)
    models = {'old_sampler_joint': QueueModel('lightgbm', iterations=400).fit(old_train),
              'random_sampler_joint': QueueModel('lightgbm', iterations=400).fit(train)}
    for model in models.values():
        model.tune(cal)
    event = EventQueueModel().fit(train).tune(cal)
    result = {'experiment': 'C01_event_queue', 'panel': args.panel, 'fold': args.fold,
        'protocol_sha256': protocol['protocol_sha256'], 'source_at_start': evidence(), 'dates': dates,
        'formal_Queue': None, 'sampler': train_meta, 'selection_sampler': cal_meta,
        'old_training_windows': int(len(np.unique(old_train.window))), 'assessment': []}
    for seed in cfg['simulator_assessment_seeds']:
        test, metadata = sample_events(raw_for(release, args.panel, parts['assessment']), net, topo,
                                      args.panel, seed, per_condition=5)
        row = {'seed': seed, 'sampler': metadata, 'models': {}}
        for name, model in models.items():
            p = model.probability(test.x)
            row['models'][name] = {**grouped_iou(test.y, p>=model.threshold, test.window, test.condition),
                                  'observation_proxy_Brier': float(np.mean((p-test.y)**2))}
        p = event.probability(test)
        row['models']['separate_event_models'] = {**grouped_iou(test.y, event.classify(test), test.window, test.condition),
                                      'observation_proxy_Brier': float(np.mean((p-test.y)**2))}
        result['assessment'].append(row)
    result['seconds'] = time.perf_counter()-start
    result['limitations'] = 'Same 14 public-train days across seeds; samples overlap and are not independent. No true queue labels. Architecture comparison is development evidence.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({name:np.mean([r['models'][name]['proxy_IoU'] for r in result['assessment']])
           for name in list(models)+['separate_event_models']}, flush=True)


if __name__ == '__main__':
    main()
