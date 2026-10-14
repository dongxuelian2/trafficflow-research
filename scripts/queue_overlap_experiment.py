"""Matched new-seed BCE versus window-overlap losses, full forecast decisions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.graph_queue import CausalGraphQueue
from trafficflow.queue_overlap_loss import queue_overlap_training_loss
from trafficflow.probability_calibration import weighted_bce_probability
from trafficflow.full_queue_calibration import tune_full_probabilities, full_queue_decisions
from trafficflow.full_domain_queue import FullDomainQueueModel
from trafficflow.queue import QueueModel
from trafficflow.queue_domain import score_full_forecast
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import register_protocol
from compare_queue_events import comparison


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_npz(path, digest):
    if sha256_file(path) != digest:
        raise ValueError('Frozen sequence/probability cache changed: ' + str(path))
    return dict(np.load(path, allow_pickle=False))


def predict(model, arrays, graph, device):
    out = []
    model.eval()
    with torch.no_grad():
        for x in arrays['x']:
            logits, _ = model(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(device), *graph)
            out.append(weighted_bce_probability(logits.cpu().numpy()[0], 4.))
    return np.stack(out)


def brier(p, arrays):
    domain = arrays['available'].copy()
    domain[:, :5] &= arrays['ongoing'][:, None, None]
    per_event = ((p-arrays['labels'])**2 * domain).sum(axis=(1, 2))/domain.sum(axis=(1, 2)).clip(min=1)
    return float(np.mean([per_event[arrays['ongoing'] == value].mean() for value in [False, True]]))


def decision_metrics(p, cache, setting):
    predicted = full_queue_decisions(p, cache['window'], cache['condition'], cache['lead_minutes'],
        setting['thresholds'], setting['require_nonempty'], setting['timing'])
    return score_full_forecast(cache['labels'], predicted, cache['window'], cache['condition'], cache['score_mask'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-study', required=True)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report_path = Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report_path.exists():
        raise FileExistsError('Never overwrite matched overlap evidence')
    probability_report_path = Path('reports/research/C38_full_queue_probability_cache.json')
    probability_report = load_json(probability_report_path)
    source = next(r for r in probability_report['source_records'] if Path(r['report']).stem == args.source_study)
    sequence_dir = Path(source['causal_cache'])
    sequence_report = load_json(sequence_dir/'experiment.json')
    if sha256_file(sequence_dir/'experiment.json') != source['causal_cache_report_sha256']:
        raise ValueError('Frozen sequence producer changed')
    records = [r for r in probability_report['full_probability_caches'] if r['study'] == args.source_study]
    old_full_report_path = Path('reports/research/C39_full_domain_Queue_calibration.json')
    old_full = next(r for r in load_json(old_full_report_path)['studies'] if r['source_study'] == args.source_study)
    config = sequence_report['registered']['protocol']
    registered = register_protocol(report_path.with_suffix('.registered.json'), {
        'id': 'Matched_Queue_window_overlap_new_training_seed', 'panel': config['panel'],
        'source_study': args.source_study, 'source_probability_report_sha256': sha256_file(probability_report_path),
        'source_sequence_report_sha256': sha256_file(sequence_dir/'experiment.json'),
        'source_full_calibration_report_sha256': sha256_file(old_full_report_path),
        'training_seed': args.seed, 'fixed_event_sampler_seed': config['seed'],
        'variants': ['bce', 'soft_iou', 'lovasz'], 'overlap_strength': .25,
        'model': 'Same 30-minute public-geometry CausalGraphQueue,hidden32,identical inputs/init/batches/budget',
        'training': '20 epochs x80 steps,batch4,AdamW .001,wd1e-4,clip1; equal condition sampling',
        'loss': 'Existing pos_weight4 equal-window BCE+.1 speed MAE; overlap arms add .25 window surrogate on logit-log4',
        'checkpoint': 'Minimum cal condition-macro event Brier every5 epochs, unchanged across arms',
        'decision': 'Full public forecast domain, onset timing before optional force; cal-only mixture alpha0/.25/.5/.75/1 and threshold selection',
        'known_loss_limit': 'Sigmoid(logit-log4) is exact weighted-BCE correction only at its unconstrained optimum; auxiliary IoU breaks this interpretation. Treat as a calibrated score, not posterior.',
        'Lovasz_reference': 'https://arxiv.org/html/1705.08790; independent implementation of set-extension equations',
        'assessment': 'Old public TRAIN proxy dates; load after all model/decision choices sealed. Not final.',
        'formal_Queue': None, 'formal_Total': None, 'uploaded': False})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    fit = load_npz(sequence_dir/('fit_'+str(config['seed'])+'.npz'), sequence_report['cache_sha256']['fit'])
    cal = load_npz(sequence_dir/'selection_411.npz', sequence_report['cache_sha256']['selection'])
    cal_record = next(r for r in records if r['phase'] == 'selection')
    full_cal = load_npz(cal_record['path'], cal_record['sha256'])
    for key, source_key in [('labels', 'labels'), ('available', 'score_mask')]:
        if not np.array_equal(cal[key].ravel(), full_cal[source_key]):
            raise ValueError('Full-domain calibration label order differs')
    geometry_path = Path('artifacts/research')/args.source_study/'wave_30min_geometry.npz'
    geometry = dict(np.load(geometry_path, allow_pickle=False))
    graph = [torch.tensor(geometry[k], device=device) for k in ['incoming', 'outgoing', 'geometry']]
    groups = {value: np.flatnonzero(fit['ongoing'] == value) for value in [False, True]}
    if any(not len(indices) for indices in groups.values()): raise ValueError('Both fit conditions required')
    tree = QueueModel.load(Path('artifacts/research')/Path(config['matched_study']).stem/'shared_threshold.pkl')
    tree_setting = tune_full_probabilities(full_cal['base'], full_cal['window'], full_cal['condition'],
        full_cal['lead_minutes'], full_cal['labels'], full_cal['score_mask'])['selected']
    settings = {'tree': {'alpha': 0., 'setting': tree_setting}}
    result = {'registered': registered, 'source_snapshot': snapshot, 'dates': sequence_report['dates'],
        'training': {}, 'selection': {'tree': {'metrics': decision_metrics(full_cal['base'], full_cal, tree_setting)}},
        'assessment': [], 'source_geometry_sha256': sha256_file(geometry_path), 'exported_models': {}}
    for kind in ['bce', 'soft_iou', 'lovasz']:
        torch.manual_seed(args.seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)
        rng = np.random.default_rng(args.seed)
        model = CausalGraphQueue(hidden=32, use_graph=True).to(device)
        init_digest = hashlib.sha256()
        for key, tensor in model.state_dict().items():
            init_digest.update(key.encode()); init_digest.update(tensor.detach().cpu().numpy().tobytes())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        samples = hashlib.sha256(); best = np.inf; history = []
        for epoch in range(20):
            model.train(); losses = []
            for step in range(80):
                indices = np.array([rng.choice(groups[bool(rng.integers(2))]) for _ in range(4)], dtype=np.int64)
                samples.update(indices.tobytes())
                x = torch.from_numpy(fit['x'][indices].transpose(0, 3, 1, 2).copy()).to(device)
                targets = [torch.from_numpy(fit[k][indices]).to(device) for k in ['labels', 'truth_speed', 'available', 'ongoing']]
                optimizer.zero_grad(set_to_none=True)
                logits, speed = model(x, *graph)
                loss = queue_overlap_training_loss(logits, speed, *targets, kind=kind, strength=.25)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite overlap loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if (epoch+1) % 5 == 0:
                value = brier(predict(model, cal, graph, device), cal)
                history.append({'epoch': epoch+1, 'training_loss': float(np.mean(losses)), 'cal_macro_event_Brier': value})
                if value < best:
                    best = value; torch.save(model.state_dict(), args.output/(kind+'.pt'))
                print(args.output.name, kind, epoch+1, 'Brier', value, flush=True)
        model.load_state_dict(torch.load(args.output/(kind+'.pt'), map_location=device, weights_only=True))
        p = predict(model, cal, graph, device).ravel(); choices = []
        for alpha in [0., .25, .5, .75, 1.]:
            mixed = (1-alpha)*full_cal['base']+alpha*p
            tuning = {'selected': tree_setting} if alpha == 0 else tune_full_probabilities(mixed, full_cal['window'],
                full_cal['condition'], full_cal['lead_minutes'], full_cal['labels'], full_cal['score_mask'])
            setting = tuning['selected']
            choices.append({'alpha': alpha, 'setting': setting, 'metrics': decision_metrics(mixed, full_cal, setting)})
        choice = max(choices, key=lambda row: (row['metrics']['proxy_IoU'], -row['alpha']))
        settings[kind] = {'alpha': choice['alpha'], 'setting': choice['setting']}
        result['selection'][kind] = {'metrics': choice['metrics'], 'choices': choices, 'score_Brier': brier(p.reshape(cal['labels'].shape), cal)}
        state = torch.load(args.output/(kind+'.pt'), map_location='cpu', weights_only=True)
        portable = FullDomainQueueModel(tree, full_cal['links'].tolist(), choice['setting'], choice['alpha'], state, geometry)
        model_path = args.output/(kind+'.pkl'); portable.save(model_path)
        result['exported_models'][kind] = {'path': model_path.as_posix(), 'sha256': sha256_file(model_path)}
        np.savez_compressed(args.output/(kind+'_cal_probabilities.npz'), probability=p)
        result['training'][kind] = {'history': history, 'parameters': sum(p.numel() for p in model.parameters()),
            'initial_tensor_sha256': init_digest.hexdigest(), 'batch_sequence_sha256': samples.hexdigest(),
            'checkpoint_sha256': sha256_file(args.output/(kind+'.pt'))}
        del model, optimizer
        print(kind, 'cal selection', choice['alpha'], choice['setting'], flush=True)
    if len({r['initial_tensor_sha256'] for r in result['training'].values()}) != 1 or len({
        r['batch_sequence_sha256'] for r in result['training'].values()}) != 1:
        raise ValueError('Matched initialization/sample controls failed')
    selected = max(settings, key=lambda name: (result['selection'][name]['metrics']['proxy_IoU'], name == 'tree', name == 'bce'))
    result['selected'] = selected
    seal_path = args.output/'selection_sealed.json'
    seal_path.write_text(json.dumps({'selected': selected, 'settings': settings,
        'checkpoints': {k: v['checkpoint_sha256'] for k, v in result['training'].items()}}, indent=2, allow_nan=False), encoding='utf-8')
    result['selection_seal_sha256'] = sha256_file(seal_path)
    old_setting = old_full['selected']['before']['tuning']['selected']
    old_alpha = old_full['selected']['before']['alpha']
    for record in records:
        if record['phase'] != 'assessment': continue
        prior = next(r for r in sequence_report['assessment'] if r['seed'] == record['seed'])
        arrays = load_npz(sequence_dir/('assessment_'+str(record['seed'])+'.npz'), prior['cache_sha256'])
        full = load_npz(record['path'], record['sha256'])
        if not np.array_equal(arrays['labels'].ravel(), full['labels']) or not np.array_equal(arrays['available'].ravel(), full['score_mask']):
            raise ValueError('Assessment full-domain cache order differs')
        metrics = {'tree': decision_metrics(full['base'], full, tree_setting),
            'previous_seed_mixture': decision_metrics((1-old_alpha)*full['base']+old_alpha*full['neural'][:, 2], full, old_setting)}
        for kind in ['bce', 'soft_iou', 'lovasz']:
            model = CausalGraphQueue(hidden=32, use_graph=True).to(device)
            model.load_state_dict(torch.load(args.output/(kind+'.pt'), map_location=device, weights_only=True))
            p = predict(model, arrays, graph, device).ravel(); choice = settings[kind]
            metrics[kind] = decision_metrics((1-choice['alpha'])*full['base']+choice['alpha']*p, full, choice['setting'])
            np.savez_compressed(args.output/(kind+'_assessment_'+str(record['seed'])+'_probabilities.npz'), probability=p)
            del model
        result['assessment'].append({'seed': record['seed'], 'models': metrics})
    result['comparisons'] = {name: comparison(result, 'bce', name) for name in ['soft_iou', 'lovasz']}
    result['selected_vs_tree'] = comparison(result, 'tree', selected)
    result['selected_vs_previous_seed'] = comparison(result, 'previous_seed_mixture', selected)
    result.update(seconds=time.perf_counter()-start, source=evidence(gpu_used=device.type == 'cuda'),
        gpu_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
        matched_initialization_and_batches=True, formal_Queue=None, formal_Total=None, uploaded=False,
        limitations='Fixed noisy positive-event sampler and previously inspected TRAIN development dates. New optimizer seed but same fit events. Scores are not reliable posteriors under overlap objectives. No promotion without additional seasons/directions/seeds.')
    text = json.dumps(result, indent=2, allow_nan=False)
    (args.output/'experiment.json').write_text(text, encoding='utf-8'); report_path.write_text(text, encoding='utf-8')
    print('complete', args.output.name, 'selected', selected, 'loss comparisons', result['comparisons'], 'seconds', result['seconds'], flush=True)


if __name__ == '__main__': main()
