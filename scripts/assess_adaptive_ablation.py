"""Assess every previously calibrated architecture without changing selection.

This is a retrospective development ablation, not an untouched final test.
Each checkpoint and mixture comes from the original calibration-only record.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.adaptive_graph_state import AdaptiveGraphState, reconstruction_input
from trafficflow.data import Release, VALUES
from trafficflow.experiments import partition_date, day_template
from trafficflow.state import StateModel, StateFeatures
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import register_protocol, paired_block_interval
from adaptive_graph_experiment import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    output = args.directory/'matched_ablation'
    if output.exists():
        raise FileExistsError('Never overwrite a matched ablation')
    original = json.loads((args.directory/'experiment.json').read_text())
    config = original['registered']['protocol']
    if original['selected'] != json.loads((args.directory/'selection_sealed.json').read_text()):
        raise ValueError('Original calibration selection changed')
    choices = {}
    for name, row in original['selection'].items():
        path = args.directory/(name+'.pt')
        expected = original['model_sha256'][path.name]
        if sha256_file(path) != expected:
            raise ValueError('Calibrated checkpoint changed')
        choices[name] = {'alpha': row['selected_alpha'], 'checkpoint_sha256': expected}
    baseline_path = Path(config['baseline_file'])
    if sha256_file(baseline_path) != config['baseline_sha256']:
        raise ValueError('Strong comparator changed')
    report = Path('reports/research')/(args.directory.name+'.ablation.json')
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'Calibrated_neural_architecture_development_ablation',
        'original_report_sha256': sha256_file(args.directory/'experiment.json'),
        'panel': config['panel'], 'fold': config['fold'], 'choices': choices,
        'original_selected': original['selected'],
        'selection_policy': 'Keep the original calibration winner. No assessment-driven reselection.',
        'assessment_status': 'Previously inspected public-train development dates',
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
    output.mkdir()
    snapshot = capture_source_snapshot(output/'source_snapshot.zip')
    start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    release = Release('data/kaggle_public')
    panel = config['panel']
    net, topo = release.network(panel), release.topology(panel)
    baseline = StateModel.load(baseline_path)
    cache = Path(config['cache'])
    paths = {partition_date(release, p): p for p in release.files(panel, 'train')}
    template = release.template(1, panel, 'train')
    days = []
    for path in sorted((cache/'assessment').glob('*.npz')):
        day = dict(np.load(path, allow_pickle=False))
        f = StateFeatures(release.read(paths[path.stem]), net, topo,
            baseline.profile, version=baseline.feature_version)
        if not np.array_equal(reconstruction_input(f), day['x']):
            raise ValueError('Audited features differ')
        target = day_template(release, panel, 'train', f.frame, template)
        if not np.array_equal(f.lookup(target), day['target_indices']):
            raise ValueError('Target ordering changed')
        prediction = baseline.predict(f, target)[VALUES].to_numpy(float, copy=True)
        prediction[:, 0] = np.minimum(prediction[:, 0], target.link_id.map(net.free_speed_kmh).to_numpy(float)*1.05)
        prediction[:, 1] = np.minimum(prediction[:, 1], target.link_id.map(net.capacity_vph).to_numpy(float)*1.05)/target.link_id.map(net.lanes).to_numpy(float)
        day['comparator'] = prediction
        days.append(day)
    if not days:
        raise ValueError('Assessment missing')
    graph_np = dict(np.load(cache/'graph.npz'))
    graph = [torch.tensor(graph_np[k], device=device) for k in ('incoming', 'outgoing', 'fd')]
    factor = (net.lanes*net.length_km).reindex(days[0]['links']).to_numpy(np.float32)
    control = evaluate(None, days, graph, device, 0., factor)
    result = {'registered': registered, 'source_snapshot': snapshot, 'baseline': control, 'architectures': {}}
    for name, choice in choices.items():
        model = AdaptiveGraphState(hidden=config['hidden'], heads=2, attention=name != 'graph_geometry').to(device)
        model.load_state_dict(torch.load(args.directory/(name+'.pt'), map_location=device, weights_only=True))
        model.eval()
        metrics = evaluate(model, days, graph, device, choice['alpha'], factor)
        result['architectures'][name] = {'sealed_choice': choice, 'metrics': metrics,
            'paired_daily_vs_strong_LGB': paired_block_interval(
                [r['daily_State'] for r in control['days']], [r['daily_State'] for r in metrics['days']])}
        del model
    geometry = result['architectures']['graph_geometry']['metrics']
    for name, row in result['architectures'].items():
        row['paired_daily_vs_geometry_graph'] = paired_block_interval(
            [r['daily_State'] for r in geometry['days']], [r['daily_State'] for r in row['metrics']['days']])
    selected = result['architectures'][original['selected']['name']]['metrics']
    if abs(selected['S_state']-original['selected_assessment']['S_state']) > 1e-7:
        raise ValueError('Original selected assessment did not reproduce')
    result.update(seconds=time.perf_counter()-start, source=evidence(gpu_used=device.type == 'cuda'),
        original_selection_preserved=True, formal_State=None, formal_Physics=None, formal_Total=None,
        limitations='Calibration-selected mixtures and checkpoints are fixed. Assessments were already seen; these paired ablations diagnose modules and cannot validate a new assessment-selected winner.')
    (output/'experiment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(args.directory.name, {k: v['metrics']['S_state'] for k, v in result['architectures'].items()}, flush=True)
