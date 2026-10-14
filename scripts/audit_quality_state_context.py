"""Any-quality visibility diagnostics and unchanged legacy-model replay."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from trafficflow.data import Release, VALUES
from trafficflow.state import StateModel, StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.experiments import partition_date, day_template
from trafficflow.runtime import sha256_file, evidence
from trafficflow.research_eval import state_from_sufficient_stats
from train_graph_experiment import stats
from diagnose_graph_density import increment_stats
from audit_public_missing_runs import run_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite visibility or compatibility evidence')
    info = json.loads((args.cache/'prepared.json').read_text())
    groups = {}
    for part in ['fit', 'selection', 'assessment']:
        for path in sorted((args.cache/part).glob('*.npz')):
            with np.load(path) as arrays:
                x, mask, regime = arrays['x'], arrays['mask'], str(arrays['regime'].item())
            item = groups.setdefault(regime, {'n_cells': 0, 'target_cells': 0, 'strict_available': 0,
                'any_available': 0, 'extra': 0, 'strict_gap_runs': [], 'any_gap_runs': []})
            strict = np.all(x[..., 2:4] > .5, axis=-1)
            any_quality = np.all(x[..., 14:16] > .5, axis=-1)
            if (strict & ~any_quality).any():
                raise ValueError('Any-quality input lost an original visible cell')
            item['n_cells'] += mask.size
            item['target_cells'] += int(mask.sum())
            item['strict_available'] += int(strict.sum())
            item['any_available'] += int(any_quality.sum())
            item['extra'] += int((any_quality & ~strict).sum())
            for j in range(mask.shape[1]):
                item['strict_gap_runs'].extend(run_lengths(~strict[:, j]).tolist())
                item['any_gap_runs'].extend(run_lengths(~any_quality[:, j]).tolist())
    for item in groups.values():
        for key in ['strict_gap_runs', 'any_gap_runs']:
            values = np.array(item.pop(key))*5
            item[key+'_minutes'] = {'p95': float(np.quantile(values, .95)), 'p99': float(np.quantile(values, .99)), 'maximum': int(values.max())}
        item['strict_available_fraction'] = item['strict_available']/item['n_cells']
        item['any_available_fraction'] = item['any_available']/item['n_cells']
    original = json.loads((args.legacy/'experiment.json').read_text())
    path = args.legacy/'State_selected.pkl'
    digest = sha256_file(path)
    if digest != original['deployable_model_sha256']:
        raise ValueError('Frozen legacy model changed')
    model = StateModel.load(path)
    missing_schema = not hasattr(model, 'input_schema')
    release = Release('data/kaggle_public')
    panel = info['panel']
    net, topology = release.network(panel), release.topology(panel)
    files = {partition_date(release, p): p for p in release.files(panel, 'train')}
    template = release.template(1, panel, 'train')
    rows, inc = [], []
    for source in sorted((args.cache/'selection').glob('*.npz')):
        day = dict(np.load(source, allow_pickle=False))
        f = StateFeatures(release.read(files[source.stem]), net, topology, model.profile, version=model.feature_version)
        if not np.array_equal(reconstruction_input(f), day['x_control']):
            raise ValueError('Old strict feature construction changed')
        target = day_template(release, panel, 'train', f.frame, template)
        p = model.predict(f, target)[VALUES].to_numpy(float, copy=True)
        p[:, 1] /= target.link_id.map(net.lanes).to_numpy(float)
        rows.append(stats(day, p))
        factor = (net.lanes*net.length_km).reindex(day['links']).to_numpy(float)
        day['x'] = day['x_control']
        inc.append(increment_stats(day, p, factor))
    replay = {**state_from_sufficient_stats(rows),
        'density_target_pair_relative_L1': sum(r['error_L1'] for r in inc)/sum(r['truth_increment_L1'] for r in inc)}
    selected = original['selected']
    reference = next(r['metrics'] for r in original['selection'][selected['name']]['ensemble_grid'] if r['alpha'] == selected['alpha'])
    if abs(replay['S_state']-reference['S_state']) > 1e-7 or abs(replay['density_target_pair_relative_L1']-reference['density_target_pair_relative_L1']) > 1e-7:
        raise ValueError('Legacy default-schema predictions changed')
    if sha256_file(path) != digest:
        raise ValueError('Compatibility check modified frozen model bytes')
    result = {'panel': panel, 'visibility_by_regime': groups,
        'legacy_model_sha256': digest, 'legacy_schema_attribute_absent': missing_schema,
        'n_legacy_calibration_days_replayed': len(rows), 'legacy_model_file_unchanged': True,
        'legacy_replay_metrics': replay, 'formal_State': None, 'formal_Physics': None, 'formal_Total': None,
        'source': evidence(gpu_used=bool(torch.cuda.is_available() and model.alpha)),
        'limitations': 'Input availability is not accuracy of low-quality measurements. Only public train cache observed geometry audited; no true latent labels. Legacy replay covers the old cal dates, not retraining or private performance.'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(groups, 'legacy_cal_State', replay['S_state'], 'legacy_model_unchanged', True, flush=True)


if __name__ == '__main__':
    main()
