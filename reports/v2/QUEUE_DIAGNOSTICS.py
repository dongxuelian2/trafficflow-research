"""Post-experiment observed-proxy diagnostics; never model selection criteria."""
import json
from pathlib import Path
import numpy as np
from trafficflow.data import Release

ROOT = Path('reports/v2/QUEUE_CONDITIONAL_RUN_V2')
ARMS = ['old_probability_old_decoder', 'old_probability_new_decoder',
        'event_probability_old_decoder', 'event_probability_new_decoder']


def mean(values):
    return float(np.mean(values)) if values else None


def run():
    release = Release('data/kaggle_public')
    result = {}
    study = json.loads((ROOT/'experiment.json').read_text(encoding='utf-8'))
    for case in study['cases']:
        name = case['case']
        report = json.loads((ROOT/name/'case_completed.json').read_text(encoding='utf-8'))
        topo = release.topology(case['panel'])
        rows = {}
        for event_seed in [2909, 3253, 4001]:
            arrays = dict(np.load(Path('artifacts/research')/name/('assessment_'+str(event_seed)+'.npz')))
            prediction = np.load(ROOT/name/('assessment_'+str(event_seed)+'_20261016_predictions.npz'))
            links = {str(link): i for i, link in enumerate(arrays['links'])}
            edges = [(i, links[v]) for u, i in links.items() if u in topo.index
                     for v in str(topo.loc[u, 'outgoing_link_ids']).split(';') if v in links]
            for i, origin in enumerate(arrays['origins']):
                if origin in rows:
                    continue
                valid, truth = arrays['available'][i], arrays['labels'][i].astype(bool)
                ongoing = bool(arrays['ongoing'][i])
                active = valid.copy()
                if not ongoing:
                    active[:5] = False
                row = {'condition': 'ongoing' if ongoing else 'onset', 'arms': {}}
                row['old_probability_Brier'] = float(np.mean((prediction['old_probability'][i][active]-truth[active])**2))
                row['event_probability_Brier'] = float(np.mean((prediction['event_probability'][i][active]-truth[active])**2))
                edge_valid = np.stack([valid[:, u] & valid[:, v] for u, v in edges], axis=1) if edges else np.zeros((6, 0), bool)
                true_front = np.stack([truth[:, u] != truth[:, v] for u, v in edges], axis=1) if edges else edge_valid
                row['observed_edge_time_pairs'] = int(edge_valid.sum()) if ongoing else 0
                for arm in ARMS:
                    p = prediction[arm][i].astype(bool)
                    intersection = int((p & truth & valid).sum())
                    union = int(((p | truth) & valid).sum())
                    value = intersection/union if union else (1. if valid.any() else 0.)
                    row['arms'][arm] = {'observed_window_IoU': value}
                    if not ongoing:
                        row['arms'][arm]['observed_T30_cardinality_absolute_error'] = int(abs(int((p[-1] & valid[-1]).sum())-int((truth[-1] & valid[-1]).sum())))
                    else:
                        front = np.stack([p[:, u] != p[:, v] for u, v in edges], axis=1) if edges else edge_valid
                        inter = int((front & true_front & edge_valid).sum())
                        union = int(((front | true_front) & edge_valid).sum())
                        row['arms'][arm]['observed_sensor_edge_boundary_IoU'] = inter/union if union else (1. if edge_valid.any() else 0.)
                rows[str(origin)] = row
        grouped = {}
        for condition in ['onset', 'ongoing']:
            selected = [r for r in rows.values() if r['condition'] == condition]
            grouped[condition] = {'n_origins': len(selected),
                'old_probability_Brier': mean([r['old_probability_Brier'] for r in selected]),
                'event_probability_Brier': mean([r['event_probability_Brier'] for r in selected]),
                'arms': {arm: {metric: mean([r['arms'][arm][metric] for r in selected])
                         for metric in selected[0]['arms'][arm]} for arm in ARMS},
                'observed_edge_time_pairs': int(sum(r['observed_edge_time_pairs'] for r in selected))}
        result[name] = grouped
    output = Path('reports/v2/QUEUE_DIAGNOSTICS.json')
    if output.exists():
        raise FileExistsError('Never overwrite diagnostic evidence')
    output.write_text(json.dumps({'posthoc_descriptive_only': True, 'scenario_seed': 20261016,
        'cases': result, 'location_limitation': 'No observed causal bottleneck or complete physical front truth exists. Edge-boundary IoU covers only directly connected observed sensor pairs; topology pair reachability is under2%. Not a physical full-corridor front metric. Cardinality uses scorer-only availability after full predictions exist.'}, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
