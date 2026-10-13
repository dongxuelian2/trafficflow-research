"""Archive bounds for the completed timing-only control, without queue truth."""
import json
from pathlib import Path
from trafficflow.data import Release, csv


if __name__ == '__main__':
    output = Path('reports/research/K02_public_timing_decision.json')
    if output.exists():
        raise FileExistsError('Never overwrite controlled experiment evidence')
    root = Path('artifacts/research/K02_Queue_only_public_timing')
    diff = json.loads((root / 'controlled_diff.json').read_text())
    audit = json.loads(Path('reports/research/C23_released_onset_timing_audit.json').read_text())
    if not diff['full_row_diff_passed'] or diff['changed_rows'] != {'queue': 5}:
        raise ValueError('The independently checked change set is required')
    release = Release('data/kaggle_public')
    by_split = {}
    for split in ('validation', 'private'):
        indices = {p: csv(release.root / 'task2' / p / split / 'window_index.csv')
                   for p in release.panels() if not p.startswith('D12_I405')}
        families = {p.rsplit('_', 1)[0] for p in indices}
        bounds = []
        for row in audit['own_forecasts']:
            if row['split'] != split or not row['early_positive_predictions']:
                continue
            panel = row['panel']
            index = indices[panel]
            family = panel.rsplit('_', 1)[0]
            directions = sum(p.rsplit('_', 1)[0] == family for p in indices)
            conditions = index.condition.nunique()
            windows = int((index.condition == 'queue_onset').sum())
            weight = 1 / len(families) / directions / conditions / windows
            early, final = row['early_positive_predictions'], row['final_positive_predictions']
            bounds.append({**row, 'official_Queue_window_weight': weight,
                'unconditional_Total_gain_upper_bound': .30 * weight,
                'all_predicted_cells_scored_conditional_Total_gain_upper_bound':
                    .30 * weight * early / (early + final)})
        by_split[split] = {'changed_windows': bounds,
            'unconditional_Total_gain_upper_bound': sum(r['unconditional_Total_gain_upper_bound'] for r in bounds),
            'all_predicted_cells_scored_conditional_Total_gain_upper_bound': sum(
                r['all_predicted_cells_scored_conditional_Total_gain_upper_bound'] for r in bounds)}
    result = {'candidate_sha256': diff['candidate_sha256'],
        'full_row_diff': diff, 'split_bounds': by_split, 'formal_Total': None,
        'formal_Queue': None, 'uploaded': False,
        'decision': 'Keep local. Reserve formal comparisons for candidates with potential private-split improvement.',
        'reason': 'Only two validation windows change. Every private prediction is byte-identical to the frozen formal baseline.',
        'unconditional_derivation': 'Each changed window IoU can increase by at most 1; multiply by the published family/direction/condition/window weights and Queue weight .30.',
        'conditional_derivation': 'If every old predicted cell is scored, deleting e known false positives from f final predictions improves IoU by at most e/(f+e).',
        'eligibility_caveat': 'Scoring eligibility in withheld queue targets is unknown. The conditional .001369 bound is not an unconditional bound; the safe validation bound is .0075.',
        'no_hidden_labels_or_future_inputs': True,
        'public_source': 'https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350',
        'scoring_source': 'vendor/official/src/task2/score_task2.py'}
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v['unconditional_Total_gain_upper_bound'] for k, v in by_split.items()})
