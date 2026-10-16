"""Run independent-vs-official contract checks on real public TRAIN partitions."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.runtime import capture_source_snapshot, sha256_file, evidence
from trafficflow.research_eval import register_protocol
from trafficflow.v2 import score_aligned as own


def official(task):
    path = Path(f'vendor/official/src/task{task}/score_task{task}.py')
    spec = importlib.util.spec_from_file_location(f'v2_official_task{task}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equality(a, b, keys, columns):
    a, b = [f.sort_values(keys).reset_index(drop=True) for f in [a, b]]
    pd.testing.assert_frame_equal(a[keys], b[keys], check_dtype=False)
    largest = 0.
    for c in columns:
        av, bv = a[c].to_numpy(float), b[c].to_numpy(float)
        np.testing.assert_allclose(av, bv, rtol=1e-12, atol=1e-9, equal_nan=True)
        finite = np.isfinite(av) & np.isfinite(bv)
        if finite.any():
            largest = max(largest, float(np.abs(av[finite]-bv[finite]).max()))
    return largest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('artifacts/research/v2_score_audit_v1'))
    ap.add_argument('--days-per-regime', type=int, default=2)
    args = ap.parse_args()
    report = Path('reports/v2')/(args.output.name+'.json')
    if args.output.exists() or report.exists():
        raise FileExistsError('Audit evidence is immutable; choose a new output')
    r = Release('data/kaggle_public')
    from task1.baseline_task1_historical_mean import regime_of_dates
    panels = r.panels()
    choices = {}
    for panel in panels:
        counts = {g: 0 for g in own.REGIMES}
        selected = []
        for p in r.files(panel, 'train'):
            # Metadata filename date; inspect only training-era dates, no assessment selection.
            date = p.stem.replace('part-', '')
            import re
            match = re.search(r'(20\d\d)[_-](\d\d)[_-](\d\d)', date)
            date = '-'.join(match.groups()) if match else str(pd.read_parquet(p, columns=['date']).date.iloc[0])
            g = str(regime_of_dates(panel, pd.Series([date]))[0])
            if counts[g] < args.days_per_regime:
                selected.append((p, date, g)); counts[g] += 1
            if min(counts.values()) >= args.days_per_regime:
                break
        choices[panel] = selected
    registered = register_protocol(report.with_suffix('.registered.json'), {
        'id': 'V2_real_train_score_contract', 'days': {p: [(d, g) for _, d, g in v] for p, v in choices.items()},
        'source_commit': json.loads(Path('vendor/official/UPSTREAM.json').read_text())['commit'],
        'prediction': 'Masked-only per-station bidirectional interpolation with medians; no raw labels',
        'scope': 'All 10 directions, all 3 regimes, two earliest public-train dates each; full target sets within dates, not full 273-day calendar',
        'formal_Physics': None, 'hidden_boundary_access': False})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    stage = args.output/'release'
    stage.mkdir()
    task1, task3 = official(1), official(3)
    manifest = json.loads(Path('vendor/official/config/corridors.json').read_text())
    families = {p['corridor_id']: p['family_id'] for p in manifest['panels']}
    modes = json.loads(Path('vendor/official/config/task3_lwr_modes.json').read_text())
    records, state_rows, official_state = [], [], []
    public_sources = []
    for panel, selected in choices.items():
        original = r.root/'corridors'/panel
        folder = stage/'corridors'/panel
        (folder/'network').mkdir(parents=True)
        for source in (original/'network').glob('*.csv'):
            shutil.copy2(source, folder/'network'/source.name)
        dates = {d for _, d, _ in selected}
        template = own.canonical(r.template(1, panel, 'train'))
        template = template[template.timestamp.dt.strftime('%Y-%m-%d').isin(dates)]
        target_path = stage/'task1'/panel/'train/sample_submission_state.csv'
        target_path.parent.mkdir(parents=True)
        template.to_csv(target_path, index=False)
        raw_sources = {p.name: p for p in r.files(panel, 'train', False)}
        all_raw, predictions, masked_frames = [], [], []
        for source, date, g in selected:
            raw_source = raw_sources[source.name]
            for src, dest in [(source, folder/f'train/mainline_states_masked/mask_regime={g}'/source.name),
                              (raw_source, folder/'train/mainline_states'/source.name)]:
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest)
            raw, masked = map(own.canonical, [r.read(raw_source), r.read(source)])
            all_raw.append(raw); masked_frames.append((g, masked, raw))
            pred_grid = masked.sort_values(['station_id', 'timestamp']).copy()
            for c in own.VALUES:
                pred_grid[c] = pred_grid.groupby('station_id')[c].transform(lambda s: s.interpolate(limit_direction='both'))
                pred_grid[c] = pred_grid[c].fillna(pred_grid[c].median())
            target = template[template.timestamp.dt.strftime('%Y-%m-%d') == date]
            pred = target.drop(columns=own.VALUES).merge(pred_grid[own.KEY+own.VALUES], on=own.KEY, validate='one_to_one')
            predictions.append(pred)
            public_sources.extend([{'path': src.as_posix(), 'sha256': sha256_file(src)} for src in [source, raw_source]])
        # Only matching public ramp partitions; their names share the same date spelling.
        for path in (original/'train/ramp_states').rglob('*.parquet'):
            f = pd.read_parquet(path, columns=['timestamp'])
            if set(f.timestamp.astype(str).str[:10]) & dates:
                dest = folder/'train/ramp_states'/path.relative_to(original/'train/ramp_states')
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
                public_sources.append({'path': path.as_posix(), 'sha256': sha256_file(path)})
        raw, prediction = pd.concat(all_raw), pd.concat(predictions)
        params, upstream_params = own.network_parameters(folder), task3.network_parameters(folder)
        parameter_error = equality(params, upstream_params, ['link_id'], [c for c in params if c != 'link_id'])
        ramps, upstream_ramps = own.ramp_flows(folder), task3.load_ramp_flows(folder, 'train')
        ramp_error = equality(ramps, upstream_ramps, ['timestamp', 'link_id'], [c for c in ramps if c not in ['timestamp', 'link_id']])
        topology = pd.read_csv(folder/'network/lwr_mainline_topology.csv', dtype=str).set_index('link_id')
        upstream_rows = task1.score_panel(panel, stage, 'train', prediction)
        official_state.extend([z for z in upstream_rows if z['level'] == 'panel_regime'])
        for g in own.REGIMES:
            group_raw = raw[raw.date.astype(str).isin([d for _, d, rg in selected if rg == g])]
            group_template, group_pred = template[template.mask_regime == g], prediction[prediction.mask_regime == g]
            universe = pd.concat([own.state_targets(rr, mm) for gg, mm, rr in masked_frames if gg == g])
            intersection = universe[own.KEY].merge(group_template[own.KEY], on=own.KEY, how='outer', indicator=True)
            score_stats = own.state_statistics(universe, group_pred, r.station_lanes(panel, universe))
            state_rows.append({'panel': panel, 'regime': g, **score_stats})
            upstream = next(z for z in upstream_rows if z.get('regime') == g)
            this_score = own.state_macro([state_rows[-1]], {panel: families[panel]})['details']
            this_score = next(z for z in this_score if z['regime'] == g)
            assert score_stats['n'] == upstream['n_target_cells'] and score_stats['missing'] == upstream['n_missing_predictions']
            np.testing.assert_allclose(this_score['S_state'], upstream['S_state'], atol=1e-12)
            anchored, missing = own.anchored_frame(group_raw, group_template, group_pred)
            up_anchored, up_missing = task3.state_for_regime(panel, folder, 'train', g, prediction)
            assert missing == up_missing
            anchor_error = equality(anchored, up_anchored, ['timestamp', 'link_id'], own.VALUES+['is_target'])
            frame = own.physics_frame(anchored, panel, g, params, topology, ramps)
            up_frame, _ = task3.derive_physics(panel, stage, 'train', prediction, g, params=upstream_params.set_index('link_id'), ramp_flows=upstream_ramps)
            physcols = sorted(task3.PHYS_COLUMNS-set(['panel', 'timestamp', 'link_id', 'mask_regime']))+['is_target']
            frame_error = equality(frame, up_frame, ['timestamp', 'link_id'], physcols)
            score = own.physics_score(frame, topology, modes['panels'][panel]['mode'])
            up_score = task3.score_panel(panel, up_frame, stage, modes, g)
            for key in score:
                np.testing.assert_allclose(score[key], up_score[key], atol=1e-12)
            records.append({'panel': panel, 'regime': g, 'template_targets': len(group_template), 'official_state_targets': len(universe),
                'state_extra_blanks': int((intersection['_merge'] == 'left_only').sum()), 'template_not_state_targets': int((intersection['_merge'] == 'right_only').sum()),
                'missing_template_predictions': missing, 'state_stats': score_stats,
                'max_parameter_error': parameter_error, 'max_ramp_error': ramp_error, 'max_anchor_error': anchor_error,
                'max_physics_frame_error': frame_error, 'public_topology_fallback': score})
        print(panel, 'all contracts matched', flush=True)
    macro = own.state_macro(state_rows, families)
    official_panels = {p: float(np.mean([z['S_state'] for z in official_state if z['panel'] == p])) for p in panels}
    official_macro = float(np.mean([np.mean([s for p, s in official_panels.items() if families[p] == f]) for f in set(families.values())]))
    np.testing.assert_allclose(macro['S_state'], official_macro, atol=1e-12)
    # Public task2 reducer semantics; these constructed labels are contract tests, not accuracy.
    task2 = official(2)
    empty_target = pd.DataFrame({'is_score_eligible': [False], 'queue_true': [0]})
    assert task2.score_window(empty_target, pd.DataFrame())[0] == 0
    result = {'registered': registered, 'source_snapshot': snapshot, 'source_hashes': public_sources,
              'records': records, 'state_macro': macro, 'official_state_macro': official_macro,
              'state_template_mismatch_total': sum(z['state_extra_blanks']+z['template_not_state_targets'] for z in records),
              'official_sources': {f'task{i}': sha256_file(f'vendor/official/src/task{i}/score_task{i}.py') for i in range(1, 5)},
              'all_numeric_comparisons_passed': True, 'seconds': time.perf_counter()-start, 'runtime': evidence(),
              'formal_Total': None, 'formal_Physics': None,
              'task4': 'Public evaluator computes S_link only (25% of ODME), no hidden paths accessed',
              'aggregation': 'State/Queue/OD family macro; standalone Task3 averages panels directly',
              'boundary_claim': 'load_boundary_flux docstring asserts observation projection; hidden b* not accessible and numerical equality unverified'}
    report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('PASS', 'target mismatch', result['state_template_mismatch_total'], 'seconds', result['seconds'], flush=True)


if __name__ == '__main__':
    # Import upstream hash function only for date selection; independent score math above.
    import sys
    sys.path.insert(0, str(Path('vendor/official/src').resolve()))
    main()
