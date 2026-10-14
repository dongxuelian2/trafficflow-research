"""Build a local one-direction State/Physics control with full task coverage.

Queue and OD stay at the exact frozen formal baseline. Task 3 is derived from
State, so its true score cannot be held fixed or measured locally.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release, STATE_KEY, VALUES, csv
from trafficflow.experiments import day_template
from trafficflow.state import StateModel, StateFeatures
from trafficflow.adaptive_graph_state import AdaptiveStateModel
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import register_protocol
from trafficflow.submission import merge_files, validate_task


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--production', type=Path, required=True)
    parser.add_argument('--architecture', choices=['adaptive_attention', 'adaptive_attention_density'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a complete candidate')
    production = json.loads((args.production/'experiment.json').read_text())
    config = production['registered']['protocol']
    panel = config['panel']
    model_path = args.production/(args.architecture+'.pkl')
    if sha256_file(model_path) != production['models'][args.architecture]['deployable_model_sha256']:
        raise ValueError('Production model changed')
    source = Path('artifacts/submission_best')
    expected = {'state': '49c48651db680a46a05f952ae4a89001374ea5912d73627d9a2ad8117e53a294',
        'queue': '420080d4582ff74f96a33bc5ca726aedb541ca3f576ced416be307d6c22d3fa4',
        'odme': '7091490ad973fe0ebc017f1a69523441fce330d1d731fcda02ba7f6ff4bd5f83'}
    for task, digest in expected.items():
        if sha256_file(source/(task+'.csv')) != digest:
            raise ValueError('Frozen component changed')
    if sha256_file(source/'submission.csv') != '43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1':
        raise ValueError('Original best changed')
    registered = register_protocol(Path('reports/research')/(args.output.name+'.registered.json'), {
        'id': 'Local_one_direction_adaptive_State_Physics_control', 'panel': panel,
        'architecture': args.architecture, 'model_sha256': sha256_file(model_path),
        'production_report_sha256': sha256_file(args.production/'experiment.json'),
        'alpha': .75, 'baseline_ref': 57138874, 'baseline_formal_Total': .82940,
        'hypothesis': 'A development-supported missing-aware State mixture may improve reconstruction; true Physics impact remains unmeasured.',
        'changed_fields': 'Only speed_kmh and flow_vph for this direction, validation and private',
        'frozen_fields': 'Every other State direction, every Queue prediction and every OD path flow',
        'legal_inputs': 'Public train labels for fitting; released masked frames for offline State inference',
        'future_State_to_Queue': False, 'uploaded': False,
        'formal_State': None, 'formal_Physics': None, 'formal_Total': None})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    start = time.perf_counter()
    release = Release('data/kaggle_public')
    model = StateModel.load(model_path)
    if not isinstance(model, AdaptiveStateModel) or model.alpha != .75:
        raise ValueError('Unexpected production artifact')
    net, topo = release.network(panel), release.topology(panel)
    predictions = []
    daily = []
    for split in ('validation', 'private'):
        template = release.template(1, panel, split)
        rows = 0
        for path in release.files(panel, split):
            frame = release.read(path)
            target = day_template(release, panel, split, frame, template)
            if not len(target):
                continue
            f = StateFeatures(frame, net, topo, model.profile, version=model.feature_version)
            pred = model.predict(f, target)
            validate_task(pred, 'state')
            predictions.append(pred)
            rows += len(pred)
            daily.append({'split': split, 'date': str(target.timestamp.iloc[0])[:10], 'rows': len(pred),
                'source_sha256': sha256_file(path)})
        if rows != len(template):
            raise ValueError('Incomplete direction/split prediction')
    modified = pd.concat(predictions, ignore_index=True)
    validate_task(modified, 'state')
    lookup = modified.set_index(STATE_KEY)
    state_path = args.output/'state.csv'
    written = 0
    replaced = 0
    for chunk in pd.read_csv(source/'state.csv', dtype=str, keep_default_na=False, chunksize=100000):
        chosen = chunk.panel.eq(panel)
        if chosen.any():
            keys = pd.MultiIndex.from_frame(chunk.loc[chosen, STATE_KEY])
            replacement = lookup.reindex(keys)[VALUES]
            if not np.isfinite(replacement.to_numpy(float)).all():
                raise ValueError('Source component requires an absent candidate key')
            for column in VALUES:
                chunk.loc[chosen, column] = replacement[column].map(repr).to_numpy()
            replaced += int(chosen.sum())
        chunk.to_csv(state_path, index=False, mode='w' if not written else 'a', header=not written)
        written += len(chunk)
    if replaced != len(modified):
        raise ValueError('Candidate predictions and required frozen keys differ')
    files = {'state': state_path, 'queue': source/'queue.csv', 'odme': source/'odme.csv'}
    merged = merge_files(files, release.root/'submission_key.csv', args.output/'submission.csv')
    result = {'registered': registered, 'source_snapshot': snapshot, 'coverage': merged,
        'replaced_State_rows': replaced, 'prediction_days': daily,
        'component_sha256': {task: sha256_file(path) for task, path in files.items()},
        'submission_sha256': sha256_file(args.output/'submission.csv'),
        'seconds': time.perf_counter()-start, 'source': evidence(gpu_used=True),
        'uploaded': False, 'formal_State': None, 'formal_Physics': None, 'formal_Total': None,
        'limitations': 'Local complete candidate, not a promoted formal best. No validation/private labels are read. Task 3 uses changed State values, and its hidden boundary-flow score is unknown.'}
    (args.output/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (Path('reports/research')/(args.output.name+'.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(args.output, result['submission_sha256'], replaced, result['seconds'], flush=True)
