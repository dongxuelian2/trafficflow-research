"""Complete LOCAL OD-only expert control; never contacts a submission API."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release, OD_KEY
from trafficflow.odme import operator
from trafficflow.od_expert import prior_expert_inverse
from trafficflow.metrics import od_link_score
from trafficflow.submission import merge_files
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot, evidence, sha256_file
from od_expert_experiment import select_expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a candidate')
    study = json.loads(args.study.read_text())
    source_dir = Path('artifacts/research')/args.study.stem
    selection = json.loads((source_dir/'selection_sealed.json').read_text())
    config = json.loads((source_dir/'OD_expert.json').read_text())
    development = json.loads((source_dir/'development_only.json').read_text())
    if selection != study['selection'] or select_expert(development) != selection or config != selection['deployment_config']:
        raise ValueError('Development choice or frozen deployment config changed')
    if not selection['development_gate_passed'] or study['optimizer_failures'] or study['paired_interval']['ci95'][0] <= 0:
        raise ValueError('Require complete positive offline evidence before building an aggressive control')
    frozen_digest = '43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1'
    baseline = Path('artifacts/frozen_baselines/57138874/submission.csv')
    if sha256_file(baseline) != frozen_digest:
        raise ValueError('Formal baseline changed')
    hypothesis = register_protocol(Path('reports/research')/(args.output.name+'.registered.json'), {
        'id': 'Local_OD_expert_control', 'baseline_ref': 57138874, 'baseline_formal_Total': .82940,
        'baseline_sha256': frozen_digest, 'source_study_sha256': sha256_file(args.study),
        'configuration': config, 'changed_task': 'ODME only',
        'retained_components': 'Original State and Queue; Physics derived from unchanged State',
        'hypothesis': 'Entropy resolves broad weak-prior mismatch; concentration-gated MAP handles sparse demand. Report actual branch selection on public inputs.',
        'offline_evidence': 'Nine known-truth distributions, five public corridor families, three new research seeds; still synthetic assumptions',
        'formal_gain': None, 'formal_ODME': None, 'uploaded': False,
        'platform_policy': 'Local modeling only per latest user direction; this build has no upload/publish/push step'})
    args.output.mkdir(parents=True)
    snapshot = capture_source_snapshot(args.output/'source_snapshot.zip')
    started = time.perf_counter()
    release = Release('data/kaggle_public')
    frames, diagnostics = [], []
    for panel in release.panels():
        for split in ['validation', 'private']:
            A, c, b, paths = operator(release, panel, split)
            f, info = prior_expert_inverse(A, c, b, config)
            target = release.template(4, panel, split)
            if set(target.path_id) != set(paths.path_id):
                raise ValueError('OD template path coverage differs')
            indexed = paths.set_index('path_id')
            for zone in ['origin_zone', 'destination_zone']:
                if not np.array_equal(target[zone].to_numpy(), target.path_id.map(indexed[zone]).to_numpy()):
                    raise ValueError('OD template zone identity differs')
            target['path_flow'] = target.path_id.map(dict(zip(paths.path_id, f)))
            if not np.isfinite(target.path_flow.to_numpy(float)).all() or (target.path_flow < 0).any():
                raise ValueError('Invalid or absent OD prediction')
            frames.append(target[OD_KEY+['path_flow']])
            diagnostics.append({'panel': panel, 'split': split, **info,
                'S_link_only': od_link_score(A, f, c), 'formal_ODME': None})
    pd.concat(frames, ignore_index=True).to_csv(args.output/'odme.csv', index=False)
    original = Path('artifacts/submission_best')
    files = {'state': original/'state.csv', 'queue': original/'queue.csv', 'odme': args.output/'odme.csv'}
    expected = {'state': '49c48651db680a46a05f952ae4a89001374ea5912d73627d9a2ad8117e53a294',
        'queue': '420080d4582ff74f96a33bc5ca726aedb541ca3f576ced416be307d6c22d3fa4'}
    if any(sha256_file(files[k]) != digest for k, digest in expected.items()):
        raise ValueError('Retained component changed')
    coverage = merge_files(files, release.root/'submission_key.csv', args.output/'submission.csv')
    result = {'registered': hypothesis, 'source_snapshot': snapshot, 'coverage': coverage,
        'OD_diagnostics': diagnostics, 'component_sha256': {k: sha256_file(v) for k, v in files.items()},
        'submission_sha256': sha256_file(args.output/'submission.csv'), 'seconds': time.perf_counter()-started,
        'source': evidence(), 'formal_Total': None, 'formal_ODME': None, 'uploaded': False,
        'n_split_panels_using_alternative': sum(r['alternative_applied'] for r in diagnostics),
        'scope': 'Complete candidate only. Separate full row-wise baseline diff required; offline score is not formal score.'}
    (args.output/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report = Path('reports/research')/(args.output.name+'.json')
    if report.exists():
        raise FileExistsError('Never overwrite a candidate report')
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('sha256', result['submission_sha256'], 'coverage', coverage, 'seconds', result['seconds'], 'uploaded', False, flush=True)


if __name__ == '__main__':
    main()
