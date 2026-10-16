"""Materialize and fully check the unchanged, auditable fallback candidate.

No unpromoted research model is integrated. This is the historical .82940 file,
not a claimed reconstruction of the user's separately reported .83021 best.
"""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from trafficflow.runtime import sha256_file, evidence
from trafficflow.submission import OUTPUT, KEYS, VALUES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('artifacts/research/v2_gated_fallback_v1'))
    args = ap.parse_args()
    report = Path('reports/v2')/(args.output.name+'.json')
    if args.output.exists() or report.exists():
        raise FileExistsError('Never overwrite a verified candidate')
    start = time.perf_counter()
    frozen = Path('artifacts/frozen_baselines/57138874/submission.csv')
    expected_hash = '43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1'
    if sha256_file(frozen) != expected_hash:
        raise ValueError('Frozen reference bytes changed')
    args.output.mkdir(parents=True)
    candidate = args.output/'submission.csv'
    shutil.copy2(frozen, candidate)
    reader = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, chunksize=250000)
    counts = {k: 0 for k in KEYS}
    numeric_ranges = {c: [float('inf'), -float('inf')] for c in OUTPUT[2:]}
    changes = {c: 0 for c in OUTPUT}
    natural_hashes = {k: [] for k in KEYS}
    last_id, total = 0, 0
    sentinel = object()
    for key, pred, original in itertools.zip_longest(reader('data/kaggle_public/submission_key.csv'), reader(candidate), reader(frozen), fillvalue=sentinel):
        if any(x is sentinel for x in [key, pred, original]):
            raise ValueError('Candidate/master/reference row lengths differ')
        if pred.columns.tolist() != OUTPUT:
            raise ValueError('Submission schema mismatch')
        if len(pred) != len(key) or not np.array_equal(pred[['submission_id', 'task']].to_numpy(), key[['submission_id', 'task']].to_numpy()):
            raise ValueError('Master key order or coverage mismatch')
        ids = pd.to_numeric(pred.submission_id, errors='raise').to_numpy(np.int64)
        if not (ids[0] > last_id and np.all(np.diff(ids) > 0)):
            raise ValueError('Duplicate or unordered submission IDs')
        last_id = int(ids[-1])
        for c in OUTPUT:
            changes[c] += int((pred[c].to_numpy() != original[c].to_numpy()).sum())
        if not pred.task.isin(KEYS).all():
            raise ValueError('Unknown task')
        for c in OUTPUT[2:]:
            values = pd.to_numeric(pred[c], errors='raise').to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Nonfinite or negative field')
            numeric_ranges[c][0] = min(numeric_ranges[c][0], float(values.min()))
            numeric_ranges[c][1] = max(numeric_ranges[c][1], float(values.max()))
        for task in KEYS:
            selected = pred.task == task
            counts[task] += int(selected.sum())
            keys = key.loc[selected, KEYS[task]]
            if keys.isna().any().any() or keys.eq('').any().any():
                raise ValueError('Empty natural key')
            # If hashes are all unique, natural keys must be unique. A collision
            # can only cause a conservative rejection, never hide a duplicate.
            natural_hashes[task].append(pd.util.hash_pandas_object(keys, index=False).to_numpy(np.uint64))
            active = VALUES[task]
            for c in OUTPUT[2:]:
                if c not in active and (pd.to_numeric(pred.loc[selected, c]) != 0).any():
                    raise ValueError('Inactive task field is nonzero')
            if task == 'queue' and not pd.to_numeric(pred.loc[selected, 'queue_pred']).isin([0, 1]).all():
                raise ValueError('Nonbinary Queue output')
        total += len(pred)
    for task, hashes in natural_hashes.items():
        values = np.concatenate(hashes)
        if len(np.unique(values)) != len(values):
            raise ValueError('Natural key duplicate or conservative hash collision: '+task)
    digest = sha256_file(candidate)
    if digest != expected_hash or any(changes.values()) or total != 6980503:
        raise ValueError('Unchanged fallback verification failed')
    result = {'candidate': candidate.as_posix(), 'candidate_bytes': candidate.stat().st_size, 'sha256': digest,
        'rows': total, 'task_rows': counts, 'numeric_ranges_including_inactive_zeros': numeric_ranges,
        'full_row_field_difference_counts': changes, 'master_key_coverage': True, 'submission_id_unique': True,
        'natural_key_uniqueness': 'Unique deterministic uint64 hashes of each complete natural key, across all chunks; collisions would reject',
        'gate': 'Frozen fallback only; no V2 research arm promoted', 'known_historical_total': .82940,
        'user_reported_best': .83021, 'user_best_file_identified': False,
        'new_formal_Total': None, 'uploaded': False,
        'seconds': time.perf_counter()-start, 'runtime': evidence()}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (args.output/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('VERIFIED', total, digest, flush=True)


if __name__ == '__main__':
    main()
