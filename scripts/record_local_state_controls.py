"""Freeze verified complete local State controls without platform operations."""
import argparse
import json
from pathlib import Path
from trafficflow.runtime import sha256_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, action='append', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a local control record')
    rows = []
    for directory in args.candidate:
        verification = json.loads((directory/'verification.json').read_text())
        diff = json.loads((directory/'controlled_diff.json').read_text())
        digest = sha256_file(directory/'submission.csv')
        if digest != verification['submission_sha256'] or digest != diff['candidate_sha256']:
            raise ValueError('Verified submission changed')
        if not diff['full_row_diff_passed'] or diff['allowed_task'] != 'state' or diff['allowed_panel'] != 'D12_I5_N':
            raise ValueError('Expected a fully checked one-direction State control')
        if diff['rows'] != 6980503 or verification['coverage']['rows'] != 6980503 or not verification['coverage']['complete']:
            raise ValueError('Full task coverage missing')
        if verification['uploaded'] or verification['formal_Total'] is not None:
            raise ValueError('This record is for unuploaded local candidates only')
        rows.append({'directory': directory.as_posix(), 'submission_sha256': digest,
            'submission_bytes': (directory/'submission.csv').stat().st_size,
            'hypothesis': verification['registered']['protocol'], 'coverage': verification['coverage'],
            'full_row_diff': diff, 'component_sha256': verification['component_sha256'],
            'prediction_merge_seconds': verification['seconds'],
            'verification_record_sha256': sha256_file(directory/'verification.json'),
            'diff_record_sha256': sha256_file(directory/'controlled_diff.json')})
    result = {'formal_best_ref': 57138874, 'formal_best_Total': .82940, 'formal_target_Total': .95000,
        'candidates': rows, 'uploaded': False, 'formal_candidate_Total': None,
        'platform_policy': 'User requested local model research; no forum, Notebook, GitHub push or formal submission in this phase.',
        'limitations': 'Production fits all public train labels and has no production holdout metric. True State/Physics impact requires formal measurement. Task 3 changes with State; other directions, Queue and OD values are preserved.'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({row['directory']: row['submission_sha256'] for row in rows})
