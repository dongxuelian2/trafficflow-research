"""Preserve exact first formal submission, weights, configurations and code."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path('artifacts/frozen_baselines/57138874')
    root.mkdir(parents=True, exist_ok=True)
    submitted = root / 'submission.csv'
    candidate = Path('artifacts/submission_best/submission.csv')
    if not submitted.exists():
        raise FileNotFoundError('Download own Kaggle submission before freezing')
    hashes = {'downloaded': digest(submitted), 'local_candidate': digest(candidate)}
    if hashes['downloaded'] != hashes['local_candidate']:
        raise ValueError('Formal submission does not match local candidate')
    code = root / 'source_15036c8.zip'
    if not code.exists():
        subprocess.run(['git', 'archive', '--format=zip', '--output=' + str(code), '15036c8'], check=True)
    sources = list(Path('artifacts/best_models').rglob('*.pkl'))
    sources += list(Path('artifacts/best_models').rglob('*.json'))
    sources += list(Path('artifacts/campaign_v1').rglob('experiment.json'))
    sources += list(Path('artifacts/campaign_v1').rglob('refit.json'))
    sources += list(Path('artifacts/queue_refinement').rglob('research.json'))
    sources += list(Path('artifacts/submission_best').glob('*.json'))
    sources += list(Path('reports/official').glob('*'))
    sources += [Path('requirements.lock.txt'), Path('config/research_targets.json'), Path('RESULTS.md')]
    for source in sources:
        if not source.is_file():
            continue
        destination = root / 'snapshot' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != digest(source):
                raise ValueError(f'Attempted frozen snapshot overwrite: {source}')
        else:
            shutil.copy2(source, destination)
    files = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
              'sha256': digest(p)} for p in sorted(root.rglob('*')) if p.is_file()]
    record = {'submission_ref': 57138874, 'formal_public_total': .82940,
              'formal_components': {'state': None, 'queue': None, 'physics': None, 'odme': None},
              'verification': 'own-team Kaggle API; own downloaded file matches local candidate SHA256',
              'submission_sha256': hashes['downloaded'], 'code_commit': '15036c8',
              'snapshot_root': root.as_posix(), 'files': files,
              'reproducibility_limits': 'Historical source was vectorized during campaign with recorded zero feature/inference difference; full independent retraining is not yet verified.'}
    destination = Path('reports/research/baseline_57138874_frozen.json')
    if destination.exists() and json.loads(destination.read_text(encoding='utf-8')) != record:
        raise ValueError('Freeze manifest already exists with different contents')
    destination.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print({'submission_sha256': hashes['downloaded'], 'frozen_files': len(files),
           'snapshot_bytes': sum(r['bytes'] for r in files)})


if __name__ == '__main__':
    main()
