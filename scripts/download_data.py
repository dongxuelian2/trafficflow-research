"""Download only the official public Kaggle package. Never prints credentials."""
import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile
import requests

COMPETITION = '2026-ieee-big-data-traffic-flow-bench'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('data'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    credentials = Path(os.environ.get('KAGGLE_CONFIG_DIR', str(Path.home() / '.kaggle'))) / 'kaggle.json'
    if os.environ.get('KAGGLE_API_TOKEN'):
        session.headers['Authorization'] = 'Bearer ' + os.environ['KAGGLE_API_TOKEN']
    elif os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        session.auth = (os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'])
    elif credentials.exists():
        config = json.loads(credentials.read_text())
        session.auth = (config['username'], config['key'])
    url = f'https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETITION}'
    response = session.get(url, stream=True, timeout=(30, 120))
    kind = response.headers.get('Content-Type', '')
    if response.status_code != 200 or ('zip' not in kind and 'octet-stream' not in kind):
        status = {'status': 'blocked', 'http_status': response.status_code, 'content_type': kind,
                  'reason': 'Official endpoint did not return a data archive; Kaggle access/rules acceptance may be required.'}
        (args.output / 'download_status.json').write_text(json.dumps(status, indent=2))
        print(json.dumps(status))
        raise SystemExit(2)
    archive = args.output / (COMPETITION + '.zip')
    if shutil.disk_usage(args.output).free < 30 * 1024 ** 3:
        raise RuntimeError('At least 30 GiB free is required for archive plus extraction.')
    total = 0
    with archive.open('wb') as out:
        for chunk in response.iter_content(8 * 1024 ** 2):
            if chunk:
                out.write(chunk)
                total += len(chunk)
                if total % (256 * 1024 ** 2) < len(chunk):
                    print(f'Downloaded {total / 1024 ** 3:.2f} GiB', flush=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (args.output / member.filename).resolve()
            if not target.is_relative_to(args.output.resolve()):
                raise ValueError('Archive path escapes output directory')
        z.extractall(args.output)
    print('Downloaded and extracted official package to', args.output.resolve())

if __name__ == '__main__':
    main()
