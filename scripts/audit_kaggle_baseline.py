"""Read only the signed-in team's own submission history and submitted files.

Secrets and signed download URLs never enter logs. This script does not submit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from download_data import authenticated_session, COMPETITION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-score', type=float)
    args = parser.parse_args()
    output = Path('reports/research')
    output.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    rows = []
    own = api.competition_submissions(COMPETITION, page_size=100)
    for item in own or []:
        row = {name: getattr(item, name, None) for name in
               ('ref', 'date', 'description', 'file_name', 'total_bytes', 'public_score', 'status')}
        row['date'] = str(row['date'])
        row['status'] = str(row['status'])
        rows.append(row)
    limits = api.competition_get_submission_limits(COMPETITION)
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'competition': COMPETITION, 'source': 'authenticated own-team Kaggle API',
              'submissions': rows, 'limits': {k: getattr(limits, k) for k in
              ('num_today', 'num_total', 'num_allowed_now', 'limited_by_total')}}
    if args.download_score is not None:
        matching = [r for r in rows if r['public_score'] and
                    abs(float(r['public_score']) - args.download_score) < 1e-7]
        session = authenticated_session()
        for row in matching:
            destination = Path('artifacts/frozen_baselines') / str(row['ref']) / 'submission.csv'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                endpoint = f'https://www.kaggle.com/api/v1/competitions/submissions/download/{row["ref"]}'
                response = session.get(endpoint, stream=True, timeout=(30, 60))
                row['download_http_status'] = response.status_code
                if response.status_code != 200:
                    response.close()
                    continue
                temporary = destination.with_suffix('.part')
                with temporary.open('wb') as target:
                    for block in response.iter_content(1024 * 1024):
                        target.write(block)
                response.close()
                temporary.replace(destination)
            digest = hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest()
            row['local_file'] = destination.as_posix()
            row['sha256'] = digest
            row['downloaded_bytes'] = destination.stat().st_size
    path = output / 'kaggle_submission_history.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
