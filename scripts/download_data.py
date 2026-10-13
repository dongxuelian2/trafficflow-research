"""Download only the official public Kaggle package. Never prints credentials."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import zipfile
import requests

COMPETITION = '2026-ieee-big-data-traffic-flow-bench'
ENDPOINT = f'https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETITION}'

def authenticated_session():
    session = requests.Session()
    config_dir = Path(os.environ.get('KAGGLE_CONFIG_DIR', str(Path.home() / '.kaggle')))
    credentials = config_dir / 'kaggle.json'
    if os.environ.get('KAGGLE_API_TOKEN'):
        session.headers['Authorization'] = 'Bearer ' + os.environ['KAGGLE_API_TOKEN']
    elif os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        session.auth = (os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'])
    elif credentials.exists():
        config = json.loads(credentials.read_text())
        session.auth = (config['username'], config['key'])
    elif (config_dir / 'access_token').exists():
        session.headers['Authorization'] = 'Bearer ' + (config_dir / 'access_token').read_text().strip()
    return session

def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)

def parallel_archive(archive, response, workers, chunk_mib):
    total = int(response.headers.get('Content-Length', 0))
    etag = response.headers.get('ETag', '')
    # The signed URL stays in memory. Never put it in logs or the resume manifest.
    url = response.url
    response.close()
    if total <= 0:
        raise RuntimeError('Official server did not supply archive size')
    ledger_path = archive.with_suffix('.parts.json')
    chunk_size = chunk_mib * 1024**2
    segments = [(start, min(start+chunk_size, total)-1) for start in range(0, total, chunk_size)]
    ledger = {'total_bytes':total, 'etag':etag, 'chunk_bytes':chunk_size, 'completed':[]}
    old_size = archive.stat().st_size if archive.exists() else 0
    if ledger_path.exists():
        existing = json.loads(ledger_path.read_text())
        if any(existing[k] != ledger[k] for k in ['total_bytes','etag','chunk_bytes']):
            raise RuntimeError('Resume metadata differs from current official archive')
        ledger = existing
    elif old_size:
        # Adopt an earlier sequential download only after checking its prefix.
        if old_size >= total:
            raise RuntimeError('Existing archive has no resume ledger; verify it before reusing')
        with requests.get(url, headers={'Range':'bytes=0-65535'}, timeout=(30,60)) as prefix:
            if prefix.status_code != 206:
                raise RuntimeError('Server does not honor Range requests')
            with archive.open('rb') as f:
                if f.read(65536) != prefix.content:
                    raise RuntimeError('Existing prefix differs from official archive')
        ledger['completed'] = [i for i,(start,end) in enumerate(segments) if end < old_size]
    save_json(ledger_path, ledger)
    if not archive.exists():
        archive.touch()
    with archive.open('r+b') as f:
        f.truncate(total)
    done = set(ledger['completed'])
    lock = threading.Lock()
    initial = sum(segments[i][1]-segments[i][0]+1 for i in done)
    started = time.monotonic()
    last_report = started-30
    print(f'Official archive: {total/1024**3:.2f} GiB; verified prefix: {initial/1024**3:.2f} GiB; workers={workers}', flush=True)

    def transfer(index):
        start,end = segments[index]
        for attempt in range(5):
            try:
                with requests.get(url, headers={'Range':f'bytes={start}-{end}'}, stream=True, timeout=(30,90)) as part:
                    if part.status_code != 206 or part.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                        raise RuntimeError(f'Range response invalid (HTTP {part.status_code})')
                    count=0
                    with archive.open('r+b') as f:
                        f.seek(start)
                        for block in part.iter_content(1024**2):
                            if block:
                                f.write(block);count+=len(block)
                        f.flush()
                    if count != end-start+1:
                        raise RuntimeError('Incomplete range')
                return index
            except (requests.RequestException, RuntimeError):
                if attempt == 4:
                    raise RuntimeError(f'Range {index} failed after retries; rerun to resume') from None
                time.sleep(min(2**attempt,16))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(transfer,i) for i in range(len(segments)) if i not in done]
        for future in as_completed(futures):
            index = future.result()
            with lock:
                done.add(index);ledger['completed']=sorted(done)
                save_json(ledger_path,ledger)
                completed = sum(segments[i][1]-segments[i][0]+1 for i in done)
                now=time.monotonic()
                if now-last_report >= 15 or completed==total:
                    rate=(completed-initial)/max(now-started,1)/1024**2
                    print(f'Downloaded {completed/1024**3:.2f}/{total/1024**3:.2f} GiB ({100*completed/total:.1f}%), {rate:.1f} MiB/s',flush=True)
                    last_report=now
    return {'archive_bytes':total,'etag':etag,'workers':workers}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('data'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--chunk-mib', type=int, default=32)
    args = ap.parse_args()
    if not 1 <= args.workers <= 16 or args.chunk_mib < 1:
        raise ValueError('Workers must be 1..16; chunk size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    response = authenticated_session().get(ENDPOINT, stream=True, timeout=(30,120))
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
    metadata = parallel_archive(archive, response, args.workers, args.chunk_mib)
    print('Verifying SHA-256 and ZIP directory',flush=True)
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            digest.update(block)
    metadata['archive_sha256']=digest.hexdigest()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (args.output / member.filename).resolve()
            if not target.is_relative_to(args.output.resolve()):
                raise ValueError('Archive path escapes output directory')
        metadata['zip_entries']=len(z.infolist())
        metadata['unpacked_bytes']=sum(m.file_size for m in z.infolist())
        print(f'Extracting {metadata["zip_entries"]} entries; each entry CRC is checked by ZipFile',flush=True)
        for i,member in enumerate(z.infolist(),1):
            z.extract(member,args.output)
            if i%1000==0:
                print(f'Extracted {i}/{metadata["zip_entries"]}',flush=True)
    metadata['status']='complete'
    metadata['source']='official Kaggle competition download endpoint'
    save_json(args.output/'download_status.json',metadata)
    print('Downloaded and extracted official package to', args.output.resolve())

if __name__ == '__main__':
    main()
