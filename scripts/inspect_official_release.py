"""Stage a few CURRENT official public assets while the full archive downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from urllib.parse import quote
from download_data import authenticated_session, COMPETITION, save_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--panel',default='D12_I5_N')
    ap.add_argument('--output',type=Path,default=Path('data/early'))
    args=ap.parse_args()
    session=authenticated_session()
    files=[];token=''
    while True:
        r=session.get(f'https://www.kaggle.com/api/v1/competitions/data/list/{COMPETITION}',
                      params={'pageSize':200,'pageToken':token},timeout=60)
        if r.status_code!=200:
            raise RuntimeError(f'Official list HTTP {r.status_code}')
        obj=r.json();files.extend(obj['files']);token=obj.get('nextPageToken','')
        if len(files)%2000==0:
            print(f'Listed {len(files)} public files',flush=True)
        if not token:
            break
    # Persist only public metadata; omit download URLs and all authentication material.
    inventory=[{'name':f['name'],'bytes':f.get('totalBytes'),'creationDate':f.get('creationDate')} for f in files]
    save_json(Path('data/official_file_list.json'),{'competition':COMPETITION,'files':inventory})
    prefixes=[f'kaggle_public/corridors/{args.panel}/network/',f'kaggle_public/config/']
    selected=[]
    for f in files:
        name=f['name']
        take=name=='kaggle_public/README.md' or any(name.startswith(p) for p in prefixes)
        if name.startswith(f'kaggle_public/corridors/{args.panel}/train/') and re.search(r'2030_06_0[12]\.parquet$',name):
            take=True
        if name.startswith(f'kaggle_public/task2/{args.panel}/') or name.startswith(f'kaggle_public/task4/{args.panel}/'):
            take=True
        if take:
            selected.append(f)
    print(f'Official file list complete: {len(files)} files; staging {len(selected)} assets',flush=True)
    def download(f):
        dst=(args.output/f['name']).resolve()
        if not dst.is_relative_to(args.output.resolve()):
            raise ValueError('Unsafe public asset path')
        expected=f.get('totalBytes')
        if dst.exists() and dst.stat().st_size==expected:
            return
        response=authenticated_session().get(f'https://www.kaggle.com/api/v1/competitions/data/download/{COMPETITION}/{quote(f["name"],safe="")}',stream=True,timeout=(30,120))
        if response.status_code!=200:
            raise RuntimeError(f'Official file HTTP {response.status_code}')
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('wb') as out:
            for block in response.iter_content(1024**2):
                out.write(block)
        if expected is not None and dst.stat().st_size!=expected:
            raise ValueError('Official file size mismatch')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(download,f) for f in selected]):
            future.result()
    print('Staged public assets at',args.output.resolve(),flush=True)

if __name__=='__main__':
    main()
