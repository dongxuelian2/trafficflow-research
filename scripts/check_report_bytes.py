"""Verify raw report bytes in Git without changing the index or working tree."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess


def strict_json(data):
    def invalid(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(data,parse_constant=invalid)


def verify(path,revision):
    raw=Path(path).read_bytes();strict_json(raw)
    spec=':'+path if revision=='index' else revision+':'+path
    blob=subprocess.check_output(['git','show',spec])
    if blob!=raw:raise ValueError('Git/working bytes differ: '+path)
    return {'path':path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
        'Git_blob_exact':True,'strict_finite_JSON':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record',type=Path)
    parser.add_argument('--verify-record',type=Path);parser.add_argument('--revision',default='index')
    args=parser.parse_args()
    if bool(args.record)==bool(args.verify_record):raise ValueError('Choose record or verify-record')
    if args.verify_record:
        proof=strict_json(args.verify_record.read_bytes())
        for row in proof['reports']:
            actual=verify(row['path'],args.revision)
            if actual!=row:raise ValueError('Report changed: '+row['path'])
        verify(args.verify_record.as_posix(),args.revision)
        print('verified',len(proof['reports'])+1,'exact Git/working finite JSON blobs',args.revision,flush=True)
        return
    if args.revision!='index':raise ValueError('New records describe staged added reports only')
    names=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=A'],text=True).splitlines()
    rows=[verify(p,args.revision) for p in names if p.startswith('reports/') and p.endswith('.json')]
    if not rows:raise ValueError('No newly staged JSON reports')
    proof={'created_UTC':datetime.now(timezone.utc).isoformat(),'n_reports':len(rows),'reports':rows,
        'scope':'Added staged JSON reports byte-identical to working files. Proof itself verified separately after staging/commit.'}
    with args.record.open('x',encoding='utf-8') as stream:json.dump(proof,stream,indent=2,allow_nan=False)
    print('staged_report_bytes_exact',len(rows),'proof',args.record.as_posix(),flush=True)


if __name__=='__main__':main()
