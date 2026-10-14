"""Full row-wise diff against the exact frozen formal baseline."""
import argparse
import csv
from itertools import zip_longest
import json
from pathlib import Path
from trafficflow.runtime import sha256_file

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--allowed-task',choices=['odme','queue','state'],required=True)
    parser.add_argument('--allowed-panel',help='Optional additional restriction using the public master key')
    args=parser.parse_args()
    baseline=Path('artifacts/frozen_baselines/57138874/submission.csv')
    if sha256_file(baseline)!='43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1':
        raise ValueError('Frozen baseline checksum mismatch')
    allowed={'odme':['path_flow'],'queue':['queue_pred'],'state':['speed_kmh','flow_vph']}[args.allowed_task]
    changed={};changed_by_panel_split={}
    count=0
    from contextlib import ExitStack
    with ExitStack() as stack:
        first=stack.enter_context(baseline.open(newline='',encoding='utf-8'))
        second=stack.enter_context(args.candidate.open(newline='',encoding='utf-8'))
        a,b=csv.DictReader(first),csv.DictReader(second)
        key_path=Path('data/kaggle_public/submission_key.csv')
        keys=csv.DictReader(stack.enter_context(key_path.open(newline='',encoding='utf-8'))) if args.allowed_panel else None
        if a.fieldnames!=b.fieldnames:
            raise ValueError('Candidate columns changed')
        for pair in (zip_longest(a,b,keys) if keys is not None else zip_longest(a,b)):
            old,new=pair[:2]
            count+=1
            if old is None or new is None:
                raise ValueError('Candidate length changed')
            wanted=pair[2] if keys is not None else None
            if keys is not None and (wanted is None or wanted['submission_id']!=old['submission_id'] or wanted['task']!=old['task']):
                raise ValueError('Master-key row order or identity differs')
            differences=[key for key in old if old[key]!=new[key]]
            if differences:
                if old['task']!=args.allowed_task or not set(differences).issubset(allowed):
                    raise ValueError(f'Unexpected changed task/field at row {count}: {differences}')
                if wanted is not None:
                    if wanted['panel']!=args.allowed_panel:
                        raise ValueError('Candidate changed an unapproved direction')
                    group=wanted['panel']+'/'+wanted['split']
                    changed_by_panel_split[group]=changed_by_panel_split.get(group,0)+1
                changed[args.allowed_task]=changed.get(args.allowed_task,0)+1
    out={'baseline_sha256':sha256_file(baseline),'candidate_sha256':sha256_file(args.candidate),
         'rows':count,'changed_rows':changed,'allowed_task':args.allowed_task,
         'full_row_diff_passed':True,'formal_score':None}
    if args.allowed_panel:
        out.update(allowed_panel=args.allowed_panel,changed_by_panel_split=changed_by_panel_split,
            public_master_key_sha256=sha256_file(key_path))
    args.candidate.with_name('controlled_diff.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(out)
