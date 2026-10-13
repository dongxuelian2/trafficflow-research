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
    args=parser.parse_args()
    baseline=Path('artifacts/frozen_baselines/57138874/submission.csv')
    if sha256_file(baseline)!='43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1':
        raise ValueError('Frozen baseline checksum mismatch')
    allowed={'odme':['path_flow'],'queue':['queue_pred'],'state':['speed_kmh','flow_vph']}[args.allowed_task]
    changed={}
    count=0
    with baseline.open(newline='',encoding='utf-8') as first,args.candidate.open(newline='',encoding='utf-8') as second:
        a,b=csv.DictReader(first),csv.DictReader(second)
        if a.fieldnames!=b.fieldnames:
            raise ValueError('Candidate columns changed')
        for old,new in zip_longest(a,b):
            count+=1
            if old is None or new is None:
                raise ValueError('Candidate length changed')
            differences=[key for key in old if old[key]!=new[key]]
            if differences:
                if old['task']!=args.allowed_task or not set(differences).issubset(allowed):
                    raise ValueError(f'Unexpected changed task/field at row {count}: {differences}')
                changed[args.allowed_task]=changed.get(args.allowed_task,0)+1
    out={'baseline_sha256':sha256_file(baseline),'candidate_sha256':sha256_file(args.candidate),
         'rows':count,'changed_rows':changed,'allowed_task':args.allowed_task,
         'full_row_diff_passed':True,'formal_score':None}
    args.candidate.with_name('controlled_diff.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(out)
