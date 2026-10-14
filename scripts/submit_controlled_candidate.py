"""Upload exactly one verified registered candidate, recording its own receipt.

Human authorization for controlled competition submissions is in this research
request. This script does not select final entries or submit other candidates.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from trafficflow.runtime import sha256_file

COMPETITION='2026-ieee-big-data-traffic-flow-bench'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    receipt=args.directory/'upload_receipt.json'
    if receipt.exists():
        raise FileExistsError('Receipt exists; inspect history rather than upload again')
    verification=json.loads((args.directory/'verification.json').read_text())
    difference=json.loads((args.directory/'controlled_diff.json').read_text())
    candidate=args.directory/'submission.csv'
    sha=sha256_file(candidate)
    if not verification['complete'] or not difference['full_row_diff_passed']:
        raise ValueError('Candidate validation is incomplete')
    if sha!=verification['submission_sha256'] or sha!=difference['candidate_sha256']:
        raise ValueError('Candidate changed after verification')
    api=KaggleApi()
    api.authenticate()
    previous=api.competition_submissions(COMPETITION,page_size=100) or []
    limits=api.competition_get_submission_limits(COMPETITION)
    if limits.num_allowed_now<1:
        raise RuntimeError('No submissions remaining; retain candidate locally')
    hypothesis=verification['hypothesis']['id']
    task=difference['allowed_task']
    if task not in ('state','queue','odme'):raise ValueError('Unregistered changed task')
    label={'state':'State-Physics','queue':'Queue','odme':'ODME'}[task]
    message=f'{hypothesis}; baseline 57138874; {label} only; SHA256 {sha[:16]}'
    # Write before mutation so an interrupted upload cannot trigger a blind retry.
    record={'candidate_sha256':sha,'hypothesis':hypothesis,'started_utc':datetime.now(timezone.utc).isoformat(),
            'previous_submission_refs':[s.ref for s in previous],'state':'upload_started',
            'public_total':None,'private_total':None,'competition':COMPETITION}
    receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
    response=api.competition_submit(str(candidate.resolve()),message,COMPETITION,quiet=True)
    record['state']='API_returned'
    record['message']=getattr(response,'message',None)
    record['returned_utc']=datetime.now(timezone.utc).isoformat()
    # SDK response may expose the submission number directly.
    for field in ('ref','submission_id','url'):
        value=getattr(response,field,None)
        if value is not None:
            record[field]=value
    receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(record)


if __name__=='__main__':
    main()
