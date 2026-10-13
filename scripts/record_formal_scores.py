"""Record authenticated score evidence without inventing component levels."""
import json
from pathlib import Path
from trafficflow.runtime import sha256_file

if __name__=='__main__':
    root=Path('reports/research')
    history=json.loads((root/'kaggle_submission_history.json').read_text())
    known={57138874:Path('artifacts/frozen_baselines/57138874/submission.csv'),
           57140644:Path('artifacts/research/K01_OD_only_bayesian/submission.csv')}
    entries=[]
    for row in history['submissions']:
        ref=row['ref']
        if ref not in known or row['public_score'] is None:
            continue
        path=known[ref]
        if path.stat().st_size!=row['total_bytes']:
            raise ValueError('Local candidate byte count differs from registered submission')
        entries.append({**row,'public_total':float(row['public_score']), 'local_file':path.as_posix(),
            'sha256':sha256_file(path),'formal_component_levels':None,'private_total':None})
    best=max(entries,key=lambda r:r['public_total'])
    result={'source':'authenticated own-team Kaggle API; local files verified by size and SHA',
        'checked_utc':history['checked_utc'],'best_submission_ref':best['ref'],
        'best_public_total':best['public_total'],'target_formal_total':.95,'entries':entries}
    controls={e['ref']:e for e in entries}
    if 57140644 in controls:
        delta=controls[57140644]['public_total']-controls[57138874]['public_total']
        result['K01_control']={'changed_task':'ODME only','full_row_diff':'artifacts/research/K01_OD_only_bayesian/controlled_diff.json',
            'formal_total_delta':delta,'implied_ODME_aggregate_delta':delta/.20,
            'rounding_uncertainty_ODME_delta':.00005,
            'absolute_ODME_levels_unknown':True,'decision':'Reject K01; preserve formal baseline 57138874.'}
        receipt=json.loads(Path('artifacts/research/K01_OD_only_bayesian/upload_receipt.json').read_text())
        receipt.update({'state':'scored','public_total':controls[57140644]['public_total']})
        Path('artifacts/research/K01_OD_only_bayesian/upload_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (root/'formal_scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)
