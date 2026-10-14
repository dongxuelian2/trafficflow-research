"""Follow up TRAIN-window audit without treating low-quality speed as truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from trafficflow.data import Release, eligible
from trafficflow.experiments import partition_date
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import register_protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite quality diagnostics')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Public_TRAIN_Queue_observation_quality_projection','source_audit_sha256':sha256_file(args.audit),
        'scope':'All80 previously audited released TRAIN windows; future raw observations evaluator-only',
        'diagnostic':'Finite thresholded speeds before/after pct_observed75 eligibility, not latent queue truth',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip')
    source=json.loads(args.audit.read_text());release=Release('data/kaggle_public');rows=[]
    sources={}
    for old in source['rows']:
        panel=old['panel'];origin=pd.Timestamp(old['origin']);net=release.network(panel)
        if panel not in sources:
            sources[panel]={partition_date(release,p):p for p in release.files(panel,'train',False)}
        path=sources[panel][origin.date().isoformat()]
        if sha256_file(path)!=source['source_files_sha256'][str(path)]:raise ValueError('Audited public source changed')
        times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
        future=pd.read_parquet(path,filters=[('timestamp','in',times)])
        speed=pd.to_numeric(future.speed_kmh,errors='coerce').to_numpy(float)
        cut=future.link_id.astype(str).map(net.v_cut).to_numpy(float)
        positive=np.isfinite(speed)&(speed<=cut);score_eligible=eligible(future)
        scored=int((positive&score_eligible).sum());all_finite=int(positive.sum())
        if scored!=old['n_future_observation_positives']:raise ValueError('Previous eligible positives do not reproduce')
        discarded=future.loc[positive&~score_eligible,['timestamp','link_id','speed_kmh','flow_vph','pct_observed']]
        rows.append({'window_id':old['window_id'],'panel':panel,'condition':old['condition'],
            'n_finite_speed_positive_before_quality':all_finite,'n_positive_eligible':scored,
            'n_positive_discarded_by_quality':all_finite-scored,
            'all_finite_positives_discarded':bool(all_finite>0 and scored==0),
            'low_quality_positive_cells_for_zero_scored_case':discarded.to_dict('records') if scored==0 else [],
            'formal_Queue':None})
    result={'registered':registered,'source_snapshot':snapshot,'source_audit_sha256':sha256_file(args.audit),
        'rows':rows,'summary':{'windows':len(rows),'all_finite_positives_discarded_windows':sum(r['all_finite_positives_discarded'] for r in rows),
            'positive_cells_before_quality':sum(r['n_finite_speed_positive_before_quality'] for r in rows),
            'positive_cells_eligible':sum(r['n_positive_eligible'] for r in rows)},
        'source':evidence(),'formal_Queue':None,'formal_Total':None,
        'limitations':'Low-quality finite speed is a diagnostic only and not eligible truth. Even equality with the public condition does not prove latent queue labels, selector details or validation/private behavior. No future quality or values enter any predictor.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'experiment.json').write_text(text,encoding='utf-8');print(result['summary'],flush=True)


if __name__=='__main__':main()
