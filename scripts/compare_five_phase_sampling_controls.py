"""Compare original and balanced policies on identical balanced assessment events."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.causal_queue import released_origin_rows
from trafficflow.queue import QueueModel
from trafficflow.queue_domain import score_full_forecast
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from compare_queue_events import comparison


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--balanced',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite paired sampler controls')
    old,new=[json.loads((p/'experiment.json').read_text()) for p in [args.original,args.balanced]]
    panel=new['registered']['protocol']['panel']
    if old['dates']!=new['dates'] or old['registered']['protocol']['panel']!=panel:raise ValueError('Mechanism dates/direction differ')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Frozen_original_vs_balanced_Queue_policy_same_assessment_origins',
        'original_experiment_sha256':sha256_file(args.original/'experiment.json'),
        'balanced_experiment_sha256':sha256_file(args.balanced/'experiment.json'),
        'scope':'Both saved selected roles,no refit or threshold changes. Same balanced assessment labels/complete public template.',
        'limitation':'Changes fit/checkpoint/decision/selection sampling jointly; not a single sampling-gradient ablation',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');net,topology=release.network(panel),release.topology(panel)
    paths={partition_date(release,p):p for p in release.files(panel,'train',False)}
    models={}
    for role,run in [('original',old),('balanced',new)]:
        row=run['exported_models']['selected']
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Frozen selected role changed')
        models[role]=QueueModel.load(Path(row['path']))
    assessment={'dates':{'assessment':new['dates']['assessment']},'assessment':[]}
    for record in new['cache_records']:
        if record['phase']!='assessment':continue
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Assessment cache changed')
        data=dict(np.load(record['path'],allow_pickle=False));links=data['links'].tolist();predictions={k:[] for k in models}
        for i,origin_text in enumerate(data['origins']):
            origin=pd.Timestamp(str(origin_text));token=origin.strftime('%Y-%m-%dT%H:%M:%SZ')
            history=pd.read_parquet(paths[origin.date().isoformat()],filters=[
                ('timestamp','>=',(origin-pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ')),('timestamp','<',token)])
            for key in ['timestamp','link_id','station_id']:history[key]=history[key].astype(str)
            history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
            condition='queue_ongoing' if data['ongoing'][i] else 'queue_onset'
            times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
            target=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id']).to_frame(index=False)
            for name,model in models.items():
                predictions[name].extend(model.predict_window(history,target,net,topology,origin,condition).queue_pred.to_numpy())
        metrics={name:score_full_forecast(data['labels'].ravel(),np.array(pred),data['window'],data['condition'],data['available'].ravel()) for name,pred in predictions.items()}
        expected=next(r for r in new['assessment'] if r['seed']==record['seed'])['models'][new['selected']]
        if metrics['balanced']!=expected:raise ValueError('Saved balanced policy did not reproduce')
        assessment['assessment'].append({'seed':record['seed'],'models':metrics})
    result={'registered':registered,'source_snapshot':snapshot,'assessment':assessment['assessment'],
        'paired_same_population':comparison(assessment,'original','balanced'),'balanced_model_predictions_reproduced_exact':True,
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=False),'formal_Queue':None,'formal_Total':None,
        'limitations':'Old completed study and revised sampler are development evidence. Same assessment population avoids comparing incomparable raw score levels; policy weights/features/thresholds can all differ. No formal gain claim.'}
    if any(getattr(model,'alpha',0)>0 for model in models.values()):
        import torch
        result['source']['gpu_used']=bool(torch.cuda.is_available())
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print(result['paired_same_population'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
