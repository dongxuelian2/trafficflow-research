"""Build a complete LOCAL I5_N Queue-only control, retaining formal best."""
import argparse
import json
from pathlib import Path
import time
import pandas as pd
from trafficflow.data import Release,csv,QUEUE_KEY
from trafficflow.submission import validate_task,merge_files
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite full controlled candidates')
    source=Path('artifacts/submission_best');release=Release('data/kaggle_public')
    expected={'state':'49c48651db680a46a05f952ae4a89001374ea5912d73627d9a2ad8117e53a294',
              'queue':'420080d4582ff74f96a33bc5ca726aedb541ca3f576ced416be307d6c22d3fa4',
              'odme':'7091490ad973fe0ebc017f1a69523441fce330d1d731fcda02ba7f6ff4bd5f83'}
    for task,digest in expected.items():
        if sha256_file(source/(task+'.csv'))!=digest:raise ValueError('Frozen original component changed')
    registered=register_protocol(Path('reports/research')/(args.output.name+'.registered.json'),{
        'id':'K06_I5_N_full_domain_Queue_LOCAL_control','baseline_ref':57138874,'baseline_Total':.82940,
        'scope':'Only D12_I5_N Queue values; State/OD and all other Queue panels retained',
        'hypothesis':'Causal T-inclusive tree/30min graph mixture and public onset timing with complete-domain cal, versus original baseline',
        'model':'C39 C34_before, cal-selected alpha .25, nonempty false; multiple algorithm changes, not isolated architecture test',
        'promotion_status':'Not promoted; same old noisy-proxy events, no new timing gain, one fit seed',
        'candidate_model_sha256':sha256_file(args.study/'C34_before.pkl'),
        'uploaded':False,'formal_Queue':None,'formal_Total':None})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    queue=csv(source/'queue.csv');modified=queue.copy();covered=[];changed=0
    for split in ['validation','private']:
        prediction_path=args.study/'predictions'/('D12_I5_N_'+split+'.csv')
        prediction=csv(prediction_path);template=release.template(2,'D12_I5_N',split)
        validate_task(prediction,'queue')
        keys=pd.MultiIndex.from_frame(prediction[QUEUE_KEY]);wanted=pd.MultiIndex.from_frame(template[QUEUE_KEY])
        if len(keys)!=len(wanted) or keys.has_duplicates or len(keys.difference(wanted)) or len(wanted.difference(keys)):
            raise ValueError('Controlled panel forecast key coverage differs')
        old_keys=pd.MultiIndex.from_frame(modified[QUEUE_KEY]);indices=old_keys.get_indexer(keys)
        if (indices<0).any():raise ValueError('Forecast keys absent from original complete component')
        old=pd.to_numeric(modified.iloc[indices].queue_pred).to_numpy();new=pd.to_numeric(prediction.queue_pred).to_numpy()
        changed+=int((old!=new).sum());modified.loc[modified.index[indices],'queue_pred']=prediction.queue_pred.to_numpy()
        covered.append({'split':split,'rows':len(prediction),'source_prediction_sha256':sha256_file(prediction_path),
                        'changed_values':int((old!=new).sum())})
    validate_task(modified,'queue');queue_path=args.output/'queue.csv';modified.to_csv(queue_path,index=False)
    files={'state':source/'state.csv','queue':queue_path,'odme':source/'odme.csv'}
    verification=merge_files(files,release.root/'submission_key.csv',args.output/'submission.csv')
    verification.update(registered=registered,source_snapshot=snapshot,controlled_panel_predictions=covered,
        queue_changed_values=changed,component_sha256={k:sha256_file(p) for k,p in files.items()},
        submission_sha256=sha256_file(args.output/'submission.csv'),seconds=time.perf_counter()-start,
        uploaded=False,formal_Queue=None,formal_Total=None,formal_best_retained=True)
    (args.output/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False),encoding='utf-8')
    (Path('reports/research')/(args.output.name+'.json')).write_text(json.dumps(verification,indent=2,allow_nan=False),encoding='utf-8')
    print('rows',verification['rows'],'changed_queue',changed,'sha256',verification['submission_sha256'],flush=True)


if __name__=='__main__':main()
