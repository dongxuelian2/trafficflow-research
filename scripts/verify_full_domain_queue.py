"""Rebuild every calibration history and verify deployable full-domain Queue."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.queue import QueueModel
from trafficflow.causal_queue import released_origin_rows
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.queue_domain import score_full_forecast
from trafficflow.experiments import partition_date
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite full Queue deployment replay')
    run=json.loads((args.study/'experiment.json').read_text());release=Release('data/kaggle_public')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();records=[]
    for study in run['studies']:
        panel=study['panel'];net,topology=release.network(panel),release.topology(panel)
        rawpaths={partition_date(release,p):p for p in release.files(panel,'train',False)}
        p_record=study['cal_probability_record'];path=Path(p_record['path'])
        if sha256_file(path)!=p_record['sha256']:raise ValueError('Cal probability cache changed')
        data=dict(np.load(path,allow_pickle=False))
        for timing,model_record in study['models'].items():
            model_path=Path(model_record['path'])
            if sha256_file(model_path)!=model_record['sha256']:raise ValueError('Saved model changed')
            model=QueueModel.load(model_path);prediction=np.zeros(len(data['base']),int)
            choice=study['selected'][timing];setting=choice['tuning']['selected'];count=0
            probability=data['base'] if choice['column'] is None else (1-choice['alpha'])*data['base']+choice['alpha']*data['neural'][:,choice['column']]
            expected=full_queue_decisions(probability,data['window'],data['condition'],data['lead_minutes'],setting['thresholds'],setting['require_nonempty'],setting['timing'])
            for window in np.unique(data['window']):
                indices=np.flatnonzero(data['window']==window);origin=pd.Timestamp(window.rsplit(':',1)[0].split(':',1)[1])
                token=origin.strftime('%Y-%m-%dT%H:%M:%SZ');rawpath=rawpaths[origin.date().isoformat()]
                history=pd.read_parquet(rawpath,filters=[('timestamp','>=',(origin-pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ')),('timestamp','<',token)])
                for key in ['timestamp','station_id','link_id']:history[key]=history[key].astype(str)
                history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
                times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
                target=pd.MultiIndex.from_product([times,model.links],names=['timestamp','link_id']).to_frame(index=False).assign(window_id=window)
                prediction[indices]=model.predict_window(history,target,net,topology,origin,str(data['condition'][indices[0]])).queue_pred.to_numpy(int)
                count+=1
            if not np.array_equal(prediction,expected):raise ValueError('Standalone binary predictions differ')
            metric=score_full_forecast(data['labels'],prediction,data['window'],data['condition'],data['score_mask'])
            if abs(metric['proxy_IoU']-setting['calibration_proxy_IoU'])>1e-12:raise ValueError('Cal deployment score differs')
            records.append({'source_study':study['source_study'],'timing':timing,'model_sha256':sha256_file(model_path),
                'n_cal_windows':count,'all_full_template_predictions_exact':True,'cal_proxy_IoU':metric['proxy_IoU'],
                'n_forecast_cells':len(prediction),'history_rebuilt_from_public_le_T':True})
            print(study['source_study'],timing,count,'cal windows verified',flush=True)
    gpu_used=getattr(model,'_network',None) is not None and next(model._network.parameters()).is_cuda
    output={'source_experiment_sha256':sha256_file(args.study/'experiment.json'),'source_snapshot':snapshot,'models':records,
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=gpu_used),'formal_Queue':None,'formal_Total':None,
        'scope':'All calibration windows reconstructed from public <=T history; saved models predict complete templates exactly before scoring mask. Not independent model fitting or latent/hidden score.'}
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
