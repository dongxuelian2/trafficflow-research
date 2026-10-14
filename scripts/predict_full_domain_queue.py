"""Predict one public split/panel with a full-domain causal Queue model."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from trafficflow.data import Release,csv,QUEUE_KEY
from trafficflow.queue import QueueModel
from trafficflow.causal_queue import released_origin_rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--panel',required=True);parser.add_argument('--split',choices=['train','validation','private'],required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite Queue predictions')
    release=Release('data/kaggle_public');model=QueueModel.load(args.model)
    folder=release.root/'task2'/args.panel/args.split;windows=csv(folder/'window_index.csv')
    history=pd.read_parquet(folder/'window_history.parquet');template=release.template(2,args.panel,args.split)
    for key in ['timestamp','link_id','window_id']:history[key]=history[key].astype(str)
    net,topology=release.network(args.panel),release.topology(args.panel);pieces=[]
    for row in windows.itertuples(index=False):
        h=history[history.window_id==str(row.window_id)];target=template[template.window_id==str(row.window_id)].copy()
        origin=pd.Timestamp(h.timestamp.max())+pd.Timedelta(minutes=5)
        for key in ['forecast_origin','forecast_origin_time','origin_timestamp','T']:
            if hasattr(row,key):origin=pd.Timestamp(getattr(row,key));break
        h=pd.concat([h,released_origin_rows(release,args.panel,args.split,origin)],ignore_index=True)
        pieces.append(model.predict_window(h,target,net,topology,origin,str(row.condition))[QUEUE_KEY+['queue_pred']])
    result=pd.concat(pieces,ignore_index=True)
    if len(result)!=len(template) or result.duplicated(QUEUE_KEY).any() or not np.isin(result.queue_pred,[0,1]).all():raise ValueError('Incomplete or invalid Queue output')
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    print('rows',len(result),'saved',str(args.output),flush=True)


if __name__=='__main__':main()
