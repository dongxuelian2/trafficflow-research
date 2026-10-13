"""Compare a frozen finer calibration suite; evaluate test only after selection."""
import argparse,json,time
from pathlib import Path
import numpy as np
from trafficflow.data import Release
from trafficflow.experiments import groups,raw_for
from trafficflow.queue import QueueModel,proxy_windows,queue_features,QUEUE_KEY
from trafficflow.queue_calibration import RefinedQueueModel
from trafficflow.metrics import grouped_iou
from trafficflow.runtime import evidence

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel',default='D12_I5_N')
    ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/queue_refinement/D12_I5_N'))
    ap.add_argument('--production',action='store_true');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();initial=evidence();r=Release(args.release_root);p=args.panel;parts,dates=groups(r,p)
    net,topo=r.network(p),r.topology(p)
    existing=args.output/'research.json'
    if existing.exists():
        out=json.loads(existing.read_text());selected=out['selected'];model=QueueModel.load(args.output/'selected.pkl')
        if not args.production or out['production_refit']:return
        produce(r,p,parts,model,args.output,out,started);return
    fit=proxy_windows(raw_for(r,p,parts['fit']),net,topo,p);cal=proxy_windows(raw_for(r,p,parts['calibration']),net,topo,p)
    results={};models={}
    for iterations,leaves in [(400,23),(600,63)]:
        t=time.perf_counter();name=f'lightgbm_{iterations}_{leaves}'
        model=RefinedQueueModel(iterations,leaves).fit(fit);tuning=model.tune(cal)
        metric=grouped_iou(cal.y,model.classify(cal.x,cal.window,cal.condition),cal.window,cal.condition)
        model.save(args.output/(name+'.pkl'));models[name]=model
        results[name]={'calibration':metric,'tuning':tuning,'seconds':time.perf_counter()-t}
        print(p,name,'calibration proxy',metric['proxy_IoU'],flush=True)
    selected=max(results,key=lambda name:results[name]['calibration']['proxy_IoU']);model=models[selected]
    test=proxy_windows(raw_for(r,p,parts['test']),net,topo,p)
    results[selected]['test']=grouped_iou(test.y,model.classify(test.x,test.window,test.condition),test.window,test.condition)
    model.save(args.output/'selected.pkl')
    out={'panel':p,'selected':selected,'results':results,'dates':dates,'formal_S_queue':None,
        'source_at_start':initial['source_sha256'],'label_source':'public train observation proxy',
        'seconds':time.perf_counter()-started,'runtime':evidence(),'production_refit':False}
    (args.output/'research.json').write_text(json.dumps(out,indent=2))
    print(p,'selected',selected,'test proxy',results[selected]['test']['proxy_IoU'],flush=True)
    if args.production:
        del fit,cal,test,models
        produce(r,p,parts,model,args.output,out,started)

def produce(r,p,parts,model,output,out,started):
    full=proxy_windows(raw_for(r,p,parts['fit']+parts['calibration']+parts['test']),r.network(p),r.topology(p),p)
    model.fit(full);model.save(output/'queue_final.pkl');del full
    from trafficflow.queue import predict_windows
    import pandas as pd
    pieces=[predict_windows(r,p,split,model) for split in ['validation','private']]
    pd.concat(pieces,ignore_index=True).to_csv(output/'queue.csv',index=False)
    out['production_refit']=True;out['production_seconds']=time.perf_counter()-started;out['production_runtime']=evidence()
    (output/'research.json').write_text(json.dumps(out,indent=2))
    print(p,'refined queue prediction complete',flush=True)

if __name__=='__main__':main()
