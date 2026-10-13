"""Compare causal queue context on frozen public-train date groups."""
import argparse,json,time
from pathlib import Path
import numpy as np
from trafficflow.data import Release
from trafficflow.experiments import groups,raw_for
from trafficflow.queue import QueueModel,proxy_windows
from trafficflow.metrics import grouped_iou
from trafficflow.runtime import evidence

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel',default='D12_I5_N')
    ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/queue_context'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    r=Release(args.release_root);partition,dates=groups(r,args.panel);net,topo=r.network(args.panel),r.topology(args.panel)
    data={}
    for split in ['fit','calibration']:
        started=time.perf_counter();data[split]=proxy_windows(raw_for(r,args.panel,partition[split]),net,topo,args.panel)
        print(split,'windows',len(np.unique(data[split].window)),'seconds',time.perf_counter()-started,flush=True)
    results={};models={}
    for name,method,iters in [('persistence','persistence',0),('physical_wave','wave',0),('lightgbm_150','lightgbm',150),('lightgbm_400','lightgbm',400)]:
        started=time.perf_counter();model=QueueModel(method,iterations=iters).fit(data['fit'])
        tuning=model.tune(data['calibration']) if method!='persistence' else {'selected_threshold':.5}
        d=data['calibration'];score=grouped_iou(d.y,model.probability(d.x)>=model.threshold,d.window,d.condition)
        models[name]=model;model.save(args.output/(name+'.pkl'))
        results[name]={'calibration':score,'tuning':tuning,'seconds':time.perf_counter()-started}
        print(name,'calibration proxy IoU',score['proxy_IoU'],flush=True)
    selected=max(results,key=lambda name:results[name]['calibration']['proxy_IoU'])
    # Only evaluate test after the candidate and threshold are frozen on calibration.
    d=proxy_windows(raw_for(r,args.panel,partition['test']),net,topo,args.panel)
    results[selected]['test']=grouped_iou(d.y,models[selected].probability(d.x)>=models[selected].threshold,d.window,d.condition)
    out={'panel':args.panel,'label_source':'public train observation proxy; not true queue labels',
         'dates':dates,'selected':selected,'results':results,'feature_names':data['fit'].x.columns.tolist(),
         'formal_S_queue':None,'runtime':evidence()}
    (args.output/'research.json').write_text(json.dumps(out,indent=2))
    print('selected',selected,'test proxy IoU',results[selected]['test']['proxy_IoU'],flush=True)

if __name__=='__main__':main()
