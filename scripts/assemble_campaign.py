"""Stream per-panel outputs into task files, then strictly match the full key."""
import argparse,json,time
from pathlib import Path
import pandas as pd
from trafficflow.data import Release
from trafficflow.submission import merge_files
from trafficflow.runtime import evidence,sha256_file

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--campaign',type=Path,default=Path('artifacts/campaign_v1'))
    ap.add_argument('--queue-refinement',type=Path)
    ap.add_argument('--state-override',type=Path)
    ap.add_argument('--output',type=Path,default=Path('artifacts/submission_v1'));args=ap.parse_args()
    r=Release(args.release_root);panels=r.panels();args.output.mkdir(parents=True,exist_ok=True)
    files={t:args.output/f'{t}.csv' for t in ['state','queue','odme']}
    if any(p.exists() for p in files.values()):raise FileExistsError('Use a fresh output directory')
    started=time.perf_counter();counts={};queue_sources={};state_sources={}
    for task,path in files.items():
        first=True;count=0
        for panel in panels:
            source=args.campaign/panel/'predictions'/f'{task}.csv'
            if task=='queue' and panel.startswith('D12_I405'):continue
            if not (args.campaign/panel/'predictions'/'prediction_report.json').exists():raise ValueError(f'Incomplete panel: {panel}')
            if task=='state' and args.state_override:
                if not (args.state_override/panel/'production.json').exists():raise ValueError('Physics candidate incomplete')
                source=args.state_override/panel/'state.csv'
            if task=='state':state_sources[panel]=str(source.resolve())
            if task=='queue' and args.queue_refinement:
                refined=json.loads((args.queue_refinement/panel/'research.json').read_text())
                original=json.loads((args.campaign/panel/'experiment.json').read_text())
                ref_score=refined['results'][refined['selected']]['calibration']['proxy_IoU']
                old_score=original['queue'][original['selection']['queue']]['calibration']['proxy_IoU']
                if ref_score>=old_score-1e-12:
                    if not refined['production_refit']:raise ValueError('Refined final model not fitted')
                    source=args.queue_refinement/panel/'queue.csv'
                queue_sources[panel]=str(source.resolve())
            for d in pd.read_csv(source,dtype=str,keep_default_na=False,chunksize=100000):
                d.to_csv(path,index=False,mode='w' if first else 'a',header=first);first=False;count+=len(d)
        counts[task]=count;print(task,count,flush=True)
    result=merge_files(files,r.root/'submission_key.csv',args.output/'submission.csv')
    result.update({'task_rows':counts,'seconds':time.perf_counter()-started,'runtime':evidence(),
                   'sha256':{name:sha256_file(path) for name,path in {**files,'submission':args.output/'submission.csv'}.items()},
                   'formal_public_score':None,'uploaded':False,'queue_sources':queue_sources,'state_sources':state_sources})
    (args.output/'verification.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':main()
