"""Run untouched official Task 1 builders/scorers with bounded parallelism."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from trafficflow.data import Release,REPO

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/official_state_baselines'))
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--panel',action='append')
    args=ap.parse_args();release=Release(args.release_root)
    panels=args.panel or release.panels();args.output.mkdir(parents=True,exist_ok=True)
    def run(panel):
        folder=args.output/panel;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'score.csv').exists():
            return json.loads((folder/'run.json').read_text())
        started=time.perf_counter();started_at=datetime.now(timezone.utc).isoformat()
        env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
        source=REPO/'vendor'/'official'/'src'/'task1'
        commands=[[sys.executable,str(source/'build_task1_baseline_submission.py'),'--release-root',str(release.root),
                   '--split','train','--panel',panel,'--output',str((folder/'state.csv').resolve())],
                  [sys.executable,str(source/'score_task1.py'),'--release-root',str(release.root),'--split','train',
                   '--panel',panel,'--submission',str((folder/'state.csv').resolve()),'--output',str((folder/'score.csv').resolve())]]
        report={'panel':panel,'split':'train','source':'untouched official scripts','started_UTC':started_at,'commands':commands}
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            for command in commands:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                if result.returncode:
                    report.update({'success':False,'returncode':result.returncode,'seconds':time.perf_counter()-started})
                    (folder/'run.json').write_text(json.dumps(report,indent=2));return report
        report.update({'success':True,'seconds':time.perf_counter()-started})
        (folder/'run.json').write_text(json.dumps(report,indent=2));return report
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run,p) for p in panels]):
            result=future.result();records.append(result)
            print(json.dumps(result),flush=True)
            (args.output/'suite.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
