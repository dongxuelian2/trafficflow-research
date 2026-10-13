"""Bounded, resumable queue calibration and final predictions on eight panels."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from trafficflow.data import Release

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/queue_refinement'));ap.add_argument('--panel',action='append')
    args=ap.parse_args();panels=args.panel or [p for p in Release(args.release_root).panels() if not p.startswith('D12_I405')]
    records=[]
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
    for panel in panels:
        folder=args.output/panel;folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
        command=[sys.executable,str(Path(__file__).with_name('queue_refinement.py').resolve()),'--release-root',str(args.release_root.resolve()),
                '--output',str(folder.resolve()),'--panel',panel,'--production']
        with (folder/'refinement.log').open('a',encoding='utf-8') as log:
            p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        records.append({'panel':panel,'returncode':p.returncode,'seconds':time.perf_counter()-started,'command':command})
        (args.output/'suite.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
        if p.returncode:raise RuntimeError('Queue refinement failed; inspect panel log')

if __name__=='__main__':main()
