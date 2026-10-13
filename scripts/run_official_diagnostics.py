"""Untouched official OD baseline, link scorer and anchored physics diagnostic.

Physics output is explicitly NOT_FORMAL; no organizer boundary flux is loaded.
The builder's public_reference is never used as truth or as a training label.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from trafficflow.data import Release,REPO

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/official_diagnostics'))
    ap.add_argument('--state-root',type=Path,default=Path('artifacts/official_state_baselines'))
    ap.add_argument('--panel',action='append');args=ap.parse_args()
    r=Release(args.release_root);panels=args.panel or r.panels()
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    records=[]
    for panel in panels:
        folder=args.output/panel;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'run.json').exists() and json.loads((folder/'run.json').read_text()).get('success'):
            records.append(json.loads((folder/'run.json').read_text()));continue
        src=REPO/'vendor'/'official'/'src';started=time.perf_counter()
        od=folder/'odme';od.mkdir(exist_ok=True)
        commands=[
          [sys.executable,str(src/'task4/build_task4_odme_artifacts.py'),'--release-root',str(r.root),'--split','train',
           '--panel',panel,'--output-root',str(od.resolve())],
          [sys.executable,str(src/'task4/score_task4.py'),'--release-root',str(r.root),'--split','train',
           '--panel',panel,'--submission',str((od/'baseline_submission.csv').resolve()),'--output',str((folder/'odme_link.csv').resolve())],
          [sys.executable,str(src/'task3/score_task3.py'),'--release-root',str(r.root),'--split','train',
           '--panel',panel,'--state-submission',str((args.state_root/panel/'state.csv').resolve()),
           '--output',str((folder/'physics_NOT_FORMAL.csv').resolve())]]
        report={'panel':panel,'commands':commands,'formal_S_physics':None,'S_ODME':None}
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            for command in commands:
                p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                if p.returncode:report['failed_returncode']=p.returncode;break
        report.update({'success':'failed_returncode' not in report,'seconds':time.perf_counter()-started})
        (folder/'run.json').write_text(json.dumps(report,indent=2));records.append(report)
        (args.output/'suite.json').write_text(json.dumps(records,indent=2));print(report,flush=True)
        if not report['success']:raise RuntimeError('Original baseline failed; see run.log')
    (args.output/'suite.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
