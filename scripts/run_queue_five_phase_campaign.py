"""Register four Queue phase studies before any assessment,then run sequentially."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
from trafficflow.runtime import sha256_file
from trafficflow.research_eval import register_protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--balanced-v2',action='store_true');args=parser.parse_args()
    jobs=[('C55_I5_S_autumn_five_phase','D12_I5_S','2030-10-15'),
          ('C56_I5_S_winter_five_phase','D12_I5_S','2030-11-30'),
          ('C57_I10_W_autumn_five_phase','D7_I10_W','2030-10-15'),
          ('C58_I10_W_winter_five_phase','D7_I10_W','2030-11-30')]
    if args.balanced_v2:jobs=[(name+'_balanced_v2',panel,end) for name,panel,end in jobs]
    token='C55_C58_five_phase_campaign'+('_balanced_v2' if args.balanced_v2 else '')
    path=Path('reports/research')/(token+'.registered.json')
    if path.exists():raise FileExistsError('Campaign already registered; inspect jobs rather than blindly restart')
    register_protocol(path,{'id':'Four_new_direction_five_phase_Queue_studies','jobs':[
        {'name':name,'panel':panel,'fit_end':end} for name,panel,end in jobs],
        'worker_source_sha256':sha256_file('scripts/queue_five_phase_experiment.py'),
        'sampler_source_sha256':sha256_file('src/trafficflow/queue_events.py'),'balanced_condition_sampling':args.balanced_v2,
        'scope':'Two directions,two temporal fits,three optimizer seeds per study. Fit/checkpoint/decision/selection/assessment separated.',
        'rule':'Same preregistered worker/parameters,no adjustment based on earlier campaign assessments',
        'platform':'Local only,no submission/push/publication','final_seeds_generated':False,'formal_Queue':None,'formal_Total':None})
    results=[]
    for name,panel,end in jobs:
        command=[sys.executable,'scripts/queue_five_phase_experiment.py','--panel',panel,'--fit-end',end,
                 '--output','artifacts/research/'+name]
        if args.balanced_v2:command+=['--balanced-sampling']
        print('starting',name,flush=True)
        code=subprocess.run(command,check=False).returncode
        results.append({'name':name,'exit_code':code})
        if code:
            print('terminal child failure',name,code,flush=True)
            if not args.balanced_v2:break
    output=Path('reports/research')/(token+'.status.json')
    with output.open('x',encoding='utf-8') as stream:
        json.dump({'registered_sha256':sha256_file(path),'terminal_children':results,
            'complete':len(results)==len(jobs) and all(r['exit_code']==0 for r in results),
            'formal_Queue':None,'formal_Total':None},stream,indent=2,allow_nan=False)
    if len(results)!=len(jobs) or any(r['exit_code'] for r in results):sys.exit(1)


if __name__=='__main__':main()
