"""Exact sampling-support audit using fit masks only,without evaluator labels."""
import json
from pathlib import Path
import numpy as np
import torch
from trafficflow.recurrent_state import training_window
from trafficflow.runtime import sha256_file


def main():
    report=Path('reports/research/A78_recurrent_supervision_coverage.json')
    if report.exists():raise FileExistsError('Never overwrite the coverage audit')
    rows=[]
    for cache in [Path('artifacts/research/A48_D7_I210_W_autumn_cache'),Path('artifacts/research/A02_D12_I5_N_winter_seed20261013')]:
        by_regime={};n_days=0
        for path in sorted((cache/'fit').glob('*.npz')):
            with np.load(path) as day:
                mask=day['mask'];length=len(mask);old=np.zeros(length,int);new=np.zeros(length,int)
                for left in range(length-128+1):old[left+32:left+96]+=1
                for center in range(length):
                    left,start,end=training_window(length,center,128,64);new[start:end]+=1
                if not (new>0).all():raise ValueError('Revised sampler still excludes day cells')
                regime=str(day['regime']);entry=by_regime.setdefault(regime,{'n_target_grid_cells':0,'old_never_supervised':0,'new_never_supervised':0})
                entry['n_target_grid_cells']+=int(mask.sum());entry['old_never_supervised']+=int(mask[old==0].sum())
                entry['new_never_supervised']+=int(mask[new==0].sum());n_days+=1
        total=sum(r['n_target_grid_cells'] for r in by_regime.values());excluded=sum(r['old_never_supervised'] for r in by_regime.values())
        rows.append({'cache':cache.as_posix(),'prepared_sha256':sha256_file(cache/'prepared.json'),'n_fit_days':n_days,
            'n_target_grid_cells':total,'old_never_supervised':excluded,'old_never_supervised_fraction':excluded/total,
            'new_never_supervised':0,'by_regime':by_regime})
    result={'rows':rows,'old_policy':'Uniform context left128,supervise fixed center64;time0:32and256:288excluded',
        'revised_policy':'Uniform supervised center;clip full core64and containing context128 at day edges',
        'scope':'Positive theoretical sampling probability,not a claim every target actually drawn. Each model saves actual per-cell exposure counts.',
        'runtime_shared_environment':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'original_trial_stop_record_sha256':sha256_file('reports/research/A72_restricted_supervision_stop.json'),
        'formal_State':None,'formal_Physics':None,'formal_Total':None}
    with report.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(rows,flush=True)


if __name__=='__main__':main()
