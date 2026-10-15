"""Matched synthetic controls and weakly identified local-solution diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.odme import operator
from trafficflow.runtime import sha256_file
from entropy_od_experiment import macro,interval


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--numeric',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite matched model evidence')
    study=json.loads(args.study.read_text());numeric=json.loads(args.numeric.read_text())
    rows=study['assessment'];frame=pd.DataFrame(rows)
    methods=['ridge_0.05','fixed_entropy_D08','development_hypothesis']
    keys=['panel','seed','regime'];expected=None
    for method in methods:
        subset=frame[frame.method==method]
        if not subset.success.all() or subset.duplicated(keys).any():raise ValueError('Failed/duplicate matched cases')
        current=set(map(tuple,subset[keys].to_numpy()))
        if expected is not None and current!=expected:raise ValueError('Incomplete paired cases')
        expected=current
    seeds=study['registered']['protocol']['fresh_assessment_seeds']
    families=sorted({p.rsplit('_',1)[0] for p in frame.panel})
    regimes=study['registered']['protocol']['generating_regimes'];summary={};regime_rows=[]
    for method in methods:summary[method]=macro(frame[frame.method==method],'simulated_S_ODME')
    comparisons={baseline:interval(rows,baseline,'development_hypothesis',seeds,families)
        for baseline in ['ridge_0.05','fixed_entropy_D08']}
    for regime in regimes:
        subset=frame[frame.regime==regime]
        values={method:macro(subset[subset.method==method],'simulated_S_ODME') for method in methods}
        regime_rows.append({'regime':regime,**values,'gain_vs_ridge':values['development_hypothesis']-values['ridge_0.05'],
            'gain_vs_entropy':values['development_hypothesis']-values['fixed_entropy_D08']})
    folder=Path('artifacts/research')/args.numeric.stem;diagnostics=[];release=Release('data/kaggle_public')
    for row in numeric['rows']:
        if row['independent_solutions_agree']:continue
        filename=folder/(row['panel']+'_'+row['regime']+'_'+str(row['setting_index'])+'.npz')
        with np.load(filename) as arrays:
            f,g=arrays['solver'],arrays['oracle'];A,_,_,_=operator(release,row['panel'],'train')
            diagnostics.append({**row,'NPZ_sha256':sha256_file(filename),
                'relative_path_L1_between_solutions':float(np.abs(f-g).sum()/max(np.abs(g).sum(),1e-12)),
                'relative_count_L1_between_solutions':float(np.abs(A@(f-g)).sum()/max(np.abs(A@g).sum(),1e-12)),
                'maximum_absolute_path_flow_difference':float(np.abs(f-g).max())})
    result={'study_sha256':sha256_file(args.study),'numeric_sha256':sha256_file(args.numeric),
        'n_matched_unique_simulated_cases':len(expected),'macro_simulated_S_ODME':summary,
        'paired_comparisons':comparisons,'regime_comparisons':regime_rows,'local_solver_disagreements':diagnostics,
        'model_selection_changed':False,'formal_ODME':None,'formal_Total':None,
        'limitations':'Secondary matched diagnostic after numerical repair on previously viewed research seeds. '
        'Intervals resample five fixed public families and three seeds under nine chosen synthetic laws; '
        'not independent formal/private accuracy,not calibrated posterior uncertainty. '
        'Fixed entropy is the current public-input K05 branch,not all possible synthetic OD expert policies.'}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print('macro',summary,'paired',comparisons,'disagreements',len(diagnostics),flush=True)


if __name__=='__main__':main()
