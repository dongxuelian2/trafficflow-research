"""Separate synthetic composite-score gains from path L1 and L2 diagnostics."""
import argparse
import json
from pathlib import Path
import pandas as pd
from trafficflow.runtime import sha256_file
from entropy_od_experiment import macro,interval
from zero_aware_od_experiment import NINE
from od_shift_generators import NEW_REGIMES


WEIGHTS={'simulated_S_od':.45,'simulated_S_link':.25,'simulated_S_dev':.15,'simulated_S_attr':.15}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite component evidence')
    results=[];studies=[]
    for path in args.study:
        study=json.loads(path.read_text());studies.append(study);frame=pd.DataFrame(study['assessment']);populations={}
        for population,regimes in [('same_laws',NINE),('new_shift_laws',NEW_REGIMES)]:
            subset=frame[frame.regime.isin(regimes)];summary={}
            for method in sorted(subset.method.unique()):
                data=subset[subset.method==method]
                if not data.success.all():summary[method]={'complete':False};continue
                summary[method]={'complete':True,**{k:macro(data,k) for k in list(WEIGHTS)+['simulated_S_ODME','relative_path_L1']}}
                if abs(sum(WEIGHTS[k]*summary[method][k] for k in WEIGHTS)-summary[method]['simulated_S_ODME'])>1e-12:
                    raise ValueError('Weighted component macro does not add to composite')
            reference=summary['fixed_entropy_D08'];decompositions={}
            for method,values in summary.items():
                if method=='fixed_entropy_D08' or not values['complete'] or not reference['complete']:continue
                contributions={k:WEIGHTS[k]*(values[k]-reference[k]) for k in WEIGHTS}
                gain=values['simulated_S_ODME']-reference['simulated_S_ODME']
                if abs(sum(contributions.values())-gain)>1e-12:raise ValueError('Gain components do not add')
                decompositions[method]={'gain':gain,'weighted_component_gains':contributions,
                    'mean_relative_path_L1_reduction':reference['relative_path_L1']-values['relative_path_L1']}
            populations[population]={'macro_simulated_components':summary,'gain_decompositions_vs_entropy':decompositions}
        results.append({'study':path.as_posix(),'study_sha256':sha256_file(path),'populations':populations})
    configs=[{'residual_floor':2.}|s['selection']['best_development_hypothesis']['settings'] for s in studies]
    pooled=None
    if len(studies)>1 and all(config==configs[0] for config in configs):
        combined=[r for s in studies for r in s['assessment']];frame=pd.DataFrame(combined)
        if frame.duplicated(['panel','seed','regime','method']).any():raise ValueError('Pooled study duplicates an assessment case')
        seeds=sorted(frame.seed.unique().tolist());families=sorted({p.rsplit('_',1)[0] for p in frame.panel})
        pooled={'same_hypothesis_settings':configs[0],'n_fresh_research_seeds':len(seeds),'comparisons':{}}
        for population,regimes in [('same_laws',NINE),('new_shift_laws',NEW_REGIMES)]:
            rows=[r for r in combined if r['regime'] in regimes]
            pooled['comparisons'][population]=interval(rows,'fixed_entropy_D08','development_hypothesis',seeds,families)
        pooled['scope']='Secondary pooled six-seed diagnostic of identical development-selected hypotheses;no change to either failed development gate.'
    result={'studies':results,'pooled_identical_hypothesis':pooled,'formal_ODME':None,'formal_Total':None,
        'scope':'Same frozen research predictions;independent diagnostic,not model choice or new performance evidence. '
        'L1 path recovery and squared L2 error need not rank models equally;SVD diagnostic is reported separately.'}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print('studies',len(results),'component_additivity_checked',True,flush=True)


if __name__=='__main__':main()
