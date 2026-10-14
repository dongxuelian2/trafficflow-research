"""Descriptive matched-scene summaries; population changes are not model gains."""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.runtime import sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite diagnostic summaries')
    source=json.loads(args.source.read_text());rows=source['rows'];totals=[];means=[]
    for sigma in [0.,1.,3.,8.]:
        selected=[r for r in rows if r['noise_kmh']==sigma and r['population']=='latent_selected']
        subset=[r for r in rows if r['noise_kmh']==sigma and r['population']=='observation_subset']
        eligible=sum(r['future_eligible_cells'] for r in selected)
        totals.append({'noise_kmh':sigma,'n_latent_selected_overlapping_events':sum(r['n_events'] for r in selected),
            'n_observation_subset_events':sum(r['n_events'] for r in subset),
            'n_zero_observation_positive_events':sum(r['n_zero_observed_positive'] for r in selected),
            'future_threshold_label_disagreement_fraction':sum(r['future_disagreement_cells'] for r in selected)/eligible if eligible else None,
            'eligible_cells_with_repeated_origins':eligible})
        for population in ['latent_selected','observation_subset']:
            for method in ['persistence','linear_trend']:
                for condition in ['queue_onset','queue_ongoing']:
                    use=[r for r in rows if r['noise_kmh']==sigma and r['population']==population
                         and r['methods'][method]['known_latent_IoU'][condition] is not None]
                    means.append({'noise_kmh':sigma,'population':population,'method':method,'condition':condition,
                        'n_scene_settings':len(use),
                        'mean_scene_known_latent_IoU':float(np.mean([r['methods'][method]['known_latent_IoU'][condition] for r in use])) if use else None,
                        'mean_scene_noisy_observation_IoU':float(np.mean([r['methods'][method]['noisy_observation_IoU'][condition] for r in use])) if use else None})
    # Counterfactual noise arms must preserve exact latent and quality arrays.
    for seed in source['registered']['protocol']['seeds']:
        for case in source['registered']['protocol']['cases']:
            group=[r for r in rows if r['seed']==seed and r['scenario']==case and r['population']=='latent_selected']
            if len(group)!=4 or len({r['latent_quality_sha256'] for r in group})!=1:
                raise ValueError('Unmatched noise counterfactual')
            if len({json.dumps(r['condition_counts'],sort_keys=True) for r in group})!=1:
                raise ValueError('Latent event population changed with observation noise')
    result={'source_report_sha256':sha256_file(args.source),'population_totals':totals,'scene_condition_means':means,
        'all_18_latent_quality_counterfactuals_exact':True,'formal_Queue':None,'formal_Total':None,
        'scope':'Own CTM diagnostics only. Equal means over available scene/condition settings;17 settings have events,only3 independent seeds. Repeated origins overlap. Comparing populations changes evidence,not a model-gain experiment. No confidence interval or formal-score claim.'}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(totals,flush=True)


if __name__=='__main__':main()
