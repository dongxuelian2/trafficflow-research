"""Known-latent CTM Queue sampling/measurement bias, independent of organizer."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.digital_twin import Scenario,simulate
from trafficflow.ctm_queue_diagnostics import causal_speed_queue_forecasts
from trafficflow.metrics import queue_iou
from trafficflow.runtime import capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from trafficflow.ctm_scenarios import CASES


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite known-truth diagnostics')
    seeds=[943003,943013,943031];noise=[0.,1.,3.,8.]
    base=Scenario(seed=seeds[0],days=2,mask_rate=0.)
    cases={name:{k:v for k,v in values.items() if k!='speed_noise'} for name,values in CASES.items()}
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Independent_CTM_latent_vs_observed_Queue_event_diagnostic','seeds':seeds,'noise_kmh':noise,
        'base':asdict(base),'cases':cases,'final_seeds_generated':False,
        'latent_selector':'History visible latent>=2 same-link defines ongoing; else onset. Latent queue required in horizon; onset first occurrence T+30; latent persistence IoU<=.9; history quality>=.7.',
        'observation_subset':'Within the same latent-selected events, require observed eligible future positive, onset first observed at T+30, and observed persistence IoU<=.9',
        'prediction':'Two fixed baselines see only thirteen noisy past/current slots and history quality. No future quality/latent inputs.',
        'scores':'Known latent and noisy-threshold labels scored after predictions, same future eligible mask. Separate all-latent-selected versus observation-subset populations.',
        'sampling_limit':'All qualifying five-minute origins overlap; scene/seed is the evidence unit, no per-origin independent CI',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    rows=[];events=[];previous={}
    for seed in seeds:
        for case,config in cases.items():
            for sigma in noise:
                sim=simulate(replace(base,seed=seed,speed_noise=sigma,**config))
                latent=sim['latent_speed'];observed=sim['masked_speed'];quality=sim['eligible']
                truth=sim['queue_truth'];threshold=.6*base.free_speed
                digest=hashlib.sha256(latent.tobytes()+quality.tobytes()).hexdigest()
                key=(seed,case)
                if key in previous and previous[key]!=digest:raise ValueError('Noise counterfactual changed latent dynamics or quality')
                previous[key]=digest;chosen=[]
                for at in range(12,len(latent)-6):
                    past=slice(at-12,at);future=slice(at+1,at+7)
                    if quality[past].mean()<.7:continue
                    latent_visible=truth[past]&quality[past]
                    condition='queue_ongoing' if (latent_visible.sum(axis=0)>=2).any() else 'queue_onset'
                    yf=truth[future];mask=quality[future]
                    if not yf.any() or (condition=='queue_onset' and (yf[:5].any() or not yf[5].any())):continue
                    last=np.full(base.cells,base.free_speed)
                    valid_past=quality[past]
                    for i in range(12):last=np.where(valid_past[i],latent[past][i],last)
                    lp=np.broadcast_to(last<=threshold,yf.shape)
                    if queue_iou(yf.ravel(),lp.ravel())>.9:continue
                    # Forecast before reading future observation labels or quality.
                    predictions=causal_speed_queue_forecasts(observed[at-12:at+1],quality[at-12:at+1],base.free_speed,condition)
                    label=observed[future]<=threshold
                    valid=mask&np.isfinite(observed[future])
                    has_observed=bool((label&valid).any())
                    onset_guard=condition!='queue_onset' or (not (label[:5]&valid[:5]).any() and (label[5]&valid[5]).any())
                    obs_last=np.full(base.cells,base.free_speed)
                    for i in range(12):
                        use=quality[past][i]&np.isfinite(observed[past][i]);obs_last=np.where(use,observed[past][i],obs_last)
                    op=np.broadcast_to(obs_last<=threshold,label.shape)
                    observed_persistence_iou=queue_iou(label[valid],op[valid]) if valid.any() else 0.
                    subset=bool(has_observed and onset_guard and observed_persistence_iou<=.9)
                    methods={name:{'known_latent_IoU':queue_iou(yf[valid],pred[valid]) if valid.any() else 0.,
                                   'noisy_observation_IoU':queue_iou(label[valid],pred[valid]) if valid.any() else 0.}
                             for name,pred in predictions.items()}
                    chosen.append({'at':at,'condition':condition,'observation_subset':subset,
                        'zero_observed_positive':not has_observed,'onset_observation_guard_passes':bool(onset_guard),
                        'future_label_disagreement_cells':int(((label!=yf)&valid).sum()),'n_future_eligible_cells':int(valid.sum()),
                        'methods':methods})
                for population in ['latent_selected','observation_subset']:
                    group=chosen if population=='latent_selected' else [e for e in chosen if e['observation_subset']]
                    metrics={}
                    for method in ['persistence','linear_trend']:
                        metrics[method]={field:{condition:float(np.mean([e['methods'][method][field] for e in group if e['condition']==condition]))
                            if any(e['condition']==condition for e in group) else None for condition in ['queue_onset','queue_ongoing']}
                            for field in ['known_latent_IoU','noisy_observation_IoU']}
                    rows.append({'seed':seed,'scenario':case,'noise_kmh':sigma,'population':population,'n_events':len(group),
                        'condition_counts':{c:sum(e['condition']==c for e in group) for c in ['queue_onset','queue_ongoing']},
                        'methods':metrics,'n_zero_observed_positive':sum(e['zero_observed_positive'] for e in group),
                        'future_disagreement_cells':sum(e['future_label_disagreement_cells'] for e in group),
                        'future_eligible_cells':sum(e['n_future_eligible_cells'] for e in group),'latent_quality_sha256':digest,
                        'max_conservation_error_vehicles':sim['max_conservation_error_vehicles'],'formal_Queue':None})
                events.append({'seed':seed,'scenario':case,'noise_kmh':sigma,'events':chosen})
            print(seed,case,'four matched noise levels',flush=True)
    archive=folder/'events.json';archive.write_text(json.dumps(events,indent=2,allow_nan=False),encoding='utf-8')
    result={'registered':registered,'source_snapshot':snapshot,'rows':rows,'seconds':time.perf_counter()-start,
        'source':evidence(),'formal_Queue':None,'formal_Total':None,
        'limitations':'Own CTM law,not the organizer generator. Latent labels define simulation event selection only. Forecasters use past noisy observations; public-model weights are not tested or refit. Overlapping windows and18 matched seed/scenario settings driven by only3 independent seeds; no official score or architecture-gain claim.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'experiment.json').write_text(text,encoding='utf-8');print('scenes',len(events),'rows',len(rows),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
