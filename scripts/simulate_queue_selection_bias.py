"""Known-truth synthetic stress of latent-positive vs scored-positive sampling.

Not the organizer generator, actual traffic data, or a formal Queue score.
Forecast probabilities are known simulation marginals, not sampled truth/masks.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.runtime import capture_source_snapshot
from trafficflow.research_eval import register_protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite synthetic selection stress')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Synthetic_Queue_horizon_selection_eligibility_bias','nodes':32,'onset_steps':6,
        'block_sizes':[1,4,16],'eligible_probabilities':[.5,.75,.95],'seeds':[941831,941839,941843],
        'events_per_case':5000,'latent_truth':'One uniformly located contiguous final-step queue block; early five steps empty',
        'probability_input':'Exact marginal over uniform block start, computed before event sampling; never individual truth or mask',
        'populations':['latent_positive','same_events_conditioned_scored_positive'],
        'decisions':['threshold.5','full_template_nonempty','future_eligibility_oracle_nonempty_for_diagnostic_only'],
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');rows=[]
    nodes=32;samples=5000
    for seed in [941831,941839,941843]:
        for width in [1,4,16]:
            starts_count=nodes-width+1
            marginal=np.array([sum(start<=j<start+width for start in range(starts_count))/starts_count for j in range(nodes)])
            base=marginal>=.5;forced=base.copy()
            if not forced.any():forced[np.argmax(marginal)]=True
            for rate in [.5,.75,.95]:
                rng=np.random.default_rng(seed+width*1000+round(rate*100))
                starts=rng.integers(0,starts_count,samples)
                truth=np.zeros((samples,6,nodes),bool)
                truth[:,5]=((np.arange(nodes)[None]>=starts[:,None])&(np.arange(nodes)[None]<starts[:,None]+width))
                eligible=rng.random(truth.shape)<rate
                prediction_base=np.zeros_like(truth);prediction_base[:,5]=base
                prediction_force=np.zeros_like(truth);prediction_force[:,5]=forced
                # This intentionally illegal forecast is a separate diagnostic:
                # it receives future eligibility and must not be deployed.
                oracle=prediction_base.copy()
                for i in range(samples):
                    if not (oracle[i]&eligible[i]).any():
                        available=np.flatnonzero(eligible[i,5])
                        if len(available):oracle[i,5,available[np.argmax(marginal[available])]]=True
                observed_positive=(truth&eligible).any(axis=(1,2))
                for name,pred in [('threshold.5',prediction_base),('full_template_nonempty',prediction_force),('future_eligibility_oracle_nonempty',oracle)]:
                    inter=(truth&pred&eligible).sum(axis=(1,2));union=((truth|pred)&eligible).sum(axis=(1,2))
                    score=np.divide(inter,union,out=np.ones(samples,float),where=union>0);score[~eligible.any(axis=(1,2))]=0.
                    rows.append({'seed':seed,'block_size':width,'future_eligible_probability':rate,'decision':name,
                        'n_events':samples,'n_scored_positive_events':int(observed_positive.sum()),
                        'eligible_empty_truth_fraction':float((~observed_positive).mean()),
                        'synthetic_IoU_latent_positive_population':float(score.mean()),
                        'synthetic_IoU_scored_positive_population':float(score[observed_positive].mean()),
                        'input_diagnostic_oracle':name.startswith('future_eligibility_oracle'),
                        'formal_Queue':None})
    result={'registered':registered,'source_snapshot':snapshot,'rows':rows,
        'formal_Queue':None,'formal_Total':None,
        'limitations':'Independent synthetic law, known marginals, independent quality masks. Not actual organizer selection, traffic forecasts or latent competition labels. Population-conditioned absolute scores are not paired algorithm gains; compare rules within each population. Oracle is diagnostic-only.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('synthetic_rule_case_rows',len(rows),'event_cases',27,flush=True)


if __name__=='__main__':main()
