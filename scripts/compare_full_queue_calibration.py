"""Compare the full-domain selected mixtures to equally calibrated tree controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.queue_domain import score_full_forecast
from trafficflow.runtime import sha256_file
from compare_queue_events import comparison


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--probability-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite full Queue comparisons')
    source=args.study/'experiment.json';run=json.loads(source.read_text());cache=json.loads(args.probability_report.read_text());results=[]
    for study in run['studies']:
        assembled={'dates':study['dates'],'assessment':[]}
        tree=next(r for r in study['calibration_candidates']['before'] if r['name']=='tree')['tuning']['selected']
        for previous in study['assessment']:
            record=next(r for r in cache['full_probability_caches'] if r['study']==study['source_study'] and r['phase']=='assessment' and r['seed']==previous['seed'])
            path=Path(record['path'])
            if sha256_file(path)!=record['sha256']:raise ValueError('Probability cache changed')
            d=dict(np.load(path,allow_pickle=False))
            pred=full_queue_decisions(d['base'],d['window'],d['condition'],d['lead_minutes'],tree['thresholds'],tree['require_nonempty'],tree['timing'])
            metric=score_full_forecast(d['labels'],pred,d['window'],d['condition'],d['score_mask'])
            assembled['assessment'].append({'seed':previous['seed'],'models':{'tree':metric,'mixture':previous['models']['before']}})
        results.append({'source_study':study['source_study'],'tree_cal_settings':tree,'selected_mixture':study['selected']['before'],
            'mixture_vs_equally_calibrated_tree':comparison(assembled,'tree','mixture'),
            'timing_before_vs_after':study['before_vs_after'],'formal_Queue':None})
    output={'source_experiment_sha256':sha256_file(source),'source_probability_report_sha256':sha256_file(args.probability_report),
        'studies':results,'formal_Queue':None,'formal_Total':None,
        'limitations':'Full forecast-domain calibration for both controls, identical events and frozen fit weights. One fit seed and old public-train periods. Neural richer inputs/graph plus mixture and decision settings; not an architecture-only contrast. No latent truth or final score.'}
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print([(r['source_study'],r['mixture_vs_equally_calibrated_tree']) for r in results],flush=True)


if __name__=='__main__':main()
