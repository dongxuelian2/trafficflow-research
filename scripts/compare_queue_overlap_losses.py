"""Compare frozen overlap arms at identical alphas, beyond cal fallback outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.queue_domain import score_full_forecast
from trafficflow.runtime import sha256_file, capture_source_snapshot
from trafficflow.research_eval import register_protocol
from compare_queue_events import comparison


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite fixed-alpha evidence')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Frozen_Queue_overlap_equal_alpha_comparison','source_reports':{p.as_posix():sha256_file(p) for p in args.reports},
        'alphas':[.25,1.],'selection':'Use each already-sealed calibration setting at that alpha; no retuning',
        'scope':'Supplemental loss-arm diagnostics on already inspected TRAIN development periods; not primary or blind',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip')
    probability_path=Path('reports/research/C38_full_queue_probability_cache.json')
    source=json.loads(probability_path.read_text());results=[];inputs={}
    for path in args.reports:
        run=json.loads(path.read_text());study=run['registered']['protocol']['source_study']
        producer=Path('artifacts/research')/path.stem;calibration=run['selection'];assessment={'dates':run['dates'],'assessment':[]}
        for record in source['full_probability_caches']:
            if record['study']!=study or record['phase']!='assessment':continue
            if sha256_file(record['path'])!=record['sha256']:raise ValueError('Full probability cache changed')
            data=dict(np.load(record['path'],allow_pickle=False));metrics={}
            for kind in ['bce','soft_iou','lovasz']:
                p_path=producer/(kind+'_assessment_'+str(record['seed'])+'_probabilities.npz')
                p=np.load(p_path,allow_pickle=False)['probability'];inputs[p_path.as_posix()]=sha256_file(p_path)
                if p.shape!=data['labels'].shape or not np.isfinite(p).all():raise ValueError('Prediction domain mismatch')
                for alpha in [.25,1.]:
                    choice=next(r for r in calibration[kind]['choices'] if r['alpha']==alpha);setting=choice['setting']
                    mixed=(1-alpha)*data['base']+alpha*p
                    pred=full_queue_decisions(mixed,data['window'],data['condition'],data['lead_minutes'],
                        setting['thresholds'],setting['require_nonempty'],setting['timing'])
                    name=kind+'_'+str(alpha)
                    metric=score_full_forecast(data['labels'],pred,data['window'],data['condition'],data['score_mask'])
                    # Condition/event macro Brier uses the known-allowed onset step only.
                    allowed=(data['condition']=='queue_ongoing')|(data['lead_minutes']==30)
                    briers=[]
                    for condition in ['queue_onset','queue_ongoing']:
                        values=[]
                        for window in np.unique(data['window'][data['condition']==condition]):
                            use=(data['window']==window)&data['score_mask']&allowed
                            if use.any():values.append(float(((mixed[use]-data['labels'][use])**2).mean()))
                        briers.append(float(np.mean(values)))
                    metric['condition_macro_event_score_Brier']=float(np.mean(briers));metrics[name]=metric
            assessment['assessment'].append({'seed':record['seed'],'models':metrics})
        results.append({'source_report':path.as_posix(),'training_seed':run['registered']['protocol']['training_seed'],
            'source_study':study,'primary_selected':run['selected'],'primary_comparisons':run['comparisons'],
            'fixed_alpha_comparisons':{kind+'_'+str(alpha):comparison(assessment,'bce_'+str(alpha),kind+'_'+str(alpha))
                for kind in ['soft_iou','lovasz'] for alpha in [.25,1.]},
            'assessment':assessment['assessment'],'formal_Queue':None})
    result={'registered':registered,'source_snapshot':snapshot,'probability_report_sha256':sha256_file(probability_path),
        'prediction_archives_sha256':inputs,'studies':results,'formal_Queue':None,'formal_Total':None,
        'limitations':'Same alpha controls isolate ensemble strength but cal thresholds/checkpoints can differ. Fixed-alpha metrics are secondary and never override cal-selected primary. One direction, two periods, already inspected dates; no formal gain claim.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'experiment.json').write_text(text,encoding='utf-8')
    for row in results:print(Path(row['source_report']).stem,{k:(v['paired_macro_gain'],v['ci95']) for k,v in row['fixed_alpha_comparisons'].items()},flush=True)


if __name__=='__main__':main()
