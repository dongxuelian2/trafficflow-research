"""Complete-domain cal mixture/threshold selection with sealed time-order controls."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.queue import QueueModel
from trafficflow.full_domain_queue import FullDomainQueueModel
from trafficflow.full_queue_calibration import tune_full_probabilities,full_queue_decisions
from trafficflow.queue_domain import score_full_forecast
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from compare_queue_events import comparison


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probability-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Never overwrite full Queue calibration')
    cache=json.loads(args.probability_report.read_text())
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'Full_forecast_Queue_calibration_time_order_control','probability_report_sha256':sha256_file(args.probability_report),
        'timing_orders':['after','before'],'ranges':['tree','direct','wave_15min','wave_30min'],
        'alpha_grid':[0.,.25,.5,.75,1.],'threshold_grid':'31 values .05… .80, global or public condition',
        'nonempty':[False,True],'calibration':'Full public domain; future availability exclusively scorer mask',
        'weights':'Frozen source checkpoints chosen previously by per-cell cal Brier; no refit or new training seed',
        'selection':'Best calibration IoU, ties lower alpha then tree; seal both timing orders before assessment load',
        'assessment':'Already inspected public TRAIN periods and same event origins; not final',
        'formal_Queue':None,'formal_Total':None})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    results=[]
    for source in cache['source_records']:
        wave_path=Path(source['report']);study=wave_path.stem;wave=json.loads(wave_path.read_text())
        producer=Path('artifacts/research')/study
        records=[r for r in cache['full_probability_caches'] if r['study']==study]
        def load(record):
            p=Path(record['path'])
            if sha256_file(p)!=record['sha256']:raise ValueError('Full probability cache changed')
            return dict(np.load(p,allow_pickle=False))
        cal_record=next(r for r in records if r['phase']=='selection');cal=load(cal_record)
        candidates={};chosen={}
        for timing in ['after','before']:
            rows=[]
            base=tune_full_probabilities(cal['base'],cal['window'],cal['condition'],cal['lead_minutes'],cal['labels'],cal['score_mask'],timing)
            rows.append({'name':'tree','alpha':0.,'column':None,'tuning':base})
            for column,name in enumerate(['direct','wave_15min','wave_30min']):
                for alpha in [.25,.5,.75,1.]:
                    p=(1-alpha)*cal['base']+alpha*cal['neural'][:,column]
                    tuning=tune_full_probabilities(p,cal['window'],cal['condition'],cal['lead_minutes'],cal['labels'],cal['score_mask'],timing)
                    rows.append({'name':name,'alpha':alpha,'column':column,'tuning':tuning})
            candidates[timing]=rows
            chosen[timing]=max(rows,key=lambda r:(r['tuning']['selected']['calibration_proxy_IoU'],-r['alpha'],r['name']=='tree'))
            print(study,timing,'cal',chosen[timing]['name'],chosen[timing]['alpha'],chosen[timing]['tuning']['selected'],flush=True)
        token=study.split('_',1)[0]
        seal_path=args.output/(token+'_selection_sealed.json')
        seal_path.write_text(json.dumps(chosen,indent=2,allow_nan=False),encoding='utf-8')
        selected_assessment={'dates':wave['dates'],'assessment':[]};assess=[]
        for record in records:
            if record['phase']!='assessment':continue
            data=load(record);metrics={}
            for name,choice in chosen.items():
                p=data['base'] if choice['column'] is None else (1-choice['alpha'])*data['base']+choice['alpha']*data['neural'][:,choice['column']]
                setting=choice['tuning']['selected']
                pred=full_queue_decisions(p,data['window'],data['condition'],data['lead_minutes'],setting['thresholds'],setting['require_nonempty'],setting['timing'])
                metrics[name]=score_full_forecast(data['labels'],pred,data['window'],data['condition'],data['score_mask'])
            selected_assessment['assessment'].append({'seed':record['seed'],'models':metrics});assess.append({'seed':record['seed'],'models':metrics})
        causal=json.loads((Path(source['causal_cache'])/'experiment.json').read_text())
        baseline_path=Path('artifacts/research')/Path(causal['registered']['protocol']['matched_study']).stem/'shared_threshold.pkl'
        baseline=QueueModel.load(baseline_path);models={}
        for timing,choice in chosen.items():
            state=geometry=None
            if choice['column'] is not None:
                state=torch.load(producer/(choice['name']+'.pt'),map_location='cpu',weights_only=True)
                geometry=dict(np.load(producer/(choice['name']+'_geometry.npz')))
            model=FullDomainQueueModel(baseline,cal['links'].tolist(),choice['tuning']['selected'],choice['alpha'],state,geometry)
            path=args.output/(token+'_'+timing+'.pkl');model.save(path);models[timing]={'path':path.as_posix(),'sha256':sha256_file(path)}
        results.append({'source_study':study,'source_wave_report_sha256':sha256_file(wave_path),'dates':wave['dates'],
            'calibration_candidates':candidates,'selected':chosen,'selection_seal_sha256':sha256_file(seal_path),
            'assessment':assess,'before_vs_after':comparison(selected_assessment,'after','before'),
            'models':models,'panel':wave['registered']['protocol']['panel'],'cal_probability_record':cal_record,
            'formal_Queue':None})
    result={'registered':registered,'source_snapshot':snapshot,'studies':results,'seconds':time.perf_counter()-start,
        'source':evidence(),'formal_Queue':None,'formal_Total':None,
        'limitations':'Only frozen checkpoints, one fit seed, old public TRAIN noisy proxy events. No latent labels or new formal score. Existing sampler selects observation-positive horizons, so optional nonempty can still overfit sampling. Both timing choices retain original cal seals.'}
    (args.output/'experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    report.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print([(r['source_study'],r['before_vs_after']) for r in results],flush=True)


if __name__=='__main__':main()
