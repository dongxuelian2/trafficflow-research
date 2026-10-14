"""Replay every exported ensemble policy from independently rebuilt legal history."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.causal_queue import released_origin_rows
from trafficflow.graph_queue import queue_history_input
from trafficflow.queue import QueueModel
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite ensemble deployment evidence')
    run=json.loads((args.study/'experiment.json').read_text())
    source=json.loads(Path('reports/research/C38_full_queue_probability_cache.json').read_text())
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    rows=[]
    for study in run['studies']:
        tag=study['source_study'];records=source['source_records']
        original=next(r for r in records if Path(r['report']).stem==tag)
        cache_dir=Path(original['causal_cache']);old=json.loads((cache_dir/'experiment.json').read_text())
        panel=old['registered']['protocol']['panel'];net,topology=release.network(panel),release.topology(panel)
        rawpaths={partition_date(release,p):p for p in release.files(panel,'train',False)}
        sequence=cache_dir/'selection_411.npz'
        if sha256_file(sequence)!=old['cache_sha256']['selection']:raise ValueError('History cache changed')
        arrays=dict(np.load(sequence,allow_pickle=False))
        record=next(r for r in source['full_probability_caches'] if r['study']==tag and r['phase']=='selection')
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Full probability source changed')
        full=dict(np.load(record['path'],allow_pickle=False));links=full['links'].tolist();width=6*len(links)
        scores={(kind,i):np.load(Path('artifacts/research')/name/(kind+'_cal_probabilities.npz'),allow_pickle=False)['probability']
            for i,name in enumerate(study['source_reports']) for kind in ['bce','soft_iou','lovasz']}
        candidates={c['name']:c for c in study['calibration_candidates']}
        models={};calls={policy:0 for policy in study['exported_models']}
        for policy,row in study['exported_models'].items():
            if sha256_file(row['path'])!=row['sha256']:raise ValueError('Exported ensemble changed')
            models[policy]=QueueModel.load(Path(row['path']))
        for i,origin_text in enumerate(arrays['origins']):
            origin=pd.Timestamp(str(origin_text));token=origin.strftime('%Y-%m-%dT%H:%M:%SZ')
            history=pd.read_parquet(rawpaths[origin.date().isoformat()],filters=[
                ('timestamp','>=',(origin-pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ')),('timestamp','<',token)])
            for key in ['timestamp','link_id','station_id']:history[key]=history[key].astype(str)
            history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
            condition='queue_ongoing' if arrays['ongoing'][i] else 'queue_onset'
            x=queue_history_input(history,links,net,origin,condition)
            if not np.array_equal(x,arrays['x'][i]):raise ValueError('Causal history differs from source')
            times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
            target=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id']).to_frame(index=False)
            offset=slice(i*width,(i+1)*width)
            for policy,model in models.items():
                candidate=candidates[study['policies'][policy]];variant=candidate['variant']
                if variant=='tree':expected=full['base'][offset]
                else:
                    if variant=='old_bce':p=scores['bce',0][offset]
                    elif variant=='new_bce':p=scores['bce',1][offset]
                    else:
                        method=variant.removeprefix('mean_');p=(scores[method,0][offset]+scores[method,1][offset])/2
                    expected=(1-candidate['alpha'])*full['base'][offset]+candidate['alpha']*p
                prediction=model.forecast_probability(history,target,net,topology,origin,condition).queue_score.to_numpy()
                if not np.array_equal(prediction,expected):raise ValueError('Frozen ensemble score differs')
                setting=candidate['setting']
                binary=full_queue_decisions(expected,full['window'][offset],full['condition'][offset],full['lead_minutes'][offset],
                    setting['thresholds'],setting['require_nonempty'],setting['timing'])
                deployed=model.predict_window(history,target,net,topology,origin,condition).queue_pred.to_numpy()
                if not np.array_equal(binary,deployed):raise ValueError('Full-template ensemble decision differs')
                calls[policy]+=1
        rows.append({'source_study':tag,'model_window_calls':calls,'n_cal_windows':len(arrays['origins']),
            'all_ensemble_scores_exact':True,'all_full_template_decisions_exact':True,'all_fresh_legal_inputs_exact':True})
        del models
    result={'source_experiment_sha256':sha256_file(args.study/'experiment.json'),'source_snapshot':snapshot,
        'studies':rows,'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'formal_Queue':None,'formal_Total':None,'scope':'Every saved policy on all legal calibration windows; no future values/quality/labels in forecast API. Exact replay,not score evidence.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'experiment.json').write_text(text,encoding='utf-8')
    print('scores_and_decisions_exact',sum(sum(r['model_window_calls'].values()) for r in rows),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
