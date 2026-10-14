"""Rebuild lawful histories and replay saved raw/portable overlap predictions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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
from trafficflow.graph_queue import CausalGraphQueue, queue_history_input
from trafficflow.queue import QueueModel
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.probability_calibration import weighted_bce_probability
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite deployment replays')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    source=json.loads(Path('reports/research/C38_full_queue_probability_cache.json').read_text())
    release=Release('data/kaggle_public');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    rows=[]
    for path in args.reports:
        run=json.loads(path.read_text());producer=Path('artifacts/research')/path.stem
        config=run['registered']['protocol'];panel=config['panel'];study=config['source_study']
        src=next(r for r in source['source_records'] if Path(r['report']).stem==study)
        old=json.loads((Path(src['causal_cache'])/'experiment.json').read_text())
        old_arrays_path=Path(src['causal_cache'])/'selection_411.npz'
        if sha256_file(old_arrays_path)!=old['cache_sha256']['selection']:raise ValueError('Source history cache changed')
        old_arrays=dict(np.load(old_arrays_path,allow_pickle=False))
        record=next(r for r in source['full_probability_caches'] if r['study']==study and r['phase']=='selection')
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Source full-domain cache changed')
        full=dict(np.load(record['path'],allow_pickle=False));links=full['links'].tolist()
        net,topology=release.network(panel),release.topology(panel)
        rawpaths={partition_date(release,p):p for p in release.files(panel,'train',False)}
        geometry=dict(np.load(Path('artifacts/research')/study/'wave_30min_geometry.npz'))
        graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']]
        models={};raw_models={};probabilities={};n_calls={k:0 for k in run['training']}
        for kind in run['training']:
            model_path=producer/(kind+'.pkl')
            if sha256_file(model_path)!=run['exported_models'][kind]['sha256']:raise ValueError('Portable model changed')
            pt=producer/(kind+'.pt')
            if sha256_file(pt)!=run['training'][kind]['checkpoint_sha256']:raise ValueError('Neural checkpoint changed')
            models[kind]=QueueModel.load(model_path)
            network=CausalGraphQueue(hidden=32,use_graph=True).to(device)
            network.load_state_dict(torch.load(pt,map_location=device,weights_only=True));network.eval();raw_models[kind]=network
            probabilities[kind]=np.load(producer/(kind+'_cal_probabilities.npz'),allow_pickle=False)['probability'].reshape(-1,6,len(links))
        for i,origin_text in enumerate(old_arrays['origins']):
            origin=pd.Timestamp(str(origin_text));token=origin.strftime('%Y-%m-%dT%H:%M:%SZ')
            history=pd.read_parquet(rawpaths[origin.date().isoformat()],filters=[
                ('timestamp','>=',(origin-pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ')),('timestamp','<',token)])
            for key in ['timestamp','link_id','station_id']:history[key]=history[key].astype(str)
            history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
            condition='queue_ongoing' if old_arrays['ongoing'][i] else 'queue_onset'
            x=queue_history_input(history,links,net,origin,condition)
            if not np.array_equal(x,old_arrays['x'][i]):raise ValueError('Fresh causal input does not match source cache')
            times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
            target=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id']).to_frame(index=False)
            width=6*len(links);offset=slice(i*width,(i+1)*width)
            for kind,model in models.items():
                with torch.no_grad():
                    logits,_=raw_models[kind](torch.from_numpy(x.transpose(2,0,1)[None].copy()).to(device),*graph)
                    p=weighted_bce_probability(logits.cpu().numpy()[0],4.)
                if not np.array_equal(p,probabilities[kind][i]):raise ValueError('Raw neural cal probabilities differ')
                choice=next(r for r in run['selection'][kind]['choices'] if r['alpha']==model.alpha)
                setting=choice['setting'];mixed=(1-model.alpha)*full['base'][offset]+model.alpha*p.ravel()
                expected=full_queue_decisions(mixed,full['window'][offset],full['condition'][offset],full['lead_minutes'][offset],
                    setting['thresholds'],setting['require_nonempty'],setting['timing'])
                deployed=model.predict_window(history,target,net,topology,origin,condition).queue_pred.to_numpy()
                if not np.array_equal(deployed,expected):raise ValueError('Portable full-template prediction differs')
                n_calls[kind]+=1
        original_control=None
        if config['training_seed']==old['registered']['protocol']['seed']:
            original=torch.load(Path('artifacts/research')/study/'wave_30min.pt',map_location='cpu',weights_only=True)
            fresh=torch.load(producer/'bce.pt',map_location='cpu',weights_only=True)
            original_control={'original_checkpoint_sha256':sha256_file(Path('artifacts/research')/study/'wave_30min.pt'),
                'all_tensors_exact':set(original)==set(fresh) and all(torch.equal(original[k],fresh[k]) for k in original)}
            if not original_control['all_tensors_exact']:raise ValueError('Original seed BCE training control differs')
        rows.append({'source_report':path.as_posix(),'source_report_sha256':sha256_file(path),
            'n_cal_windows':len(old_arrays['origins']),'model_window_calls':n_calls,
            'all_raw_neural_probabilities_exact':True,'all_portable_full_template_decisions_exact':True,
            'fresh_legal_history_inputs_exact':True,'original_seed_training_control':original_control})
        del models,raw_models
    result={'source_snapshot':snapshot,'studies':rows,'seconds':time.perf_counter()-start,
        'source':evidence(gpu_used=device.type=='cuda'),'formal_Queue':None,'formal_Total':None,
        'scope':'All calibration windows independently rebuilt from TRAIN history <T and public masked measurements at T; no future values in inference. Probability/decision replay and original-seed tensor controls, not hidden-score evidence.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'experiment.json').write_text(text,encoding='utf-8')
    print('exact raw and portable calls',sum(sum(r['model_window_calls'].values()) for r in rows),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
