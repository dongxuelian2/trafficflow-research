"""Verify all decision-phase deployment roles independently of cached features."""
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
from trafficflow.graph_queue import CausalGraphQueue,queue_history_input
from trafficflow.probability_calibration import weighted_bce_probability
from trafficflow.queue import QueueModel,queue_features
from trafficflow.full_queue_calibration import full_queue_decisions
from trafficflow.queue_phase_split import queue_phase_dates
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite five-phase replay')
    source=args.study/'experiment.json';run=json.loads(source.read_text());panel=run['registered']['protocol']['panel']
    release=Release('data/kaggle_public')
    dates=queue_phase_dates([partition_date(release,p) for p in release.files(panel,'train')],
        run['dates']['fit'][0],run['dates']['fit'][-1])
    if dates!=run['dates']:raise ValueError('Date phases differ')
    for name,digest in run['seal_sha256'].items():
        if sha256_file(args.study/(name+'_sealed.json'))!=digest:raise ValueError('Phase seal changed')
    required={'decision':'checkpoint','selection':'decision','assessment':'selection'}
    for row in run['phase_read_records']:
        if row['phase'] in required and row['prior_seal_sha256']!=run['seal_sha256'][required[row['phase']]]:
            raise ValueError('Later phase read did not follow its exact seal')
    record=next(r for r in run['cache_records'] if r['phase']=='decision')
    if sha256_file(record['path'])!=record['sha256']:raise ValueError('Decision cache changed')
    arrays=dict(np.load(record['path'],allow_pickle=False));links=arrays['links'].tolist();width=6*len(links)
    features={}
    for row in record['feature_files']:
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Feature cache changed')
        name='rich' if row['path'].endswith('_rich.parquet') else 'plain';features[name]=pd.read_parquet(row['path'])
    geometry=dict(np.load(args.study/'geometry.npz',allow_pickle=False))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']]
    networks={}
    for seed,training in run['training'].items():
        pt=args.study/('seed_'+seed+'.pt')
        if sha256_file(pt)!=training['selected_checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        model=CausalGraphQueue(hidden=32,use_graph=True).to(device)
        model.load_state_dict(torch.load(pt,map_location=device,weights_only=True));model.eval();networks[seed]=model
    trees={arm:QueueModel.load(args.study/('tree_'+arm+'.pkl')) for arm in ['plain','rich']}
    exported={}
    for role,row in run['exported_models'].items():
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Exported role changed')
        exported[role]=QueueModel.load(Path(row['path']))
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    paths={partition_date(release,p):p for p in release.files(panel,'train',False)}
    net,topology=release.network(panel),release.topology(panel);calls={role:0 for role in exported}
    for i,origin_text in enumerate(arrays['origins']):
        origin=pd.Timestamp(str(origin_text));token=origin.strftime('%Y-%m-%dT%H:%M:%SZ')
        history=pd.read_parquet(paths[origin.date().isoformat()],filters=[
            ('timestamp','>=',(origin-pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ')),('timestamp','<',token)])
        for key in ['timestamp','link_id','station_id']:history[key]=history[key].astype(str)
        history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
        condition='queue_ongoing' if arrays['ongoing'][i] else 'queue_onset'
        x=queue_history_input(history,links,net,origin,condition)
        if not np.array_equal(x,arrays['x'][i]):raise ValueError('Fresh causal graph input differs')
        times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
        target=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id']).to_frame(index=False);offset=slice(i*width,(i+1)*width)
        for arm in ['plain','rich']:
            rebuilt=queue_features(history,target,net,topology,origin,allow_origin=True,include_occupancy=arm=='rich')
            if arm=='rich':rebuilt['condition_ongoing']=float(condition=='queue_ongoing')
            expected=features[arm].iloc[offset].reset_index(drop=True)
            if list(rebuilt.columns)!=list(expected.columns) or not np.allclose(rebuilt.to_numpy(float),expected.to_numpy(float),rtol=0,atol=0,equal_nan=True):
                raise ValueError('Fresh tree inputs differ')
        p={}
        with torch.no_grad():
            for seed,model in networks.items():
                logits,_=model(torch.from_numpy(x.transpose(2,0,1)[None].copy()).to(device),*graph)
                p[seed]=weighted_bce_probability(logits.cpu().numpy()[0],4.).ravel()
        for role,model in exported.items():
            choice=run['decision_choices'][run['exported_models'][role]['candidate']]
            base=trees[choice['baseline_arm']].probability(features[choice['baseline_arm']].iloc[offset])
            probability=base if not choice['members'] else (1-choice['alpha'])*base+choice['alpha']*np.mean([p[s] for s in choice['members']],axis=0)
            score=model.forecast_probability(history,target,net,topology,origin,condition).queue_score.to_numpy()
            if not np.array_equal(score,probability):raise ValueError('Exported score differs')
            setting=choice['setting'];decision=full_queue_decisions(probability,arrays['window'][offset],arrays['condition'][offset],arrays['lead_minutes'][offset],
                setting['thresholds'],setting['require_nonempty'],setting['timing'])
            predicted=model.predict_window(history,target,net,topology,origin,condition).queue_pred.to_numpy()
            if not np.array_equal(predicted,decision):raise ValueError('Exported decision differs')
            calls[role]+=1
    result={'source_experiment_sha256':sha256_file(source),'source_snapshot':snapshot,'panel':panel,
        'n_decision_windows':len(arrays['origins']),'model_window_calls':calls,'all_fresh_legal_inputs_exact':True,
        'all_scores_exact':True,'all_full_template_decisions_exact':True,'all_phase_reads_after_exact_prior_seal':True,
        'all_phase_dates_disjoint_and_complete':True,'seconds':time.perf_counter()-start,'source':evidence(gpu_used=device.type=='cuda'),
        'formal_Queue':None,'formal_Total':None,'scope':'Independent decision-phase deployment and mechanism/date/seal replay. Public observation proxy,not formal gain evidence.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('exact decision role/window calls',sum(calls.values()),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
