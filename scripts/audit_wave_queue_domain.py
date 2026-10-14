"""Full-domain replay of frozen wave Queue models; no refit or cal reselection."""
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
from trafficflow.queue import QueueModel,queue_features
from trafficflow.graph_queue import CausalGraphQueue,queue_history_input
from trafficflow.causal_queue import released_origin_rows
from trafficflow.queue_calibration import decisions,public_onset_timing
from trafficflow.queue_domain import predict_full_forecast,score_full_forecast
from trafficflow.probability_calibration import weighted_bce_probability
from trafficflow.experiments import partition_date
from trafficflow.metrics import grouped_iou
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from compare_queue_events import comparison


class FixedDecision:
    def __init__(self,p,setting):self.p,self.setting=p,setting
    def classify(self,x,window,condition):
        if len(x)!=len(self.p):raise ValueError('Fixed probability domain differs')
        pred=decisions(self.p,window,condition,self.setting['thresholds'],self.setting['require_nonempty'])
        return public_onset_timing(pred,x.lead_minutes.to_numpy(),condition)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite a wave-domain replay')
    names=['C32_D12_I5_N_wave_graph_Queue','C34_D12_I5_N_winter_wave_Queue']
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Frozen_wave_Queue_full_forecast_domain_replay','studies':names,
        'scope':'Same weights/graphs/cal alpha/threshold/nonempty settings and event origins; no reselection',
        'causality':'Fresh history restricted to <=T; all reconstructed nine-channel inputs must equal cache',
        'decision':'Full public template before evaluator-only future availability filtering',
        'timing':'Retain original decision-then-onset-zero ordering for controlled replay; do not silently repair another rule',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');manifest=json.loads(Path('reports/research/model_artifacts_checkpoint_assimilation_v1.json').read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    rows,source_records,comparisons=[],[],[]
    for study in names:
        path=Path('reports/research')/(study+'.json');run=json.loads(path.read_text());producer=Path('artifacts/research')/study
        expected=run['registered']['protocol']['source_cache_report_sha256']
        matches=[Path(r['path']).parent for r in manifest['artifacts'] if r['path'].endswith('/experiment.json') and r['sha256']==expected]
        if len(matches)!=1:raise ValueError('Cannot identify unique frozen causal cache')
        cache=matches[0];cached=json.loads((cache/'experiment.json').read_text())
        if sha256_file(cache/'experiment.json')!=expected:raise ValueError('Causal cache report changed')
        panel=cached['registered']['protocol']['panel'];net,topology=release.network(panel),release.topology(panel)
        rawpaths={partition_date(release,p):p for p in release.files(panel,'train',False)}
        base_path=Path('artifacts/research')/Path(cached['registered']['protocol']['matched_study']).stem/'shared_threshold.pkl'
        baseline=QueueModel.load(base_path)
        seal=json.loads((producer/'selection_sealed.json').read_text());settings=seal['settings']
        neural={};graphs={}
        for name in ['direct','wave_15min','wave_30min']:
            if sha256_file(producer/(name+'.pt'))!=run['checkpoint_sha256'][name]:raise ValueError('Wave weights changed')
            model=CausalGraphQueue(hidden=32,use_graph=True).to(device)
            model.load_state_dict(torch.load(producer/(name+'.pt'),map_location=device,weights_only=True));model.eval();neural[name]=model
            g=dict(np.load(producer/(name+'_geometry.npz')));graphs[name]=[torch.tensor(g[k],device=device) for k in ['incoming','outgoing','geometry']]
        corrected={'dates':run['dates'],'assessment':[]}
        phases=[('selection',411,cached['samplers']['selection'],{k:v['metrics'] for k,v in run['selection'].items()})]+[
            ('assessment',r['seed'],r['sampler'],r['models']) for r in run['assessment']]
        for phase,seed,metadata,old_metrics in phases:
            source_npz=cache/(phase+'_'+str(seed)+'.npz')
            expected_hash=cached['cache_sha256']['selection'] if phase=='selection' else next(r['cache_sha256'] for r in cached['assessment'] if r['seed']==seed)
            if sha256_file(source_npz)!=expected_hash:raise ValueError('Sequence cache changed')
            arrays=dict(np.load(source_npz,allow_pickle=False));labels=[];masks=[];windows=[];conditions=[]
            legacy_pred={n:[] for n in settings};full_pred={n:[] for n in settings};changes={n:0 for n in settings}
            for i,event in enumerate(sorted(metadata['windows'],key=lambda r:r['origin'])):
                origin=pd.Timestamp(event['origin']);token=origin.strftime('%Y-%m-%dT%H:%M:%SZ');rawpath=rawpaths[origin.date().isoformat()]
                history=pd.read_parquet(rawpath,filters=[('timestamp','>=',(origin-pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ')),('timestamp','<',token)])
                for key in ['timestamp','station_id','link_id']:history[key]=history[key].astype(str)
                history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
                links=history.link_id.drop_duplicates().tolist()
                rebuilt=queue_history_input(history,links,net,origin,event['condition'])
                if not np.array_equal(rebuilt,arrays['x'][i]):raise ValueError('Fresh causal NN input differs')
                times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
                target=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id']).to_frame(index=False)
                x=queue_features(history,target,net,topology,origin,allow_origin=True,include_occupancy=False)
                wid=f'{panel}:{token}:{seed}';window=np.full(len(x),wid);condition=np.full(len(x),event['condition'])
                reference=baseline.probability(x);probabilities={'shared_threshold':reference}
                with torch.no_grad():
                    tensor=torch.from_numpy(rebuilt.transpose(2,0,1)[None].copy()).to(device)
                    for name,model in neural.items():
                        logits,_=model(tensor,*graphs[name]);p=weighted_bce_probability(logits.cpu().numpy()[0],4.).ravel()
                        probabilities[name]=(1-settings[name]['alpha'])*reference+settings[name]['alpha']*p
                predictions={name:predict_full_forecast(FixedDecision(p,settings[name]),x,window,condition) for name,p in probabilities.items()}
                # Future eligibility is used only here, after all full decisions.
                valid=arrays['available'][i].ravel();truth=arrays['labels'][i].ravel()
                for name,p in probabilities.items():
                    old=FixedDecision(p[valid],settings[name]).classify(x.loc[valid],window[valid],condition[valid])
                    legacy_pred[name].extend(old);full_pred[name].extend(predictions[name]);changes[name]+=int((old!=predictions[name][valid]).sum())
                labels.extend(truth);masks.extend(valid);windows.extend(window);conditions.extend(condition)
            labels,masks,windows,conditions=map(np.asarray,[labels,masks,windows,conditions]);metrics={}
            for name in settings:
                legacy=grouped_iou(labels[masks],np.array(legacy_pred[name]),windows[masks],conditions[masks])
                if abs(legacy['proxy_IoU']-old_metrics[name]['proxy_IoU'])>1e-12:raise ValueError('Legacy wave replay failed')
                metric=score_full_forecast(labels,np.array(full_pred[name]),windows,conditions,masks);metrics[name]=metric
                rows.append({'study':study,'phase':phase,'seed':seed,'model':name,'settings':settings[name],
                    'changed_scored_decisions':changes[name],'legacy_filtered_domain':legacy,'full_domain_then_score':metric,
                    'legacy_replay_exact':True,'fresh_causal_inputs_exact':True,'formal_Queue':None})
            if phase=='assessment':corrected['assessment'].append({'seed':seed,'models':metrics})
            print(study,phase,seed,'verified',flush=True)
        comparisons.append({'study':study,'old_selected':run['selected'],'cal_choice_retained':True,
            'old_comparison':run['selected_comparison'],
            'corrected_comparison':comparison(corrected,'shared_threshold',run['selected'])})
        source_records.append({'report':path.as_posix(),'sha256':sha256_file(path),'causal_cache':cache.as_posix(),
            'causal_cache_report_sha256':expected,'baseline_sha256':sha256_file(base_path)})
    result={'registered':registered,'source_snapshot':snapshot,'source_records':source_records,'rows':rows,'comparisons':comparisons,
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=device.type=='cuda'),'formal_Queue':None,'formal_Total':None,
        'limitations':'Frozen checkpoint/alpha/decision settings; no corrected cal reselection. Approximate noisy observation events, not latent labels or final tests. Future timing-before-force and full-domain calibration still require their own comparison.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(comparisons,flush=True)


if __name__=='__main__':main()
