"""Fixed-model paired replay of full vs future-eligible Queue decision domains.

No refitting or recalibration. Only public TRAIN labels score these replays.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release,eligible
from trafficflow.queue import QueueModel,queue_features
from trafficflow.queue_domain import predict_full_forecast,score_full_forecast
from trafficflow.causal_queue import released_origin_rows
from trafficflow.experiments import partition_date
from trafficflow.metrics import grouped_iou
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol


STUDIES=['C27_D12_I5_N_onset_specialist','C28_D7_I10_W_onset_specialist','C29_D12_I5_N_winter_onset_specialist']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite Queue domain audit')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Queue_fixed_model_full_forecast_domain_audit','studies':STUDIES,
        'scope':'Old selected origins, frozen probability/threshold models; no fitting or model reselection',
        'prediction':'Complete six-step public link domain, inputs timestamp<=T only',
        'scoring':'Read eligible public TRAIN future observations only after full prediction; apply scoring mask after set decision',
        'legacy_comparison':'Deliberately reproduce old availability-filtered set decisions to quantify bias',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');rows=[];source_records=[]
    for study in STUDIES:
        source=Path('reports/research')/(study+'.json');run=json.loads(source.read_text())
        panel=run['registered']['protocol']['panel'];net,topology=release.network(panel),release.topology(panel)
        paths={partition_date(release,p):p for p in release.files(panel,'train',False)}
        models={p.stem:QueueModel.load(p) for p in (Path('artifacts/research')/study).glob('*.pkl')}
        source_records.append({'report':source.as_posix(),'sha256':sha256_file(source),
            'models_sha256':{name:sha256_file(Path('artifacts/research')/study/(name+'.pkl')) for name in models}})
        phases=[('selection',run['samplers']['selection'],run['selection'])]+[
            ('assessment_'+str(r['seed']),r['sampler'],r['models']) for r in run['assessment']]
        for phase,metadata,legacy_scores in phases:
            old_labels=[];old_windows=[];old_conditions=[];old_predictions={n:[] for n in models}
            full_labels=[];full_windows=[];full_conditions=[];full_masks=[];full_predictions={n:[] for n in models}
            changed={n:0 for n in models}
            for chosen in sorted(metadata['windows'],key=lambda r:r['origin']):
                origin=pd.Timestamp(chosen['origin']);date=origin.date().isoformat();path=paths[date]
                begin=(origin-pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ')
                token=origin.strftime('%Y-%m-%dT%H:%M:%SZ')
                history=pd.read_parquet(path,filters=[('timestamp','>=',begin),('timestamp','<',token)])
                for key in ['timestamp','station_id','link_id']:history[key]=history[key].astype(str)
                history=pd.concat([history,released_origin_rows(release,panel,'train',origin)],ignore_index=True)
                links=history.link_id.drop_duplicates().tolist()
                times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
                target=pd.MultiIndex.from_product([times,links],names=['timestamp','link_id']).to_frame(index=False)
                x=queue_features(history,target,net,topology,origin,allow_origin=True,include_occupancy=False)
                wid=f"{panel}:{token}:{metadata['seed']}";window=np.full(len(x),wid);condition=np.full(len(x),chosen['condition'])
                predictions={name:predict_full_forecast(model,x,window,condition) for name,model in models.items()}
                # All full-domain decisions are sealed before future labels or
                # availability are materialized. No future mask is a feature.
                future=release.read(path);future=future[future.timestamp.isin(times)].copy()
                future.loc[~eligible(future),'speed_kmh']=np.nan
                speed=future.pivot_table(index='timestamp',columns='link_id',values='speed_kmh',aggfunc='mean').reindex(index=times,columns=links).to_numpy(float)
                available=np.isfinite(speed).ravel();labels=(speed<=net.v_cut.reindex(links).to_numpy(float)).ravel().astype(int)
                full_labels.extend(labels);full_masks.extend(available);full_windows.extend(window);full_conditions.extend(condition)
                old_labels.extend(labels[available]);old_windows.extend(window[available]);old_conditions.extend(condition[available])
                for name,model in models.items():
                    legacy=model.classify(x.loc[available],window[available],condition[available])
                    changed[name]+=int((predictions[name][available]!=legacy).sum())
                    old_predictions[name].extend(legacy);full_predictions[name].extend(predictions[name])
            for name,model in models.items():
                old=grouped_iou(np.array(old_labels),np.array(old_predictions[name]),np.array(old_windows),np.array(old_conditions))
                expected=legacy_scores[name]['metrics'] if phase=='selection' else legacy_scores[name]
                if abs(old['proxy_IoU']-expected['proxy_IoU'])>1e-12:
                    raise ValueError('Legacy exact window replay failed: '+study+'/'+phase+'/'+name)
                corrected=score_full_forecast(np.array(full_labels),np.array(full_predictions[name]),np.array(full_windows),np.array(full_conditions),np.array(full_masks))
                rows.append({'study':study,'phase':phase,'model':name,'was_original_cal_selected':name==run['selected'],
                    'class':type(model).__name__,'require_nonempty':getattr(model,'require_nonempty',None),
                    'full_forecast_cells':len(full_labels),'scored_cells':int(np.sum(full_masks)),
                    'changed_scored_decisions':changed[name],'legacy_filtered_domain':old,
                    'full_domain_then_score':corrected,'proxy_IoU_change':corrected['proxy_IoU']-old['proxy_IoU'],
                    'legacy_proxy_replay_exact':True,'formal_Queue':None})
            print(study,phase,'replayed',flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'source_records':source_records,'rows':rows,
        'seconds':time.perf_counter()-start,'source':evidence(),'formal_Queue':None,'formal_Total':None,
        'limitations':'Observed TRAIN labels, not latent truth. Old nonempty/expected-set arms are domain-sensitive; pointwise threshold arms can be unchanged. Original choices retained for audit, no new calibrated model or formal submission. Full-domain inputs still use the approximate public-train event sampler.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('changed_rows',sum(r['changed_scored_decisions'] for r in rows),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
