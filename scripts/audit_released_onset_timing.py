"""Check public train observations and own forecast alignment with the announcement."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from trafficflow.data import Release,csv,eligible
from trafficflow.experiments import partition_date
from trafficflow.research_eval import register_protocol


if __name__=='__main__':
    output=Path('reports/research/C23_released_onset_timing_audit.json')
    if output.exists():raise FileExistsError('Never overwrite a sampling audit')
    registered=register_protocol(output.with_suffix('.registered.json'),{
        'id':'C23_public_onset_timing','source':'Official public discussion 742350, 2026-09-21, Traffice_Flow_Bench_Organizer',
        'url':'https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350',
        'announcement':'Published queue_onset windows have no queued cells at T+5 through T+25; only T+30 can be queued.',
        'audit':'Independent public-train noisy observation check; own val/private prediction counts use metadata only',
        'validation_private_future_measurements_read':False,'formal_Queue':None})
    release=Release('data/kaggle_public')
    own=pd.read_csv('artifacts/submission_best/queue.csv',dtype=str,keep_default_na=False)
    train=[];forecasts=[]
    for panel in release.panels():
        if panel.startswith('D12_I405'):continue
        net=release.network(panel)
        raw_paths={partition_date(release,p):p for p in release.files(panel,'train',False)}
        cache={}
        index=csv(release.root/'task2'/panel/'train'/'window_index.csv')
        target=release.template(2,panel,'train')
        for window in index.itertuples(index=False):
            if window.condition!='queue_onset':continue
            origin=pd.Timestamp(window.forecast_origin)
            wanted=target[target.window_id==window.window_id][['timestamp','link_id']].copy()
            pieces=[]
            for date in wanted.timestamp.str[:10].unique():
                if date not in cache:
                    raw=release.read(raw_paths[date])
                    cache[date]=raw.loc[eligible(raw),['timestamp','link_id','speed_kmh']]
                pieces.append(cache[date][cache[date].timestamp.isin(wanted.timestamp.unique())])
            values=pd.concat(pieces).groupby(['timestamp','link_id'],as_index=False).speed_kmh.mean()
            joined=wanted.merge(values,on=['timestamp','link_id'],how='left',validate='one_to_one')
            lead=(pd.to_datetime(joined.timestamp,utc=True)-origin).dt.total_seconds()/60
            available=np.isfinite(joined.speed_kmh)
            queued=available&(joined.speed_kmh<=joined.link_id.map(net.v_cut))
            first=float(lead[queued].min()) if queued.any() else None
            train.append({'panel':panel,'window_id':window.window_id,'origin':window.forecast_origin,
                'first_queued_observation_lead':first,'early_observed_queue_cells':int((queued&(lead<30)).sum()),
                'final_observed_queue_cells':int((queued&(lead==30)).sum()),
                'eligible_future_observation_coverage':float(available.mean()),
                'latent_queue_truth':'unavailable; announcement timing is public'})
        for split in ('validation','private'):
            windows=csv(release.root/'task2'/panel/split/'window_index.csv')
            for window in windows.itertuples(index=False):
                if window.condition!='queue_onset':continue
                sub=own[own.window_id==window.window_id]
                if sub.empty:raise ValueError('Own frozen Queue forecast missing')
                lead=(pd.to_datetime(sub.timestamp,utc=True)-pd.Timestamp(window.forecast_origin)).dt.total_seconds()/60
                p=pd.to_numeric(sub.queue_pred).to_numpy()
                forecasts.append({'panel':panel,'split':split,'window_id':window.window_id,
                    'early_positive_predictions':int(((lead<30)&(p==1)).sum()),
                    'final_positive_predictions':int(((lead==30)&(p==1)).sum()),
                    'future_measurements_read':False})
    result={'registered':registered,'train_windows':train,'own_forecasts':forecasts,
        'n_public_train_onset_windows':len(train),
        'train_windows_with_first_observed_queue_at_30':sum(r['first_queued_observation_lead']==30 for r in train),
        'train_windows_without_observed_queue':sum(r['first_queued_observation_lead'] is None for r in train),
        'total_early_positive_predictions_in_own_submission':sum(r['early_positive_predictions'] for r in forecasts),
        'formal_Queue':None,'limitation':'Released train observations are noisy and partly missing. This is a sampling audit, not a queue score or hidden-label check.'}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:result[k] for k in ('n_public_train_onset_windows','train_windows_with_first_observed_queue_at_30','train_windows_without_observed_queue','total_early_positive_predictions_in_own_submission')},flush=True)
