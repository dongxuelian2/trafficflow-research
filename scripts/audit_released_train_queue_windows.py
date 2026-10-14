"""Compare released TRAIN windows with observation-derived proxy safeguards."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release, csv, eligible
from trafficflow.experiments import partition_date
from trafficflow.metrics import queue_iou
from trafficflow.runtime import sha256_file, capture_source_snapshot, evidence
from trafficflow.research_eval import register_protocol


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists(): raise FileExistsError('Never overwrite window audits')
    registered = register_protocol(args.output.with_suffix('.registered.json'), {
        'id': 'Released_TRAIN_Queue_window_observation_audit',
        'scope': 'All eight scored TRAIN panels only. No validation/private labels.',
        'labels': 'Threshold eligible released speed by public v_cut; observation proxy, not queue_true',
        'diagnostics': ['history condition agreement','first observed queue horizon','future eligibility/near-cutoff shares',
                        'noisy persistence guard','observed-positive selection versus full-domain metadata'],
        'no_new_model_selection': True, 'formal_Queue': None, 'formal_Total': None})
    folder.mkdir(parents=True); snapshot = capture_source_snapshot(folder/'source_snapshot.zip')
    start = time.perf_counter(); release = Release('data/kaggle_public')
    contract = json.loads((release.root/'config/competition_contract.json').read_text())
    rows = []; records = {}
    for panel in contract['tasks']['queue']['scored_panels']:
        net = release.network(panel); task = release.root/'task2'/panel/'train'
        windows = csv(task/'window_index.csv'); history = pd.read_parquet(task/'window_history.parquet')
        template = release.template(2, panel, 'train')
        raw_paths = {partition_date(release, p): p for p in release.files(panel, 'train', False)}
        for path in [task/'window_index.csv', task/'window_history.parquet', task/'sample_submission_queue.csv']:
            records[str(path)] = sha256_file(path)
        for w in windows.itertuples(index=False):
            origin = pd.Timestamp(w.forecast_origin)
            h = history[history.window_id.astype(str) == str(w.window_id)].copy()
            t = template[template.window_id == str(w.window_id)]
            links = t.link_id.drop_duplicates().tolist(); cutoff = net.v_cut.reindex(links).to_numpy(float)
            ts = pd.to_datetime(h.timestamp, utc=True)
            if (ts >= origin).any() or (ts < origin-pd.Timedelta(minutes=60)).any():
                raise ValueError('Released short history outside legal interval')
            for key in ['timestamp', 'link_id']: h[key] = h[key].astype(str)
            h.loc[~eligible(h), 'speed_kmh'] = np.nan
            times = [(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
            raw_path = raw_paths[origin.date().isoformat()]
            records[str(raw_path)] = sha256_file(raw_path)
            # Explicit TRAIN evaluator read; never a forecasting input.
            future = pd.read_parquet(raw_path, filters=[('timestamp','in',times)])
            for key in ['timestamp', 'link_id']: future[key] = future[key].astype(str)
            future.loc[~eligible(future), 'speed_kmh'] = np.nan
            v = future.pivot_table(index='timestamp', columns='link_id', values='speed_kmh').reindex(index=times,columns=links).to_numpy(float)
            hist_times = [(origin-pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in range(60,0,-5)]
            hv = h.pivot_table(index='timestamp',columns='link_id',values='speed_kmh').reindex(index=hist_times,columns=links).to_numpy(float)
            counts = (hv <= cutoff).sum(axis=0)
            inferred = 'queue_ongoing' if (counts >= 2).any() else 'queue_onset'
            last = np.full(len(links),np.nan)
            for values in hv[::-1]:
                take = ~np.isfinite(last)&np.isfinite(values); last[take] = values[take]
            valid = np.isfinite(v); y = v <= cutoff
            persistence = np.broadcast_to(last <= cutoff,v.shape)
            steps = np.flatnonzero(y.any(axis=1))
            rows.append({'window_id':w.window_id,'panel':panel,'condition':w.condition,'origin':origin.isoformat(),
                'history_observed_coverage':float(np.isfinite(hv).mean()),
                'index_history_coverage':float(w.history_coverage), 'inferred_history_condition':inferred,
                'history_condition_agrees':inferred == w.condition,
                'future_observed_eligible_coverage':float(valid.mean()),'index_future_coverage':float(w.future_coverage),
                'n_future_observation_positives':int(y.sum()),'future_observation_positive':bool(y.any()),
                'first_observed_queue_minutes':int((steps[0]+1)*5) if len(steps) else None,
                'onset_proxy_last_step_guard_passes':bool(not y[:5].any() and y[5].any()) if w.condition == 'queue_onset' else None,
                'observation_persistence_IoU':queue_iou(y[valid],persistence[valid]) if valid.any() else None,
                'future_near_cutoff_2kmh_share':float((np.abs(v-cutoff)[valid]<=2).mean()) if valid.any() else None})
    summaries = []
    for (panel, condition), group in pd.DataFrame(rows).groupby(['panel','condition']):
        summaries.append({'panel':panel,'condition':condition,'n_windows':len(group),
            'history_condition_disagreements':int((~group.history_condition_agrees).sum()),
            'zero_observed_positive_horizons':int((~group.future_observation_positive).sum()),
            'mean_future_eligible_coverage':float(group.future_observed_eligible_coverage.mean()),
            'minimum_future_eligible_coverage':float(group.future_observed_eligible_coverage.min()),
            'first_observed_queue_minutes_counts':group.first_observed_queue_minutes.fillna(-1).astype(int).value_counts().to_dict(),
            'onset_last_step_guard_failures':int((group.onset_proxy_last_step_guard_passes == False).sum()) if condition == 'queue_onset' else None,
            'persistence_guard_failures_gt_point9':int((group.observation_persistence_IoU>.9).sum())})
    result = {'registered':registered,'source_snapshot':snapshot,'source_files_sha256':records,
        'contract_sha256':sha256_file(release.root/'config/competition_contract.json'),
        'rows':rows,'panel_condition_summaries':summaries,'seconds':time.perf_counter()-start,
        'source':evidence(),'formal_Queue':None,'formal_Total':None,
        'limitations':'Only 80 early-June released TRAIN windows. Public thresholded observations are not the unreleased latent queue labels. Future coverage metadata is audited, never used as a forecast feature. Differences cannot identify hidden selector implementation or hidden labels.'}
    text = json.dumps(result,indent=2,allow_nan=False)
    args.output.write_text(text,encoding='utf-8'); (folder/'experiment.json').write_text(text,encoding='utf-8')
    print('windows',len(rows),'onset_guard_failures',sum(r['onset_last_step_guard_failures'] or 0 for r in summaries),
        'zero_positive',sum(r['zero_observed_positive_horizons'] for r in summaries),
        'condition_disagreements',sum(r['history_condition_disagreements'] for r in summaries),
        'seconds',result['seconds'],flush=True)


if __name__ == '__main__': main()
