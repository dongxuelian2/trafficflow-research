"""Public fit truth->new point masks;full feature rebuild and source audit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release,eligible
from trafficflow.state import StateModel,StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.gap_gated_state import missing_run_steps
from trafficflow.native_remask import native_remask,RATES
from trafficflow.experiments import partition_date
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Never overwrite remask cache')
    source=json.loads(args.source_study.read_text());config=source['registered']['protocol'];cache=Path(config['cache']);prepared=source['prepared'];panel=config['panel']
    index_path=Path('data/kaggle_public/task2')/panel/'train/window_index.csv';index=pd.read_csv(index_path,dtype=str)
    seeds=[20261203,20261207,20261211,20261213]
    registered=register_protocol(report.with_suffix('.registered.json'),{
        'id':'Fit_public_truth_native_rate_remask_point_vs_mock_queue','source_study':args.source_study.as_posix(),
        'source_study_sha256':sha256_file(args.source_study),'cache':cache.as_posix(),'prepared_sha256':sha256_file(cache/'prepared.json'),
        'baseline_file':config['baseline_file'],'baseline_sha256':config['baseline_sha256'],'fit_dates':prepared['dates']['fit'],
        'rates':RATES,'view_seeds':seeds,'variants':['point','queue'],'public_queue_index_sha256':sha256_file(index_path),
        'point_mask':'Own seeded IID link/time masks at original day regime rate;not official stable hash;all stations on a cell share selection',
        'queue_variant':'Same point coordinates plus one metadata-random origin/day/view:history T-60..T visible and future T+5..T+90 removed;neither supervised',
        'actual_public_windows':'Respect actual published train Queue history/future buffers in both variants;never use condition/future coverage/labels',
        'labels':'Raw public fit eligible link/time means;new target positions,so label population differs from original frozen control',
        'features':'All measured channels blanked together;rebuild profile/base/interpolation/availability from modified frame',
        'profile':'Original first45day frozen profile;never fitted on new fit/cal/assessment labels',
        'scope':'Only fit dates opened;raw truths are public train,not hidden/reference;original/augmented inputs and loss-mask exposure reported separately'})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');net,topology=release.network(panel),release.topology(panel)
    if sha256_file(config['baseline_file'])!=config['baseline_sha256']:raise ValueError('Baseline changed')
    baseline=StateModel.load(Path(config['baseline_file']));files={partition_date(release,p):p for p in release.files(panel,'train')}
    raw_files={p.name:p for p in release.files(panel,'train',False)};rows=[];audits=[]
    for date in prepared['dates']['fit']:
        path=files[date];raw_path=raw_files[path.name];frame=release.read(path);raw=release.read(raw_path)
        if 'mask_regime' not in raw:raw['mask_regime']=frame.mask_regime.iloc[0]
        regime=frame.mask_regime.iloc[0];cached_path=cache/'fit'/(date+'.npz');cached=dict(np.load(cached_path))
        if sha256_file(cached_path)!=source['cache_sha256'][cached_path.as_posix()]:raise ValueError('Original fit cache changed')
        original=StateFeatures(frame,net,topology,baseline.profile,baseline.feature_version)
        if not np.array_equal(reconstruction_input(original),cached['x']):raise ValueError('Original inputs differ')
        shape=cached['mask'].shape;teacher=np.zeros((*shape,2),np.float32);good=raw.loc[eligible(raw)].copy()
        indices=original.lookup(good);values=pd.DataFrame({'i':indices,'v':good.speed_kmh.to_numpy(float),
            'q':good.flow_vph.to_numpy(float)/good.link_id.map(net.lanes).to_numpy(float)}).groupby('i')[['v','q']].mean()
        teacher.reshape(-1,2)[values.index]=values.to_numpy()/np.array([25.,600.])
        if not np.array_equal(teacher[cached['mask']],cached['y'][cached['mask']]):raise ValueError('Raw teacher differs from original targets')
        origins_utc=pd.to_datetime(index.forecast_origin,utc=True);day_start=pd.Timestamp(date,tz='UTC')
        # Include neighboring-day windows whose history or90min guard intersects this day.
        public_origins=index.loc[(origins_utc>=day_start-pd.Timedelta(minutes=90))&
            (origins_utc<day_start+pd.Timedelta(days=1,minutes=60)),'forecast_origin'].tolist()
        run=missing_run_steps(original.raw,original.times).ravel()[cached['target_indices']]
        bins={'5min':int((run==1).sum()),'10min':int((run==2).sum()),'15_25min':int(((run>=3)&(run<=5)).sum()),
            '30_55min':int(((run>=6)&(run<=11)).sum()),'60_85min':int(((run>=12)&(run<=17)).sum()),
            '90_175min':int(((run>=18)&(run<=35)).sum()),'180min_plus':int((run>=36).sum())}
        if sum(bins.values())!=len(cached['target_indices']):raise ValueError('Visible or unclassified original target')
        audits.append({'date':date,'regime':regime,'station_targets':len(run),'target_gap_bins':bins,
            'public_queue_origins':public_origins,'observed_input_grid_fraction':float(np.isfinite(original.raw).all(axis=-1).mean())})
        arrays={'y':teacher};metadata=[]
        for variant in ['point','queue']:
            xs=[];masks=[];hidden=[]
            for seed in seeds:
                view,targets,details=native_remask(raw,panel,date,seed,regime,public_origins,variant=='queue')
                features=StateFeatures(view,net,topology,baseline.profile,baseline.feature_version)
                if features.links!=original.links or features.times!=original.times:raise ValueError('Grid changed')
                x=reconstruction_input(features);mask=np.zeros(shape,bool);mask.ravel()[features.lookup(raw.loc[targets])]=True
                if x[...,2:4][mask].any() or x[...,9][mask].any():raise ValueError('Supervised new target visible')
                region=np.zeros(shape,bool);region.ravel()[features.lookup(raw.loc[details['hidden_rows']])]=True
                xs.append(x);masks.append(mask);hidden.append(region)
                metadata.append({'variant':variant,'seed':seed,'mock_origin':details['mock_origin'],'n_targets':int(mask.sum()),
                    'n_target_cells_outside_original_mask':int((mask&~cached['mask']).sum()),
                    'point_rows_sha256':hashlib.sha256(details['point_rows'].tobytes()).hexdigest(),
                    'history_rows_sha256':hashlib.sha256(details['history_rows'].tobytes()).hexdigest(),
                    'guard_rows_sha256':hashlib.sha256(details['guard_rows'].tobytes()).hexdigest(),
                    'input_visible_fraction':float(np.isfinite(features.raw).all(axis=-1).mean())})
            arrays['x_'+variant]=np.stack(xs);arrays['mask_'+variant]=np.stack(masks);arrays['hidden_'+variant]=np.stack(hidden)
        output=args.output/(date+'.npz');np.savez_compressed(output,**arrays)
        rows.append({'date':date,'regime':regime,'path':output.as_posix(),'sha256':sha256_file(output),
            'original_cache_path':cached_path.as_posix(),'original_cache_sha256':sha256_file(cached_path),
            'masked_file':path.as_posix(),'masked_file_sha256':sha256_file(path),'raw_file':raw_path.as_posix(),'raw_file_sha256':sha256_file(raw_path),
            'public_origins':public_origins,'original_target_grid_cells':int(cached['mask'].sum()),'raw_eligible_grid_cells':len(values),'views':metadata})
        if len(rows)%10==0:print('native_remask_cache',panel,len(rows),'/',len(prepared['dates']['fit']),flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'rows':rows,'original_input_audit':audits,
        'original_target_gap_totals':{k:sum(a['target_gap_bins'][k] for a in audits) for k in audits[0]['target_gap_bins']},
        'n_fit_dates':len(rows),'n_views':len(rows)*8,'seconds':time.perf_counter()-start,'source':evidence(gpu_used=False),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Own fit remasks,not official selector reconstruction. Mock Queue origins uniform metadata draws,not onset/ongoing traffic-conditioned events. '
        'Additional distinct label locations are intentional;not an equal-label-population architecture comparison.'}
    text=json.dumps(result,indent=2,allow_nan=False);report.write_text(text,encoding='utf-8');(args.output/'cache.json').write_text(text,encoding='utf-8')
    print('native_remask_cache_complete',result['n_views'],'gap_totals',result['original_target_gap_totals'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
