"""Fit-only,measurement-only outage views;no raw teachers are opened here."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.data import Release
from trafficflow.state import StateModel,StateFeatures
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.state_augmentation import rebuild_after_outage
from trafficflow.outage_stress import outage_geometry
from trafficflow.experiments import partition_date
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot,sha256_file,evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--baseline-file',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new augmentation cache')
    prepared=json.loads((args.cache/'prepared.json').read_text());panel=prepared['panel']
    report=Path('reports/research')/(args.output.name+'.json');cases=[(d,f) for d in [36,72] for f in [.3,.7]]
    protocol={'id':'Fit_only_full_feature_rebuilt_public_outage_views','source_cache':args.cache.as_posix(),
        'prepared_sha256':sha256_file(args.cache/'prepared.json'),'baseline_file':args.baseline_file.as_posix(),
        'baseline_sha256':sha256_file(args.baseline_file),'fit_dates':prepared['dates']['fit'],
        'mask_seed':20261109,'cases':[list(c) for c in cases],
        'geometry':'Same metadata-only daytime centers as registered public stress;four nested sizes per day',
        'labels':'No raw teacher files opened;original cached labels/masks are not changed or added to input',
        'features':'Remove all measurements/flags then rebuild StateFeatures and strict12,including base/interpolation',
        'scope':'Only the original fit dates;not calibration/assessment/hidden splits'}
    registered=register_protocol(report.with_suffix('.registered.json'),protocol)
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');net,topology=release.network(panel),release.topology(panel)
    baseline=StateModel.load(args.baseline_file);files={partition_date(release,p):p for p in release.files(panel,'train')}
    rows=[]
    for date in protocol['fit_dates']:
        original_path=args.cache/'fit'/(date+'.npz');cached=dict(np.load(original_path,allow_pickle=False))
        frame=release.read(files[date]);original=StateFeatures(frame,net,topology,baseline.profile,version=baseline.feature_version)
        if not np.array_equal(reconstruction_input(original),cached['x']):raise ValueError('Original cache inputs/profile differ')
        views=[];geometry=[];regions=[]
        for duration,fraction in cases:
            selected,g=outage_geometry(frame,panel,date,protocol['mask_seed'],duration,fraction)
            features=rebuild_after_outage(frame,selected,net,topology,baseline.profile,baseline.feature_version)
            if features.links!=original.links or features.times!=original.times:raise ValueError('Augmentation grid changed')
            x=reconstruction_input(features);region=np.zeros(x.shape[:2],bool)
            region.ravel()[features.lookup(frame.loc[selected])]=True
            if x[...,2:4][region].any() or x[...,9][region].any():raise ValueError('Removed measurement visible')
            views.append(x);regions.append(region);geometry.append(g)
        path=args.output/(date+'.npz');np.savez_compressed(path,x=np.stack(views),regions=np.stack(regions))
        rows.append({'date':date,'path':path.as_posix(),'sha256':sha256_file(path),
            'source_cache_sha256':sha256_file(original_path),'masked_file':files[date].as_posix(),
            'masked_file_sha256':sha256_file(files[date]),'geometries':geometry,
            'original_target_cells':int(cached['mask'].sum()),
            'target_cells_in_outages':[int((r&cached['mask']).sum()) for r in regions]})
        if len(rows)%10==0:print('augmentation_cache',panel,len(rows),'/',len(protocol['fit_dates']),flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'rows':rows,'seconds':time.perf_counter()-start,
        'n_dates':len(rows),'n_views':len(rows)*4,'raw_teachers_opened':False,'source':evidence(gpu_used=False),
        'formal_State':None,'formal_Physics':None,'formal_Total':None}
    text=json.dumps(result,indent=2,allow_nan=False);report.write_text(text,encoding='utf-8');(args.output/'cache.json').write_text(text,encoding='utf-8')
    print('augmentation_cache_complete',result['n_views'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
