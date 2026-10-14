"""Quantify loss denominator scale on the exact planned fit patch sequence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from trafficflow.data import Release
from trafficflow.runtime import sha256_file, capture_source_snapshot
from trafficflow.research_eval import register_protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=20261013)
    args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite normalization evidence')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Full_density_denominator_fit_patch_audit','seed':args.seed,
        'cache_prepared_sha256':{p.as_posix():sha256_file(p/'prepared.json') for p in args.cache},
        'sampling':'Same 24*120 batches of4, uniform regime/day then64-slot patch as full_density_experiment',
        'scope':'Fit labels only, no cal/assessment inspection or model selection',
        'identity':'Full loss equals public-target-pair loss times clamped target/all observed-increment denominator ratio; non-target numerator zero',
        'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip')
    release=Release('data/kaggle_public');results=[]
    for cache in args.cache:
        info=json.loads((cache/'prepared.json').read_text());net=release.network(info['panel'])
        days=[]
        for path in sorted((cache/'fit').glob('*.npz')):
            row=next(r for r in info['records'] if r['part']=='fit' and r['date']==path.stem)
            if sha256_file(path)!=row['full_label_cache_sha256']:raise ValueError('Fit cache changed')
            d=dict(np.load(path,allow_pickle=False));x=d.get('x_control',d['x'])
            factor=(net.lanes*net.length_km).reindex(d['links']).to_numpy(np.float32)
            old=np.where(d['mask'][...,None],d['y'],x[...,:2]);full=d['y_full']
            visible=np.all(x[...,2:4]>.5,axis=-1);old_valid=visible|d['mask']
            touching=d['mask'][:-1]|d['mask'][1:]
            old_pairs=old_valid[:-1]&old_valid[1:]&touching
            public=d['lwr_pair_valid']
            # Use float32 label/factor arithmetic like training. Prefix sums
            # have float64 accumulation; ratios are numeric audits, not scores.
            old_delta=np.abs(np.diff(np.maximum(old[...,1]*600,0)/np.maximum(old[...,0]*25,1)*factor,axis=0))
            full_delta=np.abs(np.diff(np.maximum(full[...,1]*600,0)/np.maximum(full[...,0]*25,1)*factor,axis=0))
            def prefix(delta,mask):return np.r_[0.,np.cumsum(np.sum(delta*mask,axis=1,dtype=np.float64))]
            days.append({'date':str(d['times'][0]),'regime':str(d['regime']),'steps':len(d['times']),
                'denominators':[prefix(old_delta,old_pairs),prefix(full_delta,public&touching),prefix(full_delta,public)]})
        regimes={r:[d for d in days if d['regime']==r] for r in ['R1','R2','R3']}
        rng=np.random.default_rng(args.seed);sampling=hashlib.sha256();denominators=[]
        for _ in range(24*120):
            totals=np.zeros(3)
            for _ in range(4):
                regime=['R1','R2','R3'][rng.integers(3)];group=regimes[regime];day=group[rng.integers(len(group))]
                at=int(rng.integers(0,day['steps']-64+1));sampling.update(f"{regime}|{day['date']}|{at};".encode())
                totals+=[p[at+63]-p[at] for p in day['denominators']]
            denominators.append(totals)
        den=np.maximum(np.array(denominators),1.)
        ratios=den[:,1]/den[:,2]
        if ratios.max()>1+1e-9:raise ValueError('Target denominator exceeds full denominator')
        results.append({'cache':cache.as_posix(),'panel':info['panel'],'sample_sequence_sha256':sampling.hexdigest(),
            'n_batches':len(ratios),'full_loss_equivalent_target_loss_multiplier':{
                'mean':float(ratios.mean()),'min':float(ratios.min()),'max':float(ratios.max()),
                'p05_median_p95':np.quantile(ratios,[.05,.5,.95]).tolist()},
            'mean_effective_lambda_from_base_0_01':float(.01*ratios.mean()),
            'public_target_to_legacy_denominator_mean':float(np.mean(den[:,1]/den[:,0])),
            'interpretation':'Different normalization changes batch-specific strength, not new information. Coverage filtering also changes which errors enter numerator; that is not a scalar-strength equivalence.'})
    result={'registered':registered,'source_snapshot':snapshot,'rows':results,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limitations':'Planned fit patch ratios, with float64 prefix sums. Quantiles are descriptive, not confidence intervals. No checkpoint/alpha changed.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(results,flush=True)


if __name__=='__main__':main()
