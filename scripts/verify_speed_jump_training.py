"""Reconstruct both supervision counters and replay every jump-study export."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import VALUES
from trafficflow.recurrent_state import make_network
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from flow_transfer_common import load_exports,PublicDays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite jump verification')
    study=json.loads(args.study.read_text());cfg=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem
    control=json.loads(Path(cfg['control_study']).read_text());control_folder=Path('artifacts/research')/Path(cfg['control_study']).stem
    views=json.loads((Path(cfg['views'])/'cache.json').read_text());original=json.loads(Path(cfg['source_study']).read_text());old=original['registered']['protocol'];cache=Path(old['cache'])
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256'] or json.loads((source/'selection_sealed.json').read_text())!=study['selection']:raise ValueError('Seal changed')
    if study['phase_read_order']!=[{'part':'selection','prior_selection_seal_sha256':None},{'part':'assessment','prior_selection_seal_sha256':study['selection_seal_sha256']}]:raise ValueError('Assessment order changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Independent_speed_jump_exposure_initial_and_full_deployment',
        'source_study_sha256':sha256_file(args.study),'scope':'All original cell and new pair counters,initial weights,selected saved models,all28dates x7exports and saved scores. Not control refitting.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    fits=[];view_rows={r['date']:r for r in views['rows']}
    for path in sorted((cache/'fit').glob('*.npz')):
        if sha256_file(path)!=study['cache_sha256'][path.as_posix()]:raise ValueError('Original fit changed')
        d=dict(np.load(path));d['date']=path.stem;row=view_rows[path.stem]
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('New masks/labels changed')
        with np.load(row['path']) as a:d.update(raw_y=a['y'],mask_queue=a['mask_queue'])
        fits.append(d)
    groups=[[d for d in fits if str(d['regime'])==r] for r in ['R1','R2','R3']];checks=[]
    for seed in cfg['seeds']:
        role=f'jump_seed{seed}';record=study['training'][role];paired=f'seed{seed}_queue';torch.manual_seed(seed);model=make_network('tcn_gru',48)
        initial=hashlib.sha256(b''.join(k.encode()+v.detach().numpy().tobytes() for k,v in model.state_dict().items())).hexdigest()
        if initial!=record['initial_sha256'] or initial!=control['training'][paired]['initial_sha256']:raise ValueError('Shared initialization differs')
        path=Path(study['plans'][str(seed)]['path'])
        if sha256_file(path)!=study['plans'][str(seed)]['sha256'] or study['plans'][str(seed)]!=control['plans'][str(seed)]:raise ValueError('Shared plan changed')
        with np.load(path) as a:plan=a['plan']
        cells={d['date']:np.zeros(d['mask'].shape,np.int32) for d in fits};pairs={k:np.zeros_like(v) for k,v in cells.items()};patch=hashlib.sha256()
        for batch in plan:
            for group,index,center,view in batch:
                d=groups[group][index];first=min(max(int(center)-32,0),len(d['mask'])-64);last=first+64;left=min(max(first-32,0),len(d['mask'])-128)
                mask=d['mask_queue'][view] if view>=0 else d['mask'];teacher=d['raw_y'] if view>=0 else d['y'];cells[d['date']][first:last]+=mask[first:last]
                for t in range(first+1,last):pairs[d['date']][t]+=mask[t]&mask[t-1]&(np.abs(teacher[t,:,0]-teacher[t-1,:,0])>=np.float32(.8))
                patch.update((d['date']+':'+str(left)+':'+str(first)+':'+str(last)+';').encode())
        for suffix,counters,digest in [('_exposure.npz',cells,record['exposure_sha256']),('_pair_exposure.npz',pairs,record['pair_exposure_sha256'])]:
            path=source/(role+suffix)
            if sha256_file(path)!=digest:raise ValueError('Actual counters changed')
            with np.load(path) as a:
                if any(not np.array_equal(v,a[k]) for k,v in counters.items()):raise ValueError('Rebuilt counters differ')
        with np.load(control_folder/(paired+'_exposure.npz')) as a:
            if any(not np.array_equal(v,a[k]) for k,v in cells.items()):raise ValueError('Extra supervised cells or changed exposure')
        if patch.hexdigest()!=record['patch_sequence_sha256'] or patch.hexdigest()!=control['training'][paired]['patch_sequence_sha256']:raise ValueError('Core sequence changed')
        if sum(int(v.sum()) for v in cells.values())!=record['total_label_exposures'] or sum(int(v.sum()) for v in pairs.values())!=record['total_jump_pair_exposures']:raise ValueError('Recorded exposure totals differ')
        checks.append({'role':role,'initial_patch_and_full_cell_exposure_equal_control':True,'new_pair_exposure_exact':True,'total_pair_exposures':record['total_jump_pair_exposures']})
    models=load_exports(study);evaluator=PublicDays(cfg['panel'],models['nn_primary'].profile,models['nn_primary'].feature_version);calls=[];rows={};maximum=maximum_score=0.
    for seed in cfg['seeds']:
        role=f'jump_seed{seed}';record=study['training'][role]
        if sha256_file(source/('State_'+role+'.pkl'))!=record['model_sha256']:raise ValueError('Per-fit model export changed')
        selected=torch.load(source/(role+'.pt'),map_location='cpu',weights_only=True)
        if set(selected)!=set(models[role].state) or any(not torch.equal(v,models[role].state[k]) for k,v in selected.items()) or models[role].alpha!=study['selection']['choices'][role]['alpha']:raise ValueError('Chosen checkpoint/alpha differs from export')
    for record in study['prediction_files']:
        part,date=record['part'],record['date'];path=cache/part/(date+'.npz')
        if sha256_file(path)!=study['cache_sha256'][path.as_posix()] or sha256_file(record['path'])!=record['sha256']:raise ValueError('Evaluation bytes changed')
        day=evaluator.load(date,path);metrics=study['selection']['cal_metrics'] if part=='selection' else study['assessment']
        with np.load(record['path']) as saved:
            for role,model in models.items():
                key=study['selection']['primary'] if role=='primary_export' else role
                p=model.predict(day['features'],day['template'])[VALUES].to_numpy(float);p[:,1]/=day['target_link_lanes'];error=float(np.abs(p-saved[key]).max());maximum=max(maximum,error)
                if error>1e-8:raise ValueError('Saved predictor differs')
                calls.append({'part':part,'date':date,'role':role,'maximum_lane_unit_error':error})
            for role in metrics:
                p=saved[role];v=p[:,0]-day['station_truth'][:,0];q=(p[:,1]-day['station_truth'][:,1])*day['target_link_lanes']/day['station_lanes'];n=len(v)
                row={'date':date,'regime':day['regime'],'n':n,'speed_sse':float(v@v),'lane_flow_sse':float(q@q)}
                score=.54*max(0,1-np.sqrt(row['speed_sse']/n)/25)+.46*max(0,1-np.sqrt(row['lane_flow_sse']/n)/600);wanted=next(r for r in metrics[role]['days'] if r['date']==date)
                error=abs(score-wanted['daily_State']);maximum_score=max(maximum_score,error)
                if error>1e-12 or n!=wanted['n'] or any(not np.isclose(row[k],wanted[k],rtol=1e-12,atol=1e-6) for k in ['speed_sse','lane_flow_sse']):raise ValueError('Saved score differs')
                rows.setdefault((part,role),[]).append(row)
    for (part,role),values in rows.items():
        wanted=(study['selection']['cal_metrics'] if part=='selection' else study['assessment'])[role]['S_state']
        if abs(state_from_sufficient_stats(values)['S_state']-wanted)>1e-12:raise ValueError('Pooled score differs')
    a=state_from_sufficient_stats(rows[('selection','nn_primary')]);b=state_from_sufficient_stats(rows[('selection','jump_seed_mean')]);gain=b['S_state']-a['S_state'];minimum=min(v['S_state']-u['S_state'] for u,v in zip(a['details'],b['details']))
    if ('jump_seed_mean' if gain>=.0005 and minimum>=-.001 else 'nn_primary')!=study['selection']['primary']:raise ValueError('Cal gate differs')
    result={'registered':registered,'source_snapshot':snapshot,'exposure_checks':checks,'n_direct_predictor_calls':len(calls),'maximum_prediction_error_lane_units':maximum,
        'n_saved_role_day_metrics_recalculated':sum(map(len,rows.values())),'maximum_score_error':maximum_score,'prediction_checks':calls,'seconds':time.perf_counter()-start,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'scope':'All cell/pair counters and selected deployment/scoring verified. Not control refit or replay of every unselected checkpoint/alpha.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('speed_jump_verified','direct',len(calls),'maxpred',maximum,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
