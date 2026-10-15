"""Rebuild raw TRAIN teachers and all cal/assessment channel choices independently."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.data import Release,VALUES
from trafficflow.state import StateModel
from trafficflow.experiments import partition_date,day_template,target_truth
from trafficflow.research_eval import register_protocol,state_from_sufficient_stats
from trafficflow.runtime import sha256_file,capture_source_snapshot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite channel verification')
    study=json.loads(args.study.read_text());cfg=study['registered']['protocol']
    source=Path('artifacts/research')/args.study.stem;seal=source/'selection_sealed.json'
    if sha256_file(seal)!=study['selection_seal_sha256']:raise ValueError('Selection seal changed')
    if json.loads(seal.read_text())!=study['selection']:raise ValueError('Selection report differs from seal')
    if study['phase_read_order']!=[{'part':'selection','prior_selection_seal_sha256':None},
            {'part':'assessment','prior_selection_seal_sha256':study['selection_seal_sha256']}]:raise ValueError('Phase order differs')
    for path,digest in study['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Component/cache predictions changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Independent_channel_cal_grid_and_assessment_verification','source_study_sha256':sha256_file(args.study),
        'scope':'Raw TRAIN teachers/lanes,all saved channel formulas and State metrics,all50cal choices,serialized weights. No retraining or new direct predictions.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');panel=cfg['panel'];net=release.network(panel)
    files={partition_date(release,p):p for p in release.files(panel,'train')};raw_files={p.name:p for p in release.files(panel,'train',False)}
    template=release.template(1,panel,'train');cal=[];checks=[];maximum=0.;count=0
    chosen=study['selection']['choice'];roles=list(study['metrics']['selection'])
    def row(date,regime,truth,prediction,lanes,station):
        v=prediction[:,0]-truth[:,0];q=(prediction[:,1]-truth[:,1])*lanes/station;n=len(v)
        sv=float(v@v);sq=float(q@q)
        return {'date':date,'regime':regime,'n':n,'speed_sse':sv,'lane_flow_sse':sq,
            'daily_State':.54*max(0,1-np.sqrt(sv/n)/25)+.46*max(0,1-np.sqrt(sq/n)/600)}
    all_rows={part:{r:[] for r in roles} for part in ['selection','assessment']}
    for record in study['prediction_files']:
        part,date=record['part'],record['date'];path=Path(record['path'])
        if sha256_file(path)!=record['sha256']:raise ValueError('Saved channel arrays changed')
        frame=release.read(files[date]);raw=release.read(raw_files[files[date].name]);target=day_template(release,panel,'train',frame,template)
        truth=target_truth(release,panel,target,raw)[VALUES].to_numpy(float);lanes=target.link_id.map(net.lanes).to_numpy(float)
        truth[:,1]/=lanes;station=release.station_lanes(panel,target);regime=frame.mask_regime.iloc[0]
        with np.load(path) as a:
            arrays={r:a[r] for r in roles}
        if chosen:
            weight=np.array([chosen['speed_tree_weight'],chosen['flow_tree_weight']])
            expected=(1-weight)*arrays['nn_primary']+weight*arrays[chosen['variant']+'_seed_mean']
        else:expected=arrays['nn_primary']
        if not np.array_equal(expected,arrays['primary_export']):raise ValueError('Channel formula differs')
        for role,prediction in arrays.items():
            actual=row(date,regime,truth,prediction,lanes,station)
            wanted=next(r for r in study['metrics'][part][role]['days'] if r['date']==date)
            error=abs(actual['daily_State']-wanted['daily_State']);maximum=max(maximum,error)
            if actual['n']!=wanted['n'] or error>1e-12 or any(not np.isclose(actual[k],wanted[k],rtol=1e-12,atol=1e-6) for k in ['speed_sse','lane_flow_sse']):raise ValueError('Raw saved metric differs')
            all_rows[part][role].append(actual);count+=1
        if part=='selection':cal.append((date,regime,truth,lanes,station,arrays))
        checks.append({'part':part,'date':date,'raw_teacher_metrics_and_channel_formula_exact':True})
    for part in all_rows:
        for role,rows in all_rows[part].items():
            if abs(state_from_sufficient_stats(rows)['S_state']-study['metrics'][part][role]['S_state'])>1e-12:raise ValueError('Pooled State differs')
    baseline=state_from_sufficient_stats(all_rows['selection']['nn_primary']);rebuilt=[]
    for candidate in study['selection']['cal_grid']:
        w=np.array([candidate['speed_tree_weight'],candidate['flow_tree_weight']]);rows=[]
        for date,regime,truth,lanes,station,arrays in cal:
            p=(1-w)*arrays['nn_primary']+w*arrays[candidate['variant']+'_seed_mean'];rows.append(row(date,regime,truth,p,lanes,station))
        metrics=state_from_sufficient_stats(rows);gain=metrics['S_state']-baseline['S_state']
        minimum=min(a['S_state']-b['S_state'] for a,b in zip(metrics['details'],baseline['details']))
        if abs(gain-candidate['cal_gain'])>1e-12 or abs(minimum-candidate['minimum_regime_cal_gain'])>1e-12 or candidate['eligible']!=(gain>=.0005 and minimum>=-.001):raise ValueError('Cal grid/eligibility differs')
        rebuilt.append(candidate)
    eligible=[r for r in rebuilt if r['eligible']]
    winner=max(eligible,key=lambda r:(r['cal']['S_state'],-r['speed_tree_weight']-r['flow_tree_weight'])) if eligible else None
    if winner!=chosen:raise ValueError('Cal-only choice differs')
    export=study['primary_export']
    if sha256_file(export['path'])!=export['sha256']:raise ValueError('Saved model changed')
    model=StateModel.load(Path(export['path']))
    if chosen:
        w=np.array([chosen['speed_tree_weight'],chosen['flow_tree_weight']])
        if not np.array_equal(model.weights,np.array([1-w,w])):raise ValueError('Serialized channel weights differ')
    result={'registered':registered,'source_snapshot':snapshot,'n_raw_teacher_days_rebuilt':len(checks),
        'n_saved_role_day_metrics_recalculated':count,'n_cal_grid_candidates_rebuilt':len(rebuilt),'maximum_score_error':maximum,
        'cal_choice_and_serialized_weights_exact':True,'day_checks':checks,'seconds':time.perf_counter()-start,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'scope':'Independent teachers/scoring/choice audit,not independent retraining or new deployment.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('channel_verified',len(checks),'days',count,'metrics',len(rebuilt),'cal choices','maxscore',maximum,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
