"""Same-day multi-fit-seed comparison against the production-like 500k budget."""
import json
from pathlib import Path
import numpy as np
from trafficflow.data import Release,VALUES
from trafficflow.state import StateFeatures,StateModel
from trafficflow.experiments import partition_date,day_template
from trafficflow.research_eval import state_from_sufficient_stats,paired_block_interval
from train_graph_experiment import stats


if __name__=='__main__':
    output=Path('reports/research/A24_A28_multiseed_budget_comparison.json')
    if output.exists():raise FileExistsError('Never overwrite matched evidence')
    cache=Path('artifacts/research/A01_D12_I5_N_autumn_seed20261013')
    cases=[(20261013,'A24_autumn_500k_LGB_control','A14_full_label_LGB_control'),
           (20261029,'A26_autumn_500k_LGB_seed20261029','A25_autumn_full_LGB_seed20261029'),
           (20261107,'A28_autumn_500k_LGB_seed20261107','A27_autumn_full_LGB_seed20261107')]
    models={}
    for seed,old,new in cases:
        models[seed]=[StateModel.load(Path('artifacts/research')/name/'lightgbm_large_budget.pkl') for name in (old,new)]
    release=Release('data/kaggle_public')
    panel='D12_I5_N'
    net,topo=release.network(panel),release.topology(panel)
    template=release.template(1,panel,'train')
    paths={partition_date(release,p):p for p in release.files(panel,'train')}
    profile=models[cases[0][0]][0].profile
    for pair in models.values():
        for model in pair:
            if not model.profile.slots.equals(profile.slots) or not model.profile.fallback.equals(profile.fallback):
                raise ValueError('Budget comparison must use an identical earlier profile')
    rows={s:[[],[]] for s in models}
    for path in sorted((cache/'assessment').glob('*.npz')):
        day=dict(np.load(path,allow_pickle=False))
        frame=release.read(paths[path.stem])
        f=StateFeatures(frame,net,topo,profile,version=2)
        target=day_template(release,panel,'train',frame,template)
        if not np.array_equal(f.lookup(target),day['target_indices']):raise ValueError('Target order changed')
        lanes=target.link_id.map(net.lanes).to_numpy(float)
        vf=target.link_id.map(net.free_speed_kmh).to_numpy(float)
        capacity=target.link_id.map(net.capacity_vph).to_numpy(float)
        for seed,pair in models.items():
            for i,model in enumerate(pair):
                p=model.predict(f,target)[VALUES].to_numpy(float)
                p[:,0]=np.minimum(p[:,0],vf*1.05)
                p[:,1]=np.minimum(p[:,1],capacity*1.05)/lanes
                rows[seed][i].append(stats(day,p))
    results=[]
    for seed,pair in rows.items():
        old,new=[state_from_sufficient_stats(r) for r in pair]
        results.append({'fit_seed':seed,'500k_assessment':old,'full_assessment':new,
            'pooled_State_gain':new['S_state']-old['S_state'],
            'paired_day_interval':paired_block_interval([r['daily_State'] for r in pair[0]],[r['daily_State'] for r in pair[1]])})
    mean_old=np.mean([[r['daily_State'] for r in pair[0]] for pair in rows.values()],axis=0)
    mean_new=np.mean([[r['daily_State'] for r in pair[1]] for pair in rows.values()],axis=0)
    result={'panel':panel,'fold':'rolling_autumn','fit_seeds':[r['fit_seed'] for r in results],
        'rows':results,'mean_pooled_State_gain':float(np.mean([r['pooled_State_gain'] for r in results])),
        'minimum_seed_pooled_gain':float(min(r['pooled_State_gain'] for r in results)),
        'paired_day_interval_after_mean_over_fit_seeds':paired_block_interval(mean_old,mean_new),
        'formal_State':None,'formal_Total':None,
        'limitations':'Same 14 assessment days for all fit seeds, not 42 independent days. This is already-inspected public-train development evidence; comparison itself does not select an architecture.'}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result['mean_pooled_State_gain'],result['paired_day_interval_after_mean_over_fit_seeds'],flush=True)
