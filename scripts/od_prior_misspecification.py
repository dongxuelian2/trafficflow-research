"""Fixed-model stress test with independent truth and reverse-generated priors."""
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.data import Release
from trafficflow.odme import operator, solve
from trafficflow.bayesian_od import posterior
from trafficflow.metrics import od_link_score
from trafficflow.runtime import evidence
from od_probability_experiment import scores


def generate(A, public_prior, paths, seed, regime):
    rng=np.random.default_rng(seed)
    scale=max(float(np.mean(public_prior)),2.)
    truth=rng.gamma(.7,scale/.7,A.shape[1])
    if regime=='absolute_additive_prior_noise':
        b=np.maximum(0,truth+rng.normal(0,.75*scale,len(truth)))
    elif regime=='lognormal_prior_noise':
        b=truth*rng.lognormal(-.5,1.,len(truth))
    elif regime=='missing_prior_paths':
        b=truth*rng.lognormal(-.5*.4**2,.4,len(truth))
        b[rng.random(len(truth))<.35]=0.
    elif regime=='group_biased_prior':
        b=truth.copy()
        for key in ('origin_zone','destination_zone'):
            labels=paths[key].to_numpy(str)
            factors={key:rng.lognormal(-.5*.8**2,.8) for key in sorted(set(labels))}
            b*=np.array([factors[key] for key in labels])
    else:raise ValueError(regime)
    clean=A@truth
    counts=np.maximum(0,clean+rng.normal(0,.02*np.maximum(clean,10)))
    return truth,counts,b


def main():
    output=Path('reports/research/D04_broad_prior_misspecification.json')
    if output.exists():raise FileExistsError('Do not overwrite failed or successful evidence')
    registered=json.loads(Path('reports/research/D04_broad_prior_registered.json').read_text())
    d02=json.loads(Path('reports/research/D02_probabilistic_OD_deterministic.json').read_text())
    release=Release('data/kaggle_public')
    start=time.perf_counter()
    result={'registered':registered,'source_at_start':evidence(),'rows':[],
        'formal_ODME':None,'data':'Independent gamma path truth; synthetic weak priors generated from truth, public train network only',
        'no_model_retuning':True,'no_new_formal_upload':True}
    for panel in release.panels():
        A,_,public_prior,paths=operator(release,panel,'train')
        settings=d02['grid'][d02['panels'][panel]['selected_grid']]
        for seed in registered['protocol']['assessment_seeds']:
            for ri,regime in enumerate(registered['protocol']['generating_regimes']):
                truth,c,b=generate(A,public_prior,paths,seed+100*ri,regime)
                for method in ('ridge_0.05','D02_bayesian_fixed_hyperparameters'):
                    tic=time.perf_counter()
                    try:
                        if method=='ridge_0.05':
                            f,info=solve(A,c,b,'ridge',.05)
                            sd=None
                        else:
                            f,sd,info=posterior(A,c,b,paths,**settings)
                        metrics=scores(A,c,b,truth,f,paths)
                        if sd is not None:
                            metrics['approximate_interval_coverage']=float((np.abs(f-truth)<=1.96*sd).mean())
                        row={'success':True,**metrics,**info}
                    except RuntimeError as error:
                        row={'success':False,'error':str(error),'simulated_S_ODME':None}
                    result['rows'].append({'panel':panel,'seed':seed,'prior_regime':regime,'method':method,
                        'prior_S_link':od_link_score(A,b,c),'seconds':time.perf_counter()-tic,**row})
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(panel,'complete',flush=True)
    result['seconds']=time.perf_counter()-start
    result['summary']={regime:{method:float(np.mean([r['simulated_S_ODME'] for r in result['rows']
        if r['prior_regime']==regime and r['method']==method and r['success']]))
        for method in ('ridge_0.05','D02_bayesian_fixed_hyperparameters')}
        for regime in registered['protocol']['generating_regimes']}
    result['optimizer_failures']=sum(not r['success'] for r in result['rows'])
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result['summary'],result['optimizer_failures'],flush=True)


if __name__=='__main__':main()
