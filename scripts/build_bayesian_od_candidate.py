"""OD-only candidate: copy frozen State/Queue, replace OD with Bayesian MAP."""
import argparse
import json
from pathlib import Path
import time
import pandas as pd
from trafficflow.data import Release, OD_KEY
from trafficflow.odme import operator
from trafficflow.bayesian_od import posterior
from trafficflow.metrics import od_link_score
from trafficflow.submission import merge_files
from trafficflow.runtime import evidence, sha256_file
from trafficflow.research_eval import register_protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a candidate')
    experiment=json.loads(Path('reports/research/D02_probabilistic_OD_deterministic.json').read_text())
    hypothesis={'id':'K01_OD_only_hierarchical_Gaussian','baseline_ref':57138874,'baseline_total':.82940,
        'baseline_sha256':'43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1',
        'changed_component':'ODME only','unchanged_components':['State','Queue','Physics (same State input)'],
        'hypothesis':'Shared origin/destination demand variations explain weak-prior errors better than isotropic Ridge .05; posterior regularization trades tiny link-fit error for better path and attraction estimation.',
        'offline_evidence':'D02 independent known-truth simulations, ten public networks, disjoint development/assessment seeds, three generating distributions',
        'expected_formal_gain':None,'formal_components_unobservable':True,
        'selection':{p:experiment['grid'][r['selected_grid']] for p,r in experiment['panels'].items()},
        'decision':'One controlled formal upload. Preserve baseline on loss; no retuning from a single feedback value.'}
    register_protocol(Path('reports/research/K01_OD_only_preregistered.json'),hypothesis)
    args.output.mkdir(parents=True)
    started=time.perf_counter()
    release=Release('data/kaggle_public')
    frames=[]
    info=[]
    for panel in release.panels():
        for split in ('validation','private'):
            A,c,b,paths=operator(release,panel,split)
            f,sd,detail=posterior(A,c,b,paths,**hypothesis['selection'][panel])
            target=release.template(4,panel,split)
            mapping=dict(zip(paths.path_id,f))
            target['path_flow']=target.path_id.map(mapping)
            if target.path_flow.isna().any():
                raise ValueError('OD candidate does not cover template')
            frames.append(target[OD_KEY+['path_flow']])
            info.append({'panel':panel,'split':split,'S_link_only':od_link_score(A,f,c),
                         'formal_ODME':None,**detail})
    od=args.output/'odme.csv'
    pd.concat(frames,ignore_index=True).to_csv(od,index=False)
    original=Path('artifacts/submission_best')
    files={'state':original/'state.csv','queue':original/'queue.csv','odme':od}
    report=merge_files(files,release.root/'submission_key.csv',args.output/'submission.csv')
    report.update({'hypothesis':hypothesis,'OD_diagnostics':info,'source':evidence(),
        'seconds':time.perf_counter()-started,'formal_public_total':None,'uploaded':False,
        'component_sha256':{k:sha256_file(v) for k,v in files.items()},
        'submission_sha256':sha256_file(args.output/'submission.csv')})
    expected={'state':'49c48651db680a46a05f952ae4a89001374ea5912d73627d9a2ad8117e53a294',
              'queue':'420080d4582ff74f96a33bc5ca726aedb541ca3f576ced416be307d6c22d3fa4'}
    if any(report['component_sha256'][key]!=value for key,value in expected.items()):
        raise ValueError('State/Queue changed unexpectedly')
    (args.output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({'sha256':report['submission_sha256'],'seconds':report['seconds'],
           'rows':report.get('rows'),'changed':'ODME only'},flush=True)


if __name__=='__main__':
    main()
