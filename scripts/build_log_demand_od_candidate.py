"""Package a full LOCAL OD-only control after numeric repair,never upload."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from trafficflow.data import Release,OD_KEY
from trafficflow.odme import operator
from trafficflow.log_demand_od import log_demand_inverse
from trafficflow.metrics import od_link_score
from trafficflow.submission import validate_task,merge_files
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--numeric-verification',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite full candidate')
    study=json.loads(args.study.read_text());selected=study['selection']['best_hypothesis']
    study_folder=Path('artifacts/research')/args.study.stem
    if json.loads((study_folder/'selection_sealed.json').read_text())!=study['selection']:
        raise ValueError('Sealed development choice changed')
    if not study['selection']['development_gate_passed'] or study['optimizer_case_failures']:
        raise ValueError('Complete numerical success and sealed development selection required')
    numeric=json.loads(args.numeric_verification.read_text())
    relevant=[r for r in numeric['rows'] if r['setting_index']==selected['index']]
    if len(relevant)!=6 or not all(r['independent_solutions_agree'] for r in relevant):raise ValueError('Selected setting independent numerical check incomplete')
    frozen=Path('artifacts/frozen_baselines/57138874/submission.csv')
    frozen_sha='43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1'
    if sha256_file(frozen)!=frozen_sha:raise ValueError('Frozen full formal baseline changed')
    baseline=Path('artifacts/submission_best');expected={'state':'49c48651db680a46a05f952ae4a89001374ea5912d73627d9a2ad8117e53a294',
        'queue':'420080d4582ff74f96a33bc5ca726aedb541ca3f576ced416be307d6c22d3fa4',
        'odme':'7091490ad973fe0ebc017f1a69523441fce330d1d731fcda02ba7f6ff4bd5f83'}
    if {k:sha256_file(baseline/(k+'.csv')) for k in expected}!=expected:raise ValueError('Frozen original component changed')
    registered=register_protocol(Path('reports/research')/(args.output.name+'.registered.json'),{
        'id':'Log_demand_OD_only_complete_LOCAL_control','baseline_ref':57138874,'baseline_Total':.82940,
        'baseline_sha256':frozen_sha,
        'study_sha256':sha256_file(args.study),'numeric_verification_sha256':sha256_file(args.numeric_verification),
        'settings':selected['settings'],'scope':'All released validation/private OD paths only;State/Queue/Physics inputs retained',
        'hypothesis':'Coherent multiplicative zone demand factors transfer beyond independent path priors',
        'limitations':'Synthetic gains and local numerical verification do not establish actual OD recovery. K01 Bayes formal failure retained.',
        'uploaded':False,'formal_ODME':None,'formal_Total':None})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    release=Release('data/kaggle_public');predictions=[];diagnostics=[];input_hashes={}
    with threadpool_limits(limits=4):
        for panel in release.panels():
            for split in ['train','validation','private']:
                A,c,b,paths=operator(release,panel,split)
                f,info=log_demand_inverse(A,c,b,paths,**selected['settings'])
                diagnostics.append({'panel':panel,'split':split,'public_count_S_link_only':od_link_score(A,f,c),
                    'relative_prior_L1_change':float(np.abs(f-b).sum()/max(b.sum(),1e-9)),
                    'total_flow_over_prior':float(f.sum()/max(b.sum(),1e-9)),'solver_info':info,
                    'formal_S_od':None,'formal_S_attr':None,'formal_ODME':None})
                for path in [release.root/'task4'/panel/split/'synthetic_link_counts.csv',release.root/'task4'/panel/split/'synthetic_weak_prior.csv',
                    release.root/'corridors'/panel/'network'/'path_set.csv',release.root/'corridors'/panel/'network'/'path_link_incidence.csv']:
                    input_hashes[str(path)]=sha256_file(path)
                if split=='train':continue
                template=release.template(4,panel,split);ids=paths.set_index('path_id')
                if set(template.path_id)!=set(paths.path_id):raise ValueError('OD template coverage differs')
                if any(not np.array_equal(template[z].to_numpy(),template.path_id.map(ids[z]).to_numpy()) for z in ['origin_zone','destination_zone']):
                    raise ValueError('OD zone identity differs')
                result=template[OD_KEY].copy();result['path_flow']=result.path_id.map(dict(zip(paths.path_id,f)));predictions.append(result)
            print(panel,'all public splits solved',flush=True)
    od=pd.concat(predictions,ignore_index=True);validate_task(od,'odme');od_path=args.output/'odme.csv';od.to_csv(od_path,index=False)
    files={'state':baseline/'state.csv','queue':baseline/'queue.csv','odme':od_path}
    coverage=merge_files(files,release.root/'submission_key.csv',args.output/'submission.csv')
    verification=dict(coverage=coverage,registered=registered,source_snapshot=snapshot,diagnostics=diagnostics,
        input_files_sha256=input_hashes,component_sha256={k:sha256_file(p) for k,p in files.items()},
        submission_sha256=sha256_file(args.output/'submission.csv'),seconds=time.perf_counter()-start,
        source=evidence(),uploaded=False,formal_ODME=None,formal_Total=None,formal_best_retained=True,
        scope='Complete LOCAL control only;separate full row-wise baseline comparison required before use')
    (args.output/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    report=Path('reports/research')/(args.output.name+'.json')
    if report.exists():raise FileExistsError('Never overwrite a candidate report')
    report.write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print('sha256',verification['submission_sha256'],'coverage',coverage,'seconds',verification['seconds'],'uploaded',False,flush=True)


if __name__=='__main__':
    main()
