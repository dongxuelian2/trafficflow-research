"""Bounded non-prefix kernel versus exact optima in self-defined small laws."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.scenario_local_search import bounded_scenario_search,scenario_set_scores
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot


def evaluate(name,y,m,e,w,p):
    labels=np.asarray(y,bool);mask=np.asarray(m,bool);evidence=np.asarray(e,bool);weights=np.asarray(w,float);weights/=weights.sum();prob=np.asarray(p,float);links=np.array([str(i) for i in range(len(prob))]);candidate,record=bounded_scenario_search(prob,links,labels,mask,evidence,weights)
    values=[float(weights@scenario_set_scores(pred,labels,mask,evidence)) for pred in itertools.product([0,1],repeat=len(prob))];maximum=max(values)
    if record['expected_proxy_IoU']<record['initial_expected_proxy_IoU']-1e-12 or record['expected_proxy_IoU']>maximum+1e-12:raise ValueError('Search monotonicity/global upper bound failed')
    return {'name':name,'prediction':candidate.tolist(),'trace':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in record.items()},'n_candidate_sets_enumerated':len(values),'known_global_optimum_diagnostic':{'expected_IoU':maximum,'deployment_allowed':False},'remaining_gap':maximum-record['expected_proxy_IoU']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite local-search known laws')
    source=Path('reports/research/C78_nonempty_joint_mask_stress.json');prior=Path('reports/research/C81_joint_scenario_matched_summary.json');stress=json.loads(source.read_text());counter=json.loads(prior.read_text())['restricted_prefix_counterexample']
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Bounded_local_search_known_joint_laws','source_nonempty_laws_sha256':sha256_file(source),'source_counterexample_sha256':sha256_file(prior),'max_moves':8,'min_step':1e-6,
        'scope':'Own exact small distributions;global enumeration diagnostic,not a traffic score. Local search only single-bit strict improvements,not claimed generally global. No parameter grid or competition labels.', 'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();cases=[]
    for row in stress['cases']:
        s=row['hypothetical_scenarios'];cases.append(evaluate(row['name'],s['labels'],s['last_eligible'],s['evidence'],s['weights'],row['marginal_truth_probability']))
    law=counter['self_defined_distribution'];y=np.array(law['labels'],bool);w=np.array(law['weights']);cases.append(evaluate('nonprefix_counterexample',y,np.ones(y.shape,bool),np.ones(len(y),bool),w,w@y))
    if abs(cases[-1]['remaining_gap'])>1e-12 or abs(cases[-1]['trace']['expected_proxy_IoU']-.4425)>1e-12:raise ValueError('Known non-prefix gap not closed')
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'n_cases':len(cases),'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,'scope':'Finite-law correctness only;not evidence of real Queue improvement or general global convergence.'};raw=json.dumps(result,indent=2,allow_nan=False)
    with args.output.open('x',encoding='utf-8') as f:f.write(raw)
    with (folder/'experiment.json').open('x',encoding='utf-8') as f:f.write(raw)
    print('local_search_known_laws_COMPLETE',[(c['name'],round(c['remaining_gap'],9)) for c in cases],flush=True)


if __name__=='__main__':main()
