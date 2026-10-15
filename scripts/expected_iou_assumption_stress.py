"""Exact known-joint toy stress: independence and unknown future eligibility."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.expected_iou_queue import independent_iou_set
from trafficflow.runtime import capture_source_snapshot,sha256_file
from trafficflow.research_eval import register_protocol


def independent_weights(states,p):return np.prod(np.where(states,p,1-np.array(p)),axis=1)


def known_joint(name,p,kind='independent',positive_coverage=None,negative_coverage=None):
    n=len(p);states=np.array(list(itertools.product([False,True],repeat=n)));truth=independent_weights(states,p)
    if kind=='all_or_nothing':truth=np.zeros(len(states));truth[0]=.7;truth[-1]=.3
    if kind=='cluster_switch_nonempty':
        truth=np.zeros(len(states));truth[np.all(states==[1,1,0,0,0],axis=1)]=.6;truth[np.all(states==[0,0,1,1,1],axis=1)]=.4
    positive=np.ones(n) if positive_coverage is None else np.array(positive_coverage,float);negative=np.ones(n) if negative_coverage is None else np.array(negative_coverage,float)
    rates=np.where(states,positive,negative);mask_probability=np.prod(np.where(states[None,:,:],rates[:,None,:],1-rates[:,None,:]),axis=2);joint=truth[:,None]*mask_probability
    if abs(joint.sum()-1)>1e-12:raise ValueError('Known distribution must normalize')
    marginal=truth@states
    def score(pred):
        intersection=np.sum((states[:,None,:]&pred)&states[None,:,:],axis=2);union=np.sum((states[:,None,:]|pred)&states[None,:,:],axis=2)
        value=np.divide(intersection,union,out=np.ones(union.shape),where=union>0);value[:,states.sum(axis=1)==0]=0
        return float(np.sum(joint*value))
    links=np.array([str(i) for i in range(n)]);iid,detail=independent_iou_set(marginal,links);forced,_=independent_iou_set(marginal,links,allow_empty=False);threshold=marginal>=.5
    values=np.array([score(pred) for pred in states]);maximum=float(values.max());best=int(np.flatnonzero(values>=maximum-1e-12)[0]);result={'name':name,'kind':kind,'n_nodes':n,'truth_marginals':marginal.tolist(),
        'positive_coverage':positive.tolist(),'negative_coverage':negative.tolist(),'n_truth_mask_states':int(np.count_nonzero(joint)),'n_candidate_sets_enumerated':len(states),
        'independent_decoder':{'prediction':iid.tolist(),'model_expected_full_domain_IoU':detail['expected_iou'],'true_known_joint_expected_masked_IoU':score(iid),'regret_to_known_joint_optimum':maximum-score(iid)},
        'nonempty_control':{'prediction':forced.tolist(),'true_known_joint_expected_masked_IoU':score(forced)},'threshold_05':{'prediction':threshold.astype(int).tolist(),'true_known_joint_expected_masked_IoU':score(threshold)},
        'known_joint_distribution_optimum_diagnostic':{'prediction':states[best].astype(int).tolist(),'expected_masked_IoU':maximum,'deployment_allowed':False}}
    if kind=='independent' and np.all(positive==1) and np.all(negative==1) and abs(score(iid)-maximum)>1e-12:raise ValueError('IID decoder fails exhaustive known IID optimum')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite known-joint stress')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Exact_known_joint_assumption_stress','description':'Own5node finite distributions. Perfect true marginals supplied to IID decoder;enumerate all32prediction sets and known truth/mask distributions,never choose using a realized hidden answer.',
        'scenarios':['IID_sparse','IID_mixed','IID_dense','correlated_all_or_nothing','correlated_nonempty_cluster_switch','IID_nonuniform_missing','IID_label_dependent_missing'],
        'scope':'Toy mathematical assumption check,not a traffic simulator or model validation/Kaggle score. Known-joint distribution optimum diagnostic prohibited from deployment. No calibration or policy changes from results.',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    cases=[known_joint('IID_sparse',[.15,.12,.08,.04,.02]),known_joint('IID_mixed',[.8,.55,.35,.2,.05]),known_joint('IID_dense',[.9,.75,.65,.6,.45]),
        known_joint('correlated_all_or_nothing',[.3]*5,'all_or_nothing'),known_joint('correlated_nonempty_cluster_switch',[.6,.6,.4,.4,.4],'cluster_switch_nonempty'),
        known_joint('IID_nonuniform_missing',[.85,.6,.4,.2,.05],positive_coverage=[.1,1,1,1,1],negative_coverage=[.1,1,1,1,1]),
        known_joint('IID_label_dependent_missing',[.85,.6,.4,.2,.05],positive_coverage=[.1,.25,.8,.8,.8],negative_coverage=[.95]*5)]
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'n_cases':len(cases),'all_3_full_coverage_IID_cases_match_exhaustive_optimum':True,'seconds':time.perf_counter()-start,
        'formal_Queue':None,'formal_Total':None,'scope':'Known self-defined5node laws,exact expected scores. Not latent traffic truth or formal Queue. Correlated/missing cases test limitations of an independent/full-domain model even with perfect marginal probabilities.'}
    raw=json.dumps(result,indent=2,allow_nan=False)
    with args.output.open('x',encoding='utf-8') as f:f.write(raw)
    with (folder/'experiment.json').open('x',encoding='utf-8') as f:f.write(raw)
    print('known_joint_stress_COMPLETE',[(r['name'],r['independent_decoder']['regret_to_known_joint_optimum']) for r in cases],flush=True)


if __name__=='__main__':main()
