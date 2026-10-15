"""Exact latent-nonempty laws: score filtering and observed-positive selection."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.expected_iou_queue import independent_iou_set
from trafficflow.joint_queue_scenarios import scenario_prefix_set
from trafficflow.runtime import capture_source_snapshot
from trafficflow.research_eval import register_protocol


SCENES=[('single_r95','single',.95,.95,True),('single_r75','single',.75,.75,True),('single_r25','single',.25,.25,True),
        ('cluster_full','cluster',1.,1.,True),('cluster_r75','cluster',.75,.75,True),('cluster_r25','cluster',.25,.25,True),
        ('cluster_label_dependent','cluster',.15,.95,True),('cluster_label_dependent_last_evidence','cluster',.15,.95,False),
        ('cluster_node_missing','cluster',[.1,.1,1.,1.,1.],[.1,.1,1.,1.,1.],True)]


def distribution(kind,positive,negative,early_evidence):
    n=5;masks=np.array(list(itertools.product([False,True],repeat=n)));ys=np.eye(n,dtype=bool) if kind=='single' else np.array([[1,1,0,0,0],[0,0,1,1,1]],bool);py=np.ones(n)/n if kind=='single' else np.array([.6,.4]);p=np.broadcast_to(positive,(n,));q=np.broadcast_to(negative,(n,));rates=np.where(ys,p,q);weights=py[:,None]*np.prod(np.where(masks[None,:,:],rates[:,None,:],1-rates[:,None,:]),axis=2)
    y=np.repeat(ys,len(masks),axis=0);m=np.tile(masks,(len(ys),1));e=np.ones(len(m),bool) if early_evidence else m.any(axis=1);w=weights.ravel();use=w>0
    return y[use],m[use],e[use],w[use],py@ys


def run(name,kind,positive,negative,early):
    y,m,e,w,p=distribution(kind,positive,negative,early);n=len(p);links=np.array([str(i) for i in range(n)]);all_sets=np.array(list(itertools.product([False,True],repeat=n)))
    def scores(pred):
        intersection=np.sum((y&pred)&m,axis=1);union=np.sum((y|pred)&m,axis=1);return np.divide(intersection,union,out=e.astype(float),where=union>0)
    observed=(y&m).any(axis=1);proxy_weight=w*observed;proxy_weight/=proxy_weight.sum();values=np.array([np.dot(w,scores(a)) for a in all_sets]);proxy_values=np.array([np.dot(proxy_weight,scores(a)) for a in all_sets]);best=int(np.argmax(values));proxy_best=int(np.argmax(proxy_values))
    independent,_=independent_iou_set(p,links);forced,_=independent_iou_set(p,links,allow_empty=False);joint,detail=scenario_prefix_set(p,links,y,m,e,w)
    policies={'empty':np.zeros(n,int),'iid_full':independent,'iid_nonempty':forced,'joint_prefix_maskaware':joint,'observed_positive_distribution_selected':all_sets[proxy_best].astype(int)};result={}
    for role,pred in policies.items():result[role]={'prediction':pred.tolist(),'latent_population_expected_scored_IoU':float(np.dot(w,scores(pred))),'observed_positive_subset_expected_IoU':float(np.dot(proxy_weight,scores(pred)))}
    return {'name':name,'kind':kind,'positive_coverage':np.broadcast_to(positive,(n,)).tolist(),'negative_coverage':np.broadcast_to(negative,(n,)).tolist(),'earlier_five_steps_have_evidence':early,
        'marginal_truth_probability':p.tolist(),'all_latent_labels_nonempty':bool(y.any(axis=1).all()),'probability_scored_positive':float(w[observed].sum()),'probability_any_evidence':float(w[e].sum()),
        'n_positive_probability_scenarios':len(w),'hypothetical_scenarios':{'labels':y.astype(int).tolist(),'last_eligible':m.astype(int).tolist(),'evidence':e.astype(int).tolist(),'weights':w.tolist()},'policies':result,
        'joint_prefix_expected_curve':detail['curve'].tolist(),'known_joint_global_optimum_diagnostic':{'expected_IoU':float(values[best]),'prediction':all_sets[best].astype(int).tolist(),'deployment_allowed':False},
        'restricted_prefix_gap':float(values[best]-detail['expected_proxy_IoU']),'proxy_selected_population_regret':float(values[best]-np.dot(w,scores(all_sets[proxy_best])))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite nonempty-mask stress')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'Exact_latent_nonempty_joint_masks_and_proxy_selection','scenes':SCENES,
        'scope':'Own5node laws,latent truth always nonempty;future eligibility random,earlier five steps known zero labels with/without evidence. All decisions depend only on known generating law,marginal p,never realized future label/mask. Not a traffic simulator/official truth.',
        'new_question':'Extend C40/C76 to exact joint expected masked-IoU optimization,explicit six-step evidence,and transfer of observed-positive-subset-selected policy back to latent-positive population.',
        'previous_goal_turn':'progress:dfd24e7 completed probability calibration/IoU and known-joint audits','formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();cases=[run(*scene) for scene in SCENES]
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'n_cases':len(cases),'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,
        'scope':'Exact expectations over autonomous5node finite laws. Joint known-law optimum diagnostic is not a competition model and not a realized-future oracle;no policy exports or real hidden-label claims.'}
    raw=json.dumps(result,indent=2,allow_nan=False)
    with args.output.open('x',encoding='utf-8') as f:f.write(raw)
    with (folder/'experiment.json').open('x',encoding='utf-8') as f:f.write(raw)
    print('nonempty_joint_stress_COMPLETE',[(c['name'],round(c['proxy_selected_population_regret'],6),round(c['restricted_prefix_gap'],6)) for c in cases],flush=True)


if __name__=='__main__':main()
