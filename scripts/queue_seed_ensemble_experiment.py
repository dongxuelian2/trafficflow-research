"""Frozen two-seed Queue ensembles with paired-calibration sensitivity control."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.queue import QueueModel
from trafficflow.full_domain_queue import FullDomainQueueModel
from trafficflow.ensemble_queue import EnsembleFullDomainQueueModel
from trafficflow.full_queue_calibration import tune_full_probabilities, full_queue_decisions
from trafficflow.queue_selection_uncertainty import select_with_cal_uncertainty
from trafficflow.queue_domain import score_full_forecast
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol
from compare_queue_events import comparison


PAIRS=[('C32_D12_I5_N_wave_graph_Queue','C47_I5_N_autumn_window_overlap_seed1','C42_I5_N_autumn_window_overlap_seed2'),
       ('C34_D12_I5_N_winter_wave_Queue','C48_I5_N_winter_window_overlap_seed1','C43_I5_N_winter_window_overlap_seed2')]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def metric(probability,data,setting):
    pred=full_queue_decisions(probability,data['window'],data['condition'],data['lead_minutes'],
        setting['thresholds'],setting['require_nonempty'],setting['timing'])
    return score_full_forecast(data['labels'],pred,data['window'],data['condition'],data['score_mask'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report_path=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report_path.exists():raise FileExistsError('Never overwrite ensemble evidence')
    probability_report=Path('reports/research/C38_full_queue_probability_cache.json')
    probability=read(probability_report)
    manifest_path=Path('reports/research/model_artifacts_checkpoint_queue_overlap_v1.json')
    known={r['path']:r['sha256'] for r in read(manifest_path)['artifacts']}
    geometry_integrity=[]
    for study,old_name,_ in PAIRS:
        graph_path=Path('artifacts/research')/study/'wave_30min_geometry.npz'
        model_path=Path('artifacts/research')/old_name/'bce.pkl'
        if sha256_file(model_path)!=known[model_path.as_posix()]:raise ValueError('Archived geometry reference model changed')
        model=QueueModel.load(model_path);graph=dict(np.load(graph_path,allow_pickle=False))
        if set(graph)!=set(model.geometry) or not all(np.array_equal(graph[k],model.geometry[k]) for k in graph):
            raise ValueError('Legacy graph arrays differ from immutable portable model')
        known[graph_path.as_posix()]=sha256_file(graph_path)
        geometry_integrity.append({'path':graph_path.as_posix(),'sha256':known[graph_path.as_posix()],
            'legacy_reference_model_path':model_path.as_posix(),'legacy_reference_model_sha256':known[model_path.as_posix()],
            'all_arrays_exact':True,'scope':'Graph bytes first registered here; arrays equal previously hashed portable model'})
    def array(path):
        path=Path(path)
        if sha256_file(path)!=known[path.as_posix()]:raise ValueError('Frozen source array changed')
        return dict(np.load(path,allow_pickle=False))
    source_reports={name:read(Path('reports/research')/(name+'.json')) for _,old,new in PAIRS for name in [old,new]}
    registered=register_protocol(report_path.with_suffix('.registered.json'),{
        'id':'Frozen_Queue_two_optimizer_seed_score_ensemble','pairs':PAIRS,
        'source_reports_sha256':{name:sha256_file(Path('reports/research')/(name+'.json')) for name in source_reports},
        'source_manifest_sha256':sha256_file(manifest_path),'probability_report_sha256':sha256_file(probability_report),
        'legacy_geometry_integrity':geometry_integrity,
        'sources':['old_bce','new_bce','mean_bce','mean_soft_iou','mean_lovasz'],
        'averaging':'Equal arithmetic mean of two corrected scores; same method across seeds, no extra fitting',
        'alpha_grid':[0.,.25,.5,.75,1.],'calibration':'Full-template threshold/force calibration, old single-seed choices reused and replayed exactly',
        'selection':'Primary: simplest candidate within one paired-bootstrap SE gap of cal argmax; secondary: raw argmax and best mean-BCE',
        'bootstrap':'4000 circular 7-day resamples,seed20261013,all45 cal calendar days including event-empty days',
        'complexity':'number of active neural members,alpha,conditional thresholds,force,prefer BCE,prior old seed',
        'uncertainty_scope':'In-sample sensitivity only: checkpoints and thresholds used full cal. No independent confidence bound.',
        'assessment':'Old inspected public TRAIN development dates; all policy choices sealed before assessment cache load',
        'formal_Queue':None,'formal_Total':None,'uploaded':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    results=[]
    for study,old_name,new_name in PAIRS:
        old,new=[source_reports[name] for name in [old_name,new_name]]
        directories=[Path('artifacts/research')/name for name in [old_name,new_name]]
        records=[r for r in probability['full_probability_caches'] if r['study']==study]
        cal_record=next(r for r in records if r['phase']=='selection')
        cal=array(cal_record['path'])
        cal_p={(kind,i):array(directory/(kind+'_cal_probabilities.npz'))['probability']
               for i,directory in enumerate(directories) for kind in ['bce','soft_iou','lovasz']}
        if not np.array_equal(cal_p['bce',0],cal['neural'][:,2]):raise ValueError('Original BCE score cache does not reproduce')
        def source_score(kind,scores):
            if kind=='old_bce':return scores['bce',0]
            if kind=='new_bce':return scores['bce',1]
            method=kind.removeprefix('mean_')
            return (scores[method,0]+scores[method,1])/2
        tree=old['selection']['tree']['metrics']
        tree_setting=next(r for r in old['selection']['bce']['choices'] if r['alpha']==0.)['setting']
        if metric(cal['base'],cal,tree_setting)!=tree:raise ValueError('Frozen tree cal does not reproduce')
        candidates=[{'name':'tree','variant':'tree','alpha':0.,'setting':tree_setting,'metrics':tree,'complexity':[0,0,1,0,0,0]}]
        for variant in ['old_bce','new_bce','mean_bce','mean_soft_iou','mean_lovasz']:
            p=source_score(variant,cal_p)
            if p.shape!=cal['base'].shape or not np.isfinite(p).all():raise ValueError('Ensemble score domain differs')
            for alpha in [.25,.5,.75,1.]:
                mixed=(1-alpha)*cal['base']+alpha*p
                if variant in ['old_bce','new_bce']:
                    run=old if variant=='old_bce' else new
                    choice=next(r for r in run['selection']['bce']['choices'] if r['alpha']==alpha)
                    setting=choice['setting'];measured=metric(mixed,cal,setting)
                    if measured!=choice['metrics']:raise ValueError('Single seed cal choice did not reproduce')
                else:
                    setting=tune_full_probabilities(mixed,cal['window'],cal['condition'],cal['lead_minutes'],cal['labels'],cal['score_mask'])['selected']
                    measured=metric(mixed,cal,setting)
                members=1 if variant in ['old_bce','new_bce'] else 2
                name=variant+'_'+str(alpha)
                candidates.append({'name':name,'variant':variant,'alpha':alpha,'setting':setting,'metrics':measured,
                    'complexity':[members,alpha,int(setting['mode']=='condition'),int(setting['require_nonempty']),
                                  int(variant in ['mean_soft_iou','mean_lovasz']),int(variant=='new_bce')]})
            print(study,variant,'calibrated',flush=True)
        sensitivity=select_with_cal_uncertainty(candidates,old['dates']['selection'])
        by_name={c['name']:c for c in candidates}
        best_mean_bce=min((c for c in candidates if c['variant'] in ['tree','mean_bce']),
            key=lambda c:(-c['metrics']['proxy_IoU'],tuple(c['complexity']),c['name']))['name']
        # Current old BCE policy was already selected by its original calibration.
        prior_choice=max(old['selection']['bce']['choices'],key=lambda c:(c['metrics']['proxy_IoU'],-c['alpha']))
        prior_name='tree' if prior_choice['alpha']==0 else 'old_bce_'+str(prior_choice['alpha'])
        policies={'regularized':sensitivity['selected'],'argmax':sensitivity['argmax'],'mean_bce':best_mean_bce,
                  'previous_seed_mixture':prior_name,'tree':'tree'}
        token=study.split('_',1)[0]
        seal_path=args.output/(token+'_selection_sealed.json')
        seal_path.write_text(json.dumps({'policies':policies,'settings':{k:by_name[v] for k,v in policies.items()},
            'cal_sensitivity':sensitivity},indent=2,allow_nan=False),encoding='utf-8')
        src=next(r for r in probability['source_records'] if Path(r['report']).stem==study)
        causal=read(Path(src['causal_cache'])/'experiment.json')
        baseline_path=Path('artifacts/research')/Path(causal['registered']['protocol']['matched_study']).stem/'shared_threshold.pkl'
        baseline=QueueModel.load(baseline_path)
        geometry=array(Path('artifacts/research')/study/'wave_30min_geometry.npz')
        exported={}
        for policy,name in policies.items():
            choice=by_name[name];variant=choice['variant'];members=[]
            if variant=='tree':
                model=FullDomainQueueModel(baseline,cal['links'].tolist(),choice['setting'])
            else:
                method='bce' if variant in ['old_bce','new_bce'] else variant.removeprefix('mean_')
                indices=[0] if variant=='old_bce' else [1] if variant=='new_bce' else [0,1]
                for i in indices:
                    pt=directories[i]/(method+'.pt')
                    if sha256_file(pt)!=source_reports[[old_name,new_name][i]]['training'][method]['checkpoint_sha256']:
                        raise ValueError('Source network changed')
                    members.append({'path':pt.as_posix(),'sha256':sha256_file(pt)})
                states=[torch.load(row['path'],map_location='cpu',weights_only=True) for row in members]
                model=EnsembleFullDomainQueueModel(baseline,cal['links'].tolist(),choice['setting'],choice['alpha'],states,geometry)
            model_path=args.output/(token+'_'+policy+'.pkl');model.save(model_path)
            exported[policy]={'path':model_path.as_posix(),'sha256':sha256_file(model_path),'candidate':name,'members':members}
        assessment={'dates':old['dates'],'assessment':[]}
        for record in records:
            if record['phase']!='assessment':continue
            data=array(record['path'])
            p={(kind,i):array(directory/(kind+'_assessment_'+str(record['seed'])+'_probabilities.npz'))['probability']
               for i,directory in enumerate(directories) for kind in ['bce','soft_iou','lovasz']}
            if not np.array_equal(p['bce',0],data['neural'][:,2]):raise ValueError('Original assessment BCE does not reproduce')
            metrics={}
            for policy,name in policies.items():
                choice=by_name[name]
                mixed=data['base'] if choice['variant']=='tree' else (1-choice['alpha'])*data['base']+choice['alpha']*source_score(choice['variant'],p)
                metrics[policy]=metric(mixed,data,choice['setting'])
            assessment['assessment'].append({'seed':record['seed'],'models':metrics})
        results.append({'source_study':study,'source_reports':[old_name,new_name],'dates':old['dates'],
            'calibration_candidates':candidates,'cal_sensitivity':sensitivity,'policies':policies,
            'selection_seal_sha256':sha256_file(seal_path),'exported_models':exported,'assessment':assessment['assessment'],
            'comparisons':{'regularized_vs_previous':comparison(assessment,'previous_seed_mixture','regularized'),
                'argmax_vs_previous':comparison(assessment,'previous_seed_mixture','argmax'),
                'mean_bce_vs_previous':comparison(assessment,'previous_seed_mixture','mean_bce'),
                'regularized_vs_mean_bce':comparison(assessment,'mean_bce','regularized')},'formal_Queue':None})
        print(study,'policies',policies,'regularized gain',results[-1]['comparisons']['regularized_vs_previous']['paired_macro_gain'],flush=True)
    result={'registered':registered,'source_snapshot':snapshot,'studies':results,'seconds':time.perf_counter()-start,
        'source':evidence(),'formal_Queue':None,'formal_Total':None,'uploaded':False,
        'limitations':'No refit. Two optimizer seeds share events,cal checkpoint labels and previously inspected TRAIN assessment dates. Bootstrap is a model-selection sensitivity heuristic,not independent coverage. No new corridor or latent truth evidence; no promotion or formal-score claim.'}
    text=json.dumps(result,indent=2,allow_nan=False);(args.output/'experiment.json').write_text(text,encoding='utf-8')
    report_path.write_text(text,encoding='utf-8');print('complete seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
