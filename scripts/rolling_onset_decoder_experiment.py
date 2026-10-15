"""Predeclared rolling refits of the strong Queue recipe and onset decoders."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from lightgbm import LGBMClassifier
from trafficflow.data import Release
from trafficflow.experiments import partition_date
from trafficflow.queue import ProxyData
from trafficflow.queue_calibration import RefinedQueueModel
from trafficflow.full_domain_queue import FullDomainQueueModel
from trafficflow.full_queue_calibration import tune_full_probabilities,full_queue_decisions
from trafficflow.onset_rank_queue import OnsetTopKQueueModel,onset_topk_decisions
from trafficflow.onset_cardinality_queue import OnsetSetDecoderModel,onset_threshold_decisions
from trafficflow.queue_phase_split import queue_phase_dates
from trafficflow.queue_domain import score_full_forecast
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from queue_five_phase_experiment import Data
from compare_queue_events import comparison


JOBS=[('I5S_September','D12_I5_S','2030-09-15'),('I5S_October','D12_I5_S','2030-10-31'),
      ('I10W_September','D7_I10_W','2030-09-15'),('I10W_October','D7_I10_W','2030-10-31')]
SEEDS=[20261013,20261029]
EVENT_SEEDS=[2909,3253,4001]
KGRID=[1,2,4,8,16]


def write_once(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def file_record(path):
    return {'path':Path(path).as_posix(),'sha256':sha256_file(path)}


def metric(a,p):
    return score_full_forecast(a['labels'].ravel(),p,a['window'],a['condition'],a['available'].ravel())


def predictions(a,p,score,setting,threshold,k):
    base=full_queue_decisions(p,a['window'],a['condition'],a['lead_minutes'],setting['thresholds'],setting['require_nonempty'],setting['timing'])
    use=a['condition']=='queue_onset';links=np.tile(a['links'],len(a['ongoing'])*6);meta=(a['window'][use],a['lead_minutes'][use],links[use])
    fixed=base.copy();fixed[use]=onset_topk_decisions(score[use],*meta,k)
    variable=base.copy();variable[use]=onset_threshold_decisions(score[use],*meta,threshold['threshold'],threshold['force'])
    return {'incumbent':base,'fixed_k':fixed,'threshold':variable}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report.exists():raise FileExistsError('Never overwrite rolling Queue evidence')
    release=Release('data/kaggle_public');jobs=[]
    for name,panel,end in JOBS:
        dates=queue_phase_dates([partition_date(release,p) for p in release.files(panel,'train')],'2030-07-16',end)
        jobs.append({'name':name,'panel':panel,'dates':dates})
    registered=register_protocol(report.with_suffix('.registered.json'),{'id':'Rolling_onset_threshold_matched_refits','jobs':jobs,
        'previous_goal_turn':'progress:e9b268e froze C68-C70 with independent decoder audits;formal goal unmet',
        'fit':'Same original balanced-v2 sampler:500/condition,30min spacing,eventseed20261013. Fit only registered dates;checkpoint15days unused buffer.',
        'baseline':'Refit existing strong recipe:plain/rich causal tree600/63 seed20261013;all eligible six-step rows. Decision chooses stronger calibrated arm,plain on tie.',
        'onset':'For BOTH feature arms before decision:two classifier seeds20261013/20261029,600/63,lr.035,minchild60,lambda8,colsample.9,unit weights;only fit onset eligible T+30 rows. No ranker/count/neural refit.',
        'decision':'30days,eventseed4133. Same chosen baseline feature arm for onset;equal two-seed mean. Threshold31 values.05..8 x forceFalse/True,prefer noforce/closer.5. Fixedk1/2/4/8/16 is matched secondary only.',
        'primary_choice':'Decision chooses only incumbent or threshold,incumbent on tie. Later15day selection gate gain>=.01,7dayblockCI lower>0;otherwise incumbent. No assessment retuning.',
        'selection_assessment_event_seeds':EVENT_SEEDS,'evaluation_sampling':'Balanced30/condition,360min spacing. Deduplicate physical origins across event seeds for comparisons.',
        'assessment':'30days,read after selection seal. Public development dates have appeared in other studies;new fit calendars/weights,NOT blind/new public data/final/private.',
        'cross_case':'Descriptive evidence requires>=2case gain>=.01 and lowerCI>0,no case loss>.01. Cases share dates/fit inputs;not independent repetitions.',
        'platform':'Local models/Git only','final_simulator_seeds_generated':False,'formal_Queue':None,'formal_Total':None,'new_complete_submission':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter();cases=[]
    for job in jobs:
        folder=args.output/job['name'];folder.mkdir();data=Data(release,job['panel'],job['dates'],folder,balanced=True)
        fit,features=data.load('fit',SEEDS[0]);all_rows=fit['available'].ravel();onset_rows=all_rows&(fit['condition']=='queue_onset')&(fit['lead_minutes']==30)
        if set(fit['labels'].ravel()[onset_rows])!={0,1}:raise ValueError('Both onset labels required')
        rows_path=folder/'training_rows.npz';np.savez_compressed(rows_path,baseline_indices=np.flatnonzero(all_rows),onset_indices=np.flatnonzero(onset_rows),onset_labels=fit['labels'].ravel()[onset_rows].astype(int),onset_weights=np.ones(onset_rows.sum()))
        trees={};boosters={};training=[]
        for arm,x in features.items():
            view=ProxyData(x.loc[all_rows].reset_index(drop=True),fit['labels'].ravel()[all_rows],fit['window'][all_rows],fit['condition'][all_rows],np.array([str(w).split(':',1)[1][:10] for w in fit['window'][all_rows]]))
            tic=time.perf_counter();tree=RefinedQueueModel(iterations=600,num_leaves=63,seed=SEEDS[0]).fit(view)
            tree.uses_origin_timestamp=True;tree.latest_valid_queue_bookkeeping=True;tree.uses_occupancy_features=arm=='rich';tree.uses_published_condition=arm=='rich'
            path=folder/('baseline_'+arm+'.pkl');tree.save(path);text=folder/('baseline_'+arm+'.txt');tree.model.booster_.save_model(text);trees[arm]=tree
            training.append({'role':'baseline','arm':arm,'seed':SEEDS[0],'rows':int(all_rows.sum()),'trees':tree.model.booster_.num_trees(),'features':list(x.columns),'seconds':time.perf_counter()-tic,'model':file_record(path),'booster':file_record(text)})
            boosters[arm]=[]
            for seed in SEEDS:
                tic=time.perf_counter();model=LGBMClassifier(n_estimators=600,num_leaves=63,learning_rate=.035,min_child_samples=60,reg_lambda=8.,colsample_bytree=.9,n_jobs=4,random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True)
                model.fit(x.loc[onset_rows].reset_index(drop=True),fit['labels'].ravel()[onset_rows].astype(int),sample_weight=np.ones(onset_rows.sum()))
                path=folder/('onset_'+arm+'_'+str(seed)+'.txt');model.booster_.save_model(path);boosters[arm].append(model.booster_)
                training.append({'role':'onset','arm':arm,'seed':seed,'rows':int(onset_rows.sum()),'queries':len(np.unique(fit['window'][onset_rows])),'trees':model.booster_.num_trees(),'features':list(x.columns),'seconds':time.perf_counter()-tic,'model':file_record(path)})
            print('rolling_onset_fit',job['name'],arm,'onset_queries',len(np.unique(fit['window'][onset_rows])),flush=True)
        model_seal=folder/'checkpoint_sealed.json';write_once(model_seal,{'training':training,'training_rows':file_record(rows_path),'checkpoint_buffer_not_loaded':True})
        decision,dx=data.load('decision',4133);baseline_calibration={};probabilities={}
        for arm,tree in trees.items():
            probabilities[arm]=tree.probability(dx[arm]);baseline_calibration[arm]=tune_full_probabilities(probabilities[arm],decision['window'],decision['condition'],decision['lead_minutes'],decision['labels'].ravel(),decision['available'].ravel())
        arm=max(['plain','rich'],key=lambda a:baseline_calibration[a]['selected']['calibration_proxy_IoU']);setting=baseline_calibration[arm]['selected'];baseline=FullDomainQueueModel(trees[arm],data.links,setting,alpha=0.)
        score=np.mean([b.predict(dx[arm],num_threads=4) for b in boosters[arm]],axis=0);threshold_grid=[]
        for force in [False,True]:
            for t in np.linspace(.05,.8,31):
                s={'threshold':float(t),'force':force};p=predictions(decision,probabilities[arm],score,setting,s,1)['threshold'];threshold_grid.append({'setting':s,'metrics':metric(decision,p)})
        threshold_choice=max(threshold_grid,key=lambda r:(r['metrics']['proxy_IoU'],not r['setting']['force'],-abs(r['setting']['threshold']-.5)))
        k_grid=[]
        for k in KGRID:k_grid.append({'k':k,'metrics':metric(decision,predictions(decision,probabilities[arm],score,setting,threshold_choice['setting'],k)['fixed_k'])})
        k_choice=max(k_grid,key=lambda r:(r['metrics']['proxy_IoU'],-r['k']))
        pred=predictions(decision,probabilities[arm],score,setting,threshold_choice['setting'],k_choice['k']);decision_metrics={role:metric(decision,p) for role,p in pred.items()}
        challenger=max(['incumbent','threshold'],key=lambda role:decision_metrics[role]['proxy_IoU'])
        decision_seal=folder/'decision_sealed.json';write_once(decision_seal,{'feature_arm':arm,'baseline_calibration':baseline_calibration,'threshold_grid':threshold_grid,'threshold_choice':threshold_choice,'k_grid':k_grid,'k_choice':k_choice,'decision_metrics':decision_metrics,'decision_selected':challenger})
        prediction_records=[]
        def evaluate(phase,selected=None):
            result={'dates':{'assessment':job['dates'][phase]},'assessment':[]}
            for seed in EVENT_SEEDS:
                a,x=data.load(phase,seed);p=trees[arm].probability(x[arm]);s=np.mean([b.predict(x[arm],num_threads=4) for b in boosters[arm]],axis=0)
                pred=predictions(a,p,s,setting,threshold_choice['setting'],k_choice['k'])
                if selected is not None:pred['selected']=pred[selected]
                metrics={role:metric(a,values) for role,values in pred.items()};result['assessment'].append({'seed':seed,'models':metrics})
                path=folder/(phase+'_'+str(seed)+'_predictions.npz');np.savez_compressed(path,baseline_probability=p,onset_score=s,**pred);prediction_records.append({'phase':phase,'seed':seed,**file_record(path)})
            return result
        selection=evaluate('selection');gate=comparison(selection,'incumbent',challenger);selected=challenger if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else 'incumbent'
        selection_seal=folder/'selection_sealed.json';write_once(selection_seal,{'selected':selected,'decision_selected':challenger,'gate':gate})
        fixed=OnsetTopKQueueModel(baseline,[b.model_to_string() for b in boosters[arm]],list(dx[arm].columns),k_choice['k'],'classifier');threshold=OnsetSetDecoderModel(fixed,'threshold',setting=threshold_choice['setting'])
        portable={'incumbent':baseline,'fixed_k':fixed,'threshold':threshold};portable['selected']=portable[selected];exports={}
        for role,model in portable.items():
            path=folder/(role+'.pkl');model.save(path);exports[role]={**file_record(path),'candidate':selected if role=='selected' else role}
        assessment=evaluate('assessment',selected)
        case={**job,'training':training,'training_rows':file_record(rows_path),'feature_arm':arm,'baseline_setting':setting,'threshold_choice':threshold_choice,'k_choice':k_choice,'decision_metrics':decision_metrics,'decision_selected':challenger,'selection_gate':gate,'selected':selected,
            'selection':selection['assessment'],'assessment':assessment['assessment'],'comparisons':{role:comparison(assessment,'incumbent',role) for role in ['fixed_k','threshold','selected']},
            'cache_records':data.records,'phase_reads':data.phase_reads,'prediction_records':prediction_records,'exports':exports,
            'seals':{name:sha256_file(folder/(name+'_sealed.json')) for name in ['checkpoint','decision','selection']}}
        cases.append(case);write_once(folder/'case_completed.json',case);print('rolling_onset_case_COMPLETE',job['name'],'decision',challenger,'selected',selected,'gain',case['comparisons']['selected']['paired_macro_gain'],flush=True)
    def cross_gate(role):
        rows=[c['comparisons'][role] for c in cases];positive=sum(r['paired_macro_gain']>=.01 and r['ci95'] is not None and r['ci95'][0]>0 for r in rows);bounded=all(r['paired_macro_gain']>=-.01 for r in rows)
        return {'n_cases_gain_01_positive_CI':positive,'all_case_losses_bounded_01':bounded,'descriptive_gate_passed':positive>=2 and bounded}
    result={'registered':registered,'source_snapshot':snapshot,'cases':cases,'selected_evidence_gate':cross_gate('selected'),'threshold_secondary_gate':cross_gate('threshold'),
        'seconds':time.perf_counter()-start,'source':evidence(),'new_classifier_models':24,'formal_Queue':None,'formal_Total':None,'formal_model_promoted':False,'new_complete_submission':False,
        'limits':'New rolling refits on previously available/partly viewed public observation-proxy dates. No hidden latent labels or external platform actions. Fixedk is secondary;assessment does not select policies. Shared dates and overlapping fit inputs are not independent case replicates.'}
    write_once(report,result);write_once(args.output/'experiment.json',result);print('rolling_onset_COMPLETE','cases',len(cases),'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
