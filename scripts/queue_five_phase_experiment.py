"""New-direction Queue training with distinct checkpoint/decision/selection dates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release,eligible
from trafficflow.experiments import partition_date,raw_for
from trafficflow.queue_events import sample_events
from trafficflow.queue import ProxyData,queue_features
from trafficflow.queue_calibration import RefinedQueueModel
from trafficflow.causal_queue import released_origin_rows
from trafficflow.graph_queue import CausalGraphQueue,queue_history_input,queue_event_loss
from trafficflow.queue_graph_geometry import travel_time_neighborhoods
from trafficflow.probability_calibration import weighted_bce_probability
from trafficflow.full_queue_calibration import tune_full_probabilities,full_queue_decisions
from trafficflow.full_domain_queue import FullDomainQueueModel
from trafficflow.ensemble_queue import EnsembleFullDomainQueueModel
from trafficflow.queue_phase_split import queue_phase_dates,validate_phase_origins
from trafficflow.queue_domain import score_full_forecast
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from compare_queue_events import comparison


def neural_score(model,arrays,graph,device):
    p=[];model.eval()
    with torch.no_grad():
        for x in arrays['x']:
            logits,_=model(torch.from_numpy(x.transpose(2,0,1)[None].copy()).to(device),*graph)
            p.append(weighted_bce_probability(logits.cpu().numpy()[0],4.))
    return np.stack(p)


def checkpoint_brier(p,arrays):
    domain=arrays['available'].copy();domain[:,:5]&=arrays['ongoing'][:,None,None]
    values=((p-arrays['labels'])**2*domain).sum(axis=(1,2))/domain.sum(axis=(1,2)).clip(min=1)
    return float(np.mean([values[arrays['ongoing']==value].mean() for value in [False,True]]))


def metrics(p,data,setting):
    pred=full_queue_decisions(p,data['window'],data['condition'],data['lead_minutes'],
        setting['thresholds'],setting['require_nonempty'],setting['timing'])
    return score_full_forecast(data['labels'].ravel(),pred,data['window'],data['condition'],data['available'].ravel())


class Data:
    def __init__(self,release,panel,dates,folder,balanced=False):
        self.release,self.panel,self.dates,self.folder=release,panel,dates,folder
        self.net,self.topology=release.network(panel),release.topology(panel)
        paths=release.files(panel,'train');self.paths={p:[f for f in paths if partition_date(release,f) in days] for p,days in dates.items()}
        self.links=None;self.origin_cache={};self.records=[];self.phase_reads=[]
        self.balanced=balanced

    def origin(self,origin):
        key=str(origin)
        if key not in self.origin_cache:self.origin_cache[key]=released_origin_rows(self.release,self.panel,'train',origin)
        return self.origin_cache[key]

    def load(self,part,seed):
        if part in ['decision','selection','assessment']:
            required={'decision':'checkpoint_sealed.json','selection':'decision_sealed.json','assessment':'selection_sealed.json'}[part]
            if not (self.folder/required).is_file():raise ValueError('Earlier model choice must be sealed before '+part)
        frames=list(raw_for(self.release,self.panel,self.paths[part]))
        quota,spacing=(500,30) if part=='fit' else (30,360)
        filtered,meta=sample_events(frames,self.net,self.topology,self.panel,seed,per_condition=quota,
            spacing_minutes=spacing,origin_reader=self.origin,include_occupancy=True,onset_last_step=True,
            balanced_conditions=self.balanced)
        if set(filtered.condition)!=set(['queue_onset','queue_ongoing']):raise ValueError('Both proxy conditions needed in '+part)
        ordered=sorted(meta['windows'],key=lambda row:row['origin']);origins=[r['origin'] for r in ordered]
        validate_phase_origins(part,origins,self.dates)
        by_date={str(frame.timestamp.iloc[0])[:10]:frame for frame in frames}
        xs=[];labels=[];speed=[];available=[];conditions=[];windows=[];leads=[];plain=[];rich=[]
        for row in ordered:
            origin=pd.Timestamp(row['origin']);raw=by_date[origin.date().isoformat()];times=sorted(raw.timestamp.unique());at=row['at']
            if pd.Timestamp(times[at])!=origin:raise ValueError('Sample origin shifted')
            links=raw.link_id.drop_duplicates().tolist()
            if self.links is None:self.links=links
            if self.links!=links:raise ValueError('Queue node order changed')
            history=pd.concat([raw[raw.timestamp.isin(times[at-12:at])],self.origin(origin)],ignore_index=True)
            target=pd.MultiIndex.from_product([times[at+1:at+7],links],names=['timestamp','link_id']).to_frame(index=False)
            x_plain=queue_features(history,target,self.net,self.topology,origin,allow_origin=True)
            x_rich=queue_features(history,target,self.net,self.topology,origin,allow_origin=True,include_occupancy=True)
            x_rich['condition_ongoing']=float(row['condition']=='queue_ongoing')
            plain.append(x_plain);rich.append(x_rich)
            xs.append(queue_history_input(history,links,self.net,origin,row['condition']))
            # Future observations are supervised targets/evaluator arrays only.
            future=raw[raw.timestamp.isin(times[at+1:at+7])].copy()
            future.loc[~eligible(future),'speed_kmh']=np.nan
            v=future.pivot_table(index='timestamp',columns='link_id',values='speed_kmh').reindex(index=times[at+1:at+7],columns=links).to_numpy(float)
            labels.append((v<=self.net.v_cut.reindex(links).to_numpy(float)).astype(np.float32))
            available.append(np.isfinite(v));speed.append(np.nan_to_num(v/25.).astype(np.float32))
            width=len(target);conditions.extend([row['condition']]*width)
            windows.extend([f'{self.panel}:{times[at]}:{seed}']*width);leads.extend(x_plain.lead_minutes)
        arrays={'x':np.stack(xs),'labels':np.stack(labels),'truth_speed':np.stack(speed),'available':np.stack(available),
            'ongoing':np.array([r['condition']=='queue_ongoing' for r in ordered]),'origins':np.array(origins),
            'window':np.array(windows),'condition':np.array(conditions),'lead_minutes':np.array(leads),'links':np.array(self.links)}
        if not np.array_equal(arrays['labels'].ravel()[arrays['available'].ravel()],filtered.y):
            raise ValueError('Sequence and tree label order disagree')
        token=part+'_'+str(seed);npz=self.folder/(token+'.npz');np.savez_compressed(npz,**arrays)
        features={'plain':pd.concat(plain,ignore_index=True),'rich':pd.concat(rich,ignore_index=True)}
        paths=[]
        for name,x in features.items():
            p=self.folder/(token+'_'+name+'.parquet');x.to_parquet(p,index=False);paths.append({'path':p.as_posix(),'sha256':sha256_file(p)})
        self.records.append({'phase':part,'seed':seed,'path':npz.as_posix(),'sha256':sha256_file(npz),
            'feature_files':paths,'sampler':meta,'n_windows':len(origins)})
        self.phase_reads.append({'phase':part,'seed':seed,'n_label_days':len(self.dates[part]),
            'prior_seal_sha256':sha256_file(self.folder/required) if part in ['decision','selection','assessment'] else None})
        print(part,seed,'windows',len(origins),'conditions',meta['selected_counts'],flush=True)
        return arrays,features


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--panel',required=True)
    parser.add_argument('--fit-end',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--balanced-sampling',action='store_true');args=parser.parse_args()
    report_path=Path('reports/research')/(args.output.name+'.json')
    if args.output.exists() or report_path.exists():raise FileExistsError('Never overwrite five-phase evidence')
    release=Release('data/kaggle_public');dates=queue_phase_dates([partition_date(release,p) for p in release.files(args.panel,'train')],
        '2030-07-16',args.fit_end);seeds=[20261013,20261029,20261107];event_seeds=[2909,3253,4001]
    registered=register_protocol(report_path.with_suffix('.registered.json'),{
        'id':'Queue_new_direction_five_disjoint_time_phases','panel':args.panel,'dates':dates,'fit_seeds':seeds,
        'fit_event_seed':20261013,'checkpoint_event_seed':4129,'decision_event_seed':4133,'selection_assessment_event_seeds':event_seeds,
        'balanced_condition_sampling':args.balanced_sampling,
        'tree_arms':'Plain causal600/63 versus same+occupancy/provided condition; both fit only registered fit days',
        'neural':'30min graph hidden32,three optimizer seeds,20x80,batch4,AdamW .001/wd1e-4/clip1',
        'checkpoint':'Minimum checkpoint-phase event/condition Brier every5 epochs; no decision/selection labels seen',
        'decision':'30-day full-domain threshold calibration,choose incumbent tree there;fixed alpha.25 seed arms/three-seed mean',
        'selection':'15-day challenger-vs-incumbent paired origin/days;replace only if gain>=.01 and lower95CI>0. No selection retuning.',
        'assessment':'30 days,load only after selection seal;all policies retain original choices',
        'label_contract':'Eligible noisy public TRAIN observation proxy;existing scored-positive sampler bias not solved',
        'final':'Public development evidence,not final simulation/hidden truth; final simulator seeds unopened',
        'formal_Queue':None,'formal_Total':None,'uploaded':False})
    args.output.mkdir(parents=True);snapshot=capture_source_snapshot(args.output/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    data=Data(release,args.panel,dates,args.output,args.balanced_sampling);fit,fit_features=data.load('fit',20261013)
    trees={}
    for arm,x in fit_features.items():
        mask=fit['available'].ravel();view=ProxyData(x.loc[mask].reset_index(drop=True),fit['labels'].ravel()[mask],
            fit['window'][mask],fit['condition'][mask],np.array([str(w).split(':',1)[1][:10] for w in fit['window'][mask]]))
        tree=RefinedQueueModel(iterations=600,num_leaves=63,seed=20261013).fit(view)
        tree.uses_origin_timestamp=True;tree.latest_valid_queue_bookkeeping=True
        tree.uses_occupancy_features=arm=='rich';tree.uses_published_condition=arm=='rich'
        tree.save(args.output/('tree_'+arm+'.pkl'));trees[arm]=tree
    checkpoint,cp_features=data.load('checkpoint',4129)
    net=data.net;lane=net.lanes.reindex(data.links).to_numpy(float)
    geometry={'incoming':travel_time_neighborhoods(data.links,net,data.topology,30.)[0],
              'outgoing':travel_time_neighborhoods(data.links,net,data.topology,30.)[1],
              'geometry':np.stack([net.free_speed_kmh.reindex(data.links),net.capacity_vph.reindex(data.links)/lane,
                  net.k_jam.reindex(data.links)/lane,net.v_cut.reindex(data.links)]).astype(np.float32)}
    np.savez(args.output/'geometry.npz',**geometry);graph=[torch.tensor(geometry[k],device=device) for k in ['incoming','outgoing','geometry']]
    groups={c:np.flatnonzero(fit['ongoing']==c) for c in [False,True]};training={};states={}
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed);model=CausalGraphQueue(hidden=32,use_graph=True).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        best=np.inf;history=[];digest=hashlib.sha256()
        for epoch in range(20):
            model.train();losses=[]
            for step in range(80):
                idx=np.array([rng.choice(groups[bool(rng.integers(2))]) for _ in range(4)],dtype=np.int64);digest.update(idx.tobytes())
                x=torch.from_numpy(fit['x'][idx].transpose(0,3,1,2).copy()).to(device)
                targets=[torch.from_numpy(fit[k][idx]).to(device) for k in ['labels','truth_speed','available','ongoing']]
                optimizer.zero_grad(set_to_none=True);logits,speed=model(x,*graph)
                loss=queue_event_loss(logits,speed,*targets)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
            if (epoch+1)%5==0:
                value=checkpoint_brier(neural_score(model,checkpoint,graph,device),checkpoint)
                history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),'checkpoint_Brier':value})
                if value<best:best=value;torch.save(model.state_dict(),args.output/('seed_'+str(seed)+'.pt'))
                print(args.output.name,seed,epoch+1,'checkpointBrier',value,flush=True)
        states[str(seed)]=torch.load(args.output/('seed_'+str(seed)+'.pt'),map_location='cpu',weights_only=True)
        training[str(seed)]={'history':history,'batch_sequence_sha256':digest.hexdigest(),
            'selected_checkpoint_sha256':sha256_file(args.output/('seed_'+str(seed)+'.pt'))}
        del model,optimizer
    (args.output/'checkpoint_sealed.json').write_text(json.dumps(training,indent=2,allow_nan=False),encoding='utf-8')
    decision,df=data.load('decision',4133)
    tree_decision={arm:tree.probability(df[arm]) for arm,tree in trees.items()};choices={}
    for arm,p in tree_decision.items():
        setting=tune_full_probabilities(p,decision['window'],decision['condition'],decision['lead_minutes'],decision['labels'].ravel(),decision['available'].ravel())['selected']
        choices['tree_'+arm]={'baseline_arm':arm,'alpha':0.,'members':[],'setting':setting,'metrics':metrics(p,decision,setting)}
    incumbent=max(choices,key=lambda name:(choices[name]['metrics']['proxy_IoU'],name=='tree_plain'))
    baseline_arm=choices[incumbent]['baseline_arm'];base=tree_decision[baseline_arm]
    def neural_probabilities(arrays):
        result={}
        for seed,state in states.items():
            model=CausalGraphQueue(hidden=32,use_graph=True).to(device);model.load_state_dict(state)
            result[seed]=neural_score(model,arrays,graph,device).ravel();del model
        return result
    dp=neural_probabilities(decision)
    for name,members in [(f'graph_{s}',[str(s)]) for s in seeds]+[('graph_mean',[str(s) for s in seeds])]:
        p=.75*base+.25*np.mean([dp[s] for s in members],axis=0)
        setting=tune_full_probabilities(p,decision['window'],decision['condition'],decision['lead_minutes'],decision['labels'].ravel(),decision['available'].ravel())['selected']
        choices[name]={'baseline_arm':baseline_arm,'alpha':.25,'members':members,'setting':setting,'metrics':metrics(p,decision,setting)}
    decision_selected=max(choices,key=lambda name:(choices[name]['metrics']['proxy_IoU'],name==incumbent,-len(choices[name]['members'])))
    (args.output/'decision_sealed.json').write_text(json.dumps({'incumbent':incumbent,'challenger':decision_selected,'choices':choices},indent=2,allow_nan=False),encoding='utf-8')
    def evaluate(part):
        result={'dates':{'assessment':dates[part]},'assessment':[]}
        for seed in event_seeds:
            arrays,features=data.load(part,seed);p_nn=neural_probabilities(arrays);values={}
            p_tree={arm:tree.probability(features[arm]) for arm,tree in trees.items()}
            for name,choice in choices.items():
                p=p_tree[choice['baseline_arm']]
                if choice['members']:p=(1-choice['alpha'])*p+choice['alpha']*np.mean([p_nn[s] for s in choice['members']],axis=0)
                values[name]=metrics(p,arrays,choice['setting'])
            result['assessment'].append({'seed':seed,'models':values})
        return result
    selection=evaluate('selection');gate=comparison(selection,incumbent,decision_selected)
    selected=decision_selected if gate['paired_macro_gain']>=.01 and gate['ci95'] is not None and gate['ci95'][0]>0 else incumbent
    (args.output/'selection_sealed.json').write_text(json.dumps({'selected':selected,'incumbent':incumbent,
        'decision_selected':decision_selected,'gate':gate,'rule':'gain>=.01 and lowerCI>0'},indent=2,allow_nan=False),encoding='utf-8')
    exported={}
    for role,name in [('incumbent',incumbent),('decision_selected',decision_selected),('selected',selected),('graph_mean','graph_mean')]:
        choice=choices[name];tree=trees[choice['baseline_arm']]
        model=FullDomainQueueModel(tree,data.links,choice['setting']) if not choice['members'] else EnsembleFullDomainQueueModel(
            tree,data.links,choice['setting'],choice['alpha'],[states[s] for s in choice['members']],geometry)
        path=args.output/(role+'.pkl');model.save(path);exported[role]={'path':path.as_posix(),'sha256':sha256_file(path),'candidate':name}
    assessment=evaluate('assessment')
    result={'registered':registered,'source_snapshot':snapshot,'dates':dates,'training':training,'decision_choices':choices,
        'incumbent':incumbent,'decision_selected':decision_selected,'selected':selected,'selection_gate':gate,
        'selection':selection['assessment'],'assessment':assessment['assessment'],'exported_models':exported,
        'comparisons':{'selected_vs_incumbent':comparison(assessment,incumbent,selected),
            'decision_selected_vs_incumbent':comparison(assessment,incumbent,decision_selected),
            'graph_mean_vs_incumbent':comparison(assessment,incumbent,'graph_mean')},
        'cache_records':data.records,'phase_read_records':data.phase_reads,
        'seal_sha256':{p:sha256_file(args.output/(p+'_sealed.json')) for p in ['checkpoint','decision','selection']},
        'seconds':time.perf_counter()-start,'source':evidence(gpu_used=device.type=='cuda'),
        'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
        'formal_Queue':None,'formal_Total':None,'uploaded':False,
        'limitations':'New directions and disjoint mechanism phases,but public noisy scored-positive proxy remains biased. Three optimizer seeds share events. Decision-phase tree choice and thresholds are not an independent confidence guarantee;gate uses a later15-day block. Public development dates,not final/hidden labels or formal score.'}
    text=json.dumps(result,indent=2,allow_nan=False);(args.output/'experiment.json').write_text(text,encoding='utf-8');report_path.write_text(text,encoding='utf-8')
    print('complete',args.output.name,'selected',selected,result['comparisons']['selected_vs_incumbent'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
