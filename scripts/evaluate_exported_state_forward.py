"""Directly deploy every frozen State export on its registered future dates."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trafficflow.data import VALUES
from trafficflow.runtime import sha256_file,capture_source_snapshot,evidence
from trafficflow.research_eval import register_protocol,paired_block_interval
from recurrent_state_experiment import summarize
from flow_transfer_common import PublicDays,load_exports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite model forward')
    study=json.loads(args.study.read_text());cfg=study['registered']['protocol'];source=Path('artifacts/research')/args.study.stem
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256']:raise ValueError('Frozen choice changed')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{'id':'All_exported_State_models_frozen_calendar',
        'source_study':args.study.as_posix(),'source_study_sha256':sha256_file(args.study),'dates':cfg['future_dates'],'reference_role':cfg.get('reference_role','nn_primary'),
        'scope':'All exports directly called every date;no checkpoint,alpha,primary or calendar reselection. Public research,not final/private.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    models=load_exports(study);anchor=models['nn_primary'];evaluator=PublicDays(cfg['panel'],anchor.profile,anchor.feature_version)
    days=[];predictions={r:[] for r in models};outputs=[]
    for date in cfg['future_dates']:
        day=evaluator.load(date);days.append(day);values={}
        for role,model in models.items():
            p=model.predict(day['features'],day['template'])[VALUES].to_numpy(float);p[:,1]/=day['target_link_lanes'];values[role]=p;predictions[role].append(p)
        path=folder/(date+'.npz');np.savez_compressed(path,truth_link_lane_units=day['station_truth'],target_grid_indices=day['target_indices'],
            target_link_lanes=day['target_link_lanes'],station_lanes=day['station_lanes'],**values,**{'mask_'+k:v for k,v in day['diagnostic_masks'].items()})
        outputs.append({'date':date,'path':path.as_posix(),'sha256':sha256_file(path)});print('all_exports_forward',date,'complete',flush=True)
    metrics={r:summarize(days,p,evaluator.factor) for r,p in predictions.items()};comparisons={};reference=cfg.get('reference_role','nn_primary')
    for role,m in metrics.items():comparisons[role]={'pooled_State_gain':m['S_state']-metrics[reference]['S_state'],
        'paired_day_diagnostic':paired_block_interval([d['daily_State'] for d in metrics[reference]['days']],[d['daily_State'] for d in m['days']])}
    result={'registered':registered,'source_snapshot':snapshot,'metrics':metrics,'comparisons':comparisons,'exports':study['exports'],'prediction_files':outputs,
        'source_files_sha256':evaluator.hashes,'selection_changed':False,'reference_role':reference,'n_direct_predictor_calls':len(models)*len(days),'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'limits':'Public observation targets and incomplete legacy dynamics,not latent truth or formal Physics/Total.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('all_exports_forward_COMPLETE',len(models)*len(days),'calls','seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
