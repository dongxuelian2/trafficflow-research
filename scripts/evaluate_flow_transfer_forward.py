"""Fixed flow-transfer models on their preregistered public date block."""
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
from flow_transfer_common import context,compose,PublicDays,load_exports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite flow forward')
    study,cfg,source,views,original=context(args.study)
    if sha256_file(source/'selection_sealed.json')!=study['selection_seal_sha256']:raise ValueError('Transfer choice changed')
    if set(cfg['future_dates'])&{d for values in original['prepared']['dates'].values() for d in values}:raise ValueError('Forward overlaps model calendar')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Frozen_flow_transfer_new_public_calendar','source_study':args.study.as_posix(),'source_study_sha256':sha256_file(args.study),
        'dates':cfg['future_dates'],'selection':'No channel weight,checkpoint,cal gate or date reselection',
        'scope':'Public research,not final/private. Four primitive roles and fixed transfer directly inferred every date;remaining roles derived and sampled separately.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    models=load_exports(study);anchor=models['nn_primary'];evaluator=PublicDays(cfg['panel'],anchor.profile,anchor.feature_version)
    predictions={r:[] for r in models};days=[];files=[];maximum=0.;calls=0
    for date in cfg['future_dates']:
        day=evaluator.load(date);days.append(day);values={}
        for role in ['nn_primary','old_lgb','flow_seed20261013','flow_seed20261029']:
            p=models[role].predict(day['features'],day['template'])[VALUES].to_numpy(float);p[:,1]/=day['target_link_lanes'];values[role]=p;calls+=1
        values=compose(values,study['selection']['primary'])
        actual=models['transfer_fixed'].predict(day['features'],day['template'])[VALUES].to_numpy(float);actual[:,1]/=day['target_link_lanes'];calls+=1
        error=float(np.abs(actual-values['transfer_fixed']).max());maximum=max(maximum,error)
        if error>1e-8:raise ValueError('Deployed transfer differs from fixed channel formula')
        for role,p in values.items():predictions[role].append(p)
        path=folder/(date+'.npz');np.savez_compressed(path,truth_link_lane_units=day['station_truth'],target_grid_indices=day['target_indices'],
            target_link_lanes=day['target_link_lanes'],station_lanes=day['station_lanes'],**values,**{'mask_'+k:v for k,v in day['diagnostic_masks'].items()})
        files.append({'date':date,'path':path.as_posix(),'sha256':sha256_file(path)});print('flow_forward',date,'complete',flush=True)
    metrics={r:summarize(days,p,evaluator.factor) for r,p in predictions.items()}
    comparisons={r:{'pooled_State_gain':m['S_state']-metrics['nn_primary']['S_state'],
        'paired_day_diagnostic':paired_block_interval([d['daily_State'] for d in metrics['nn_primary']['days']],[d['daily_State'] for d in m['days']])} for r,m in metrics.items()}
    result={'registered':registered,'source_snapshot':snapshot,'metrics':metrics,'comparisons':comparisons,'exports':study['exports'],
        'prediction_files':files,'source_files_sha256':evaluator.hashes,'selection_changed':False,'n_direct_predictor_calls':calls,
        'maximum_deployed_transfer_error_lane_units':maximum,'seconds':time.perf_counter()-start,'source':evidence(gpu_used=torch.cuda.is_available()),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'limits':'Single winter transfer calendar;public research. Some ensemble outputs derived,not all role/day direct calls. Legacy increment diagnostic is not formal Physics.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'experiment.json').write_text(text,encoding='utf-8')
    print('flow_forward_COMPLETE','NN',metrics['nn_primary']['S_state'],'fixed',metrics['transfer_fixed']['S_state'],'calls',calls,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
