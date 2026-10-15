"""Reproduce the TF32 batch-size effect without changing models or data."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from trafficflow.data import Release
from trafficflow.state_control_model import StateControlModel
from trafficflow.state_control_data import causal_state_input
from trafficflow.joint_graph_queue import JointGraphQueue
from trafficflow.runtime import sha256_file,capture_source_snapshot
from verify_joint_dynamics_data import reference_day,reference_state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite precision diagnostic')
    source_path=Path('reports/research/C95_state_specialization_controls.json');source=json.loads(source_path.read_text());case=source['cases'][0];role='joint_state_event_cp';export=case['exports'][role];release=Release('data/kaggle_public');net=release.network(case['panel']);links=np.load(case['dataset_description']['fit_event_record']['path'])['links'].tolist();record=next(r for r in case['held_records'] if r['phase']=='decision');p=reference_day(record['raw']['path'],record['masked']['path'],links,net);x=np.stack([reference_state(p,at)['state_x'] for at in range(12,60)]);origin=pd.Timestamp(str(p['timestamps'][12]));f=release.read(record['masked']['path']);t=pd.to_datetime(f.timestamp,utc=True);history=f.loc[(t>=origin-pd.Timedelta(minutes=60))&(t<=origin)];adapted=causal_state_input(history,links,net,origin)
    if not np.array_equal(adapted,x[0]):raise ValueError('Precision diagnostic inputs differ')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=StateControlModel.load(export['path']);graph=[torch.tensor(model.geometry[k],device=device) for k in ['incoming','outgoing','geometry']];networks=[]
    for state in model.states:
        network=JointGraphQueue().to(device);network.load_state_dict(state);network.eval();networks.append(network)
    outputs={};before=torch.backends.cudnn.allow_tf32
    for allow in [True,False]:
        for batch_name,values in [('batch',x),('single',x[:1]),('repeat',np.repeat(x[:1],48,axis=0))]:
            with torch.no_grad(),torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,benchmark=False,deterministic=torch.backends.cudnn.deterministic,allow_tf32=allow):
                tensor=torch.tensor(values.transpose(0,3,1,2).copy(),device=device);outputs[('TF32' if allow else 'FP32')+'_'+batch_name]=np.mean([m.analyze(tensor,*graph).cpu().numpy() for m in networks],axis=0)
    if torch.backends.cudnn.allow_tf32!=before:raise ValueError('Precision context leaked backend flag')
    path=folder/'same_input_precision_arrays.npz';np.savez_compressed(path,input=x,**outputs)
    result={'source_study_sha256':sha256_file(source_path),'source_snapshot':snapshot,'source_export':export,'source_date':record['date'],'role':role,'exact_input_equality':True,'torch_version':torch.__version__,'CUDA_available':torch.cuda.is_available(),
        'TF32_single_batch_normalized_max_by_field':np.abs(outputs['TF32_single'][0]-outputs['TF32_batch'][0]).max(axis=1).tolist(),'TF32_repeat_actual_batch_max':float(np.abs(outputs['TF32_repeat'][0]-outputs['TF32_batch'][0]).max()),
        'FP32_single_batch_normalized_max_by_field':np.abs(outputs['FP32_single'][0]-outputs['FP32_batch'][0]).max(axis=1).tolist(),'TF32_to_FP32_batch_max_by_field':np.abs(outputs['TF32_batch']-outputs['FP32_batch']).max(axis=(0,2)).tolist(),
        'arrays':{'path':path.as_posix(),'sha256':sha256_file(path)},'backend_flag_restored':True,'scope':'Onepreselected failinginterfacewindowand48contextbatch,samemodelweights/input. Repeat48firstrowchecksotherbatchvaluesdo notchangeoutput;fullFP32reducesbatch-size numericaldifference. Not complete-global-errorbound,modelgain,criteriontuning or formal score.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'diagnostic.json').write_text(raw,encoding='utf-8');print('State_precision_diagnostic',result['TF32_single_batch_normalized_max_by_field'],result['FP32_single_batch_normalized_max_by_field'],flush=True)


if __name__=='__main__':main()
