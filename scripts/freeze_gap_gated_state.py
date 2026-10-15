"""Freeze an input-only gap expert and unseen-mask research protocol."""
import argparse
import json
from pathlib import Path
import pandas as pd
from trafficflow.state import StateModel
from trafficflow.gap_gated_state import GapGatedState
from trafficflow.outage_shift import SHIFT_CASES
from trafficflow.research_eval import register_protocol
from trafficflow.runtime import capture_source_snapshot,sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite frozen expert policy')
    training=Path('reports/research/A87_I210_W_outage_augmentation.json');study=json.loads(training.read_text())
    models=Path('artifacts/research')/training.stem;replay_path=Path('reports/research/A90_I210_W_augmentation_training_independent_replay.json')
    replay=json.loads(replay_path.read_text());paths={n:Path(replay['exports'][role]['path']) for n,role in [('control','control_seed_mean'),('expert','augmented_seed_mean'),('lgb','lgb')]}
    for n,role in [('control','control_seed_mean'),('expert','augmented_seed_mean'),('lgb','lgb')]:
        if sha256_file(paths[n])!=replay['exports'][role]['sha256']:raise ValueError('Verified model changed')
    if sha256_file(models/'selection_sealed.json')!=study['selection_seal_sha256']:raise ValueError('Original choice changed')
    anchors=pd.date_range('2030-12-11','2030-12-24').strftime('%Y-%m-%d').tolist()
    ordinary=pd.date_range('2030-12-11','2030-12-25').strftime('%Y-%m-%d').tolist()
    original=json.loads(Path(study['registered']['protocol']['source_study']).read_text())
    used={d for ds in original['prepared']['dates'].values() for d in ds}|set(study['registered']['protocol']['future_evaluation_dates'])
    if set(ordinary)&used:raise ValueError('New calendar overlaps these weights/calibration/previous A88')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Input_only180minute_gap_expert_fixed_before_shifted_research_evaluation','training_sha256':sha256_file(training),
        'training_replay_sha256':sha256_file(replay_path),'equal_blend_diagnostic_sha256':sha256_file('reports/research/A92b_equal_blend_direct_diagnostic.json'),
        'panel':'D7_I210_W','threshold_steps':36,'threshold_minutes':180,
        'gate':'At each target,use expert iff either speed/flow missing run length>=36on full released offline grid;no teacher/score mask/regime/outage metadata',
        'reference_and_expert':'Both original cal-selected two-seed mixtures frozen;no new weight fitting or newalpha selection',
        'anchor_dates':anchors,'ordinary_dates':ordinary,'mask_seeds':[20261121,20261123,20261127],'cases':SHIFT_CASES,
        'cross_midnight':'One hour before midnight plus three hours next day;same sensors;each day reconstructed separately without borrowing adjacent-day measurements',
        'target_populations':'Ordinary released Task1 targets vs all raw-train eligible rows inside own outage;never pooled together',
        'inference_exports':{n:{'path':p.as_posix(),'sha256':sha256_file(p)} for n,p in paths.items()},
        'selection':'Threshold,dates,cases,seeds fixed before these new outcomes;report every condition with paired anchor-date diagnostics',
        'scope':'New public research calendar for these policies;other past studies may have viewed dates,not final/private',
        'geometry_sha256':sha256_file('src/trafficflow/outage_shift.py'),'gate_source_sha256':sha256_file('src/trafficflow/gap_gated_state.py'),
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'uploaded':False})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip')
    gate=GapGatedState(StateModel.load(paths['control']),StateModel.load(paths['expert']),36)
    path=folder/'State_gap_gated.pkl';gate.save(path)
    result={'registered':registered,'source_snapshot':snapshot,'export':{'path':path.as_posix(),'sha256':sha256_file(path)},
        'new_model_parameters_fitted':False,'new_evaluation_labels_read':False,'formal_State':None,'formal_Physics':None,'formal_Total':None}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'policy.json').write_text(text,encoding='utf-8')
    print('gap_policy_frozen',path,'new anchors',len(anchors),'new outcome labels read:False',flush=True)


if __name__=='__main__':main()
