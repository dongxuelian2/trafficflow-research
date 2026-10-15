"""Resolve frozen tabular controls and NN anchor without fitting models."""
import json
from pathlib import Path
from trafficflow.state import StateModel
from trafficflow.bounded_state import BoundedState
from trafficflow.runtime import sha256_file


def context(study_path):
    study=json.loads(study_path.read_text());cfg=study['registered']['protocol'];folder=Path('artifacts/research')/study_path.stem
    views_path=Path('artifacts/research/A96_I210_W_native_remask_fit_views/cache.json')
    if sha256_file(views_path)!=cfg['view_report_sha256']:raise ValueError('Original native view report changed')
    views=json.loads(views_path.read_text());original_path=Path(views['registered']['protocol']['source_study']);original=json.loads(original_path.read_text())
    if sha256_file(original_path)!=cfg['source_study_sha256']:raise ValueError('Original study changed')
    if sha256_file(folder/'selection_sealed.json')!=study['selection_seal_sha256']:raise ValueError('Tabular selection changed')
    return study,cfg,folder,views,original


def load_tree_roles(study_path,export_folder):
    study,cfg,folder,views,original=context(study_path);export_folder.mkdir(parents=True,exist_ok=True)
    nn=cfg['NN_primary_export']
    if sha256_file(nn['path'])!=nn['sha256']:raise ValueError('NN anchor changed')
    paths={'nn_primary':Path(nn['path'])};models={'nn_primary':StateModel.load(Path(nn['path']))}
    old_cfg=original['registered']['protocol'];baseline_path=Path(old_cfg['baseline_file'])
    if sha256_file(baseline_path)!=old_cfg['baseline_sha256']:raise ValueError('Old tree changed')
    models['old_lgb']=BoundedState(StateModel.load(baseline_path));paths['old_lgb']=export_folder/'State_old_lgb.pkl';models['old_lgb'].save(paths['old_lgb'])
    for role,row in study['training'].items():
        if sha256_file(row['model_path'])!=row['model_sha256']:raise ValueError('Fitted tree changed')
        paths[role]=Path(row['model_path']);models[role]=StateModel.load(paths[role])
    for variant in ['point','queue']:
        paths[variant+'_seed_mean']=folder/('State_'+variant+'_seed_mean.pkl');models[variant+'_seed_mean']=StateModel.load(paths[variant+'_seed_mean'])
    primary=study['primary_export']
    if sha256_file(primary['path'])!=primary['sha256']:raise ValueError('Primary export changed')
    paths['primary_export']=Path(primary['path']);models['primary_export']=StateModel.load(paths['primary_export'])
    return paths,models
