"""Trusted frozen role construction shared by post-seal State evaluations."""
import json
from pathlib import Path
from trafficflow.state import StateModel
from trafficflow.runtime import sha256_file


def role_paths(study_path):
    study=json.loads(study_path.read_text());config=study['registered']['protocol'];folder=Path('artifacts/research')/study_path.stem
    original=json.loads(Path(config['source_study']).read_text());old=original['registered']['protocol']
    if sha256_file(config['source_study'])!=config['source_study_sha256']:raise ValueError('Original study changed')
    replay=json.loads(Path('reports/research/A74b_recurrent_I210_W_independent_replay.json').read_text())
    if config['panel']!='D7_I210_W' or replay['registered']['protocol']['source_study_sha256']!=sha256_file(config['source_study']):
        raise ValueError('Wrong original role export for this corridor')
    paths={'lgb':Path(replay['exports']['lgb']['path']),'incumbent':Path(old['incumbent_file']),
        'prior_GRU':Path(replay['exports']['tcn_gru_seed_mean']['path'])}
    for role,key in [('lgb','lgb'),('prior_GRU','tcn_gru_seed_mean')]:
        if sha256_file(paths[role])!=replay['exports'][key]['sha256']:raise ValueError('Reference export changed')
    if sha256_file(paths['incumbent'])!=old['incumbent_sha256']:raise ValueError('Incumbent changed')
    for role,row in study['reused_controls'].items():
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Reused control changed')
        paths[role]=Path(row['path'])
    for role,row in study['training'].items():
        path=folder/('State_'+role+'.pkl')
        if sha256_file(path)!=row['model_sha256']:raise ValueError('New trained role changed')
        paths[role]=path
    for variant in ['point','queue']:paths[variant+'_seed_mean']=folder/('State_'+variant+'_seed_mean.pkl')
    return paths


def load_roles(study_path,primary_export=None):
    paths=role_paths(study_path);models={n:StateModel.load(p) for n,p in paths.items()}
    if primary_export is not None:
        study=json.loads(study_path.read_text());models[study['selection']['primary']].save(primary_export)
        paths['primary_export']=primary_export;models['primary_export']=StateModel.load(primary_export)
    return paths,models
