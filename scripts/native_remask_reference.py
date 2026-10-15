"""Audited original recurrent controls for each corridor,without refitting."""
import hashlib
import json
from pathlib import Path
import torch
from trafficflow.recurrent_state import make_network
from trafficflow.runtime import sha256_file


DEFAULT_REPLAYS={'D7_I210_W':'reports/research/A74b_recurrent_I210_W_independent_replay.json',
    'D12_I5_N':'reports/research/A75_recurrent_I5_N_independent_replay.json'}


def reference_controls(source_path,replay_path=None):
    original=json.loads(source_path.read_text());config=original['registered']['protocol']
    replay_path=Path(replay_path or DEFAULT_REPLAYS[config['panel']]);replay=json.loads(replay_path.read_text())
    if replay['registered']['protocol']['source_study_sha256']!=sha256_file(source_path) or replay['n_role_day_replays']!=280:
        raise ValueError('Original reference deployment is not complete/matched')
    captures={r['path']:r['sha256'] for r in original['source_snapshot']['captured_files']}
    for path in ['src/trafficflow/recurrent_state.py','src/trafficflow/adaptive_graph_state.py','src/trafficflow/graph_state.py']:
        if sha256_file(path)!=captures[path]:raise ValueError('Original constructor source changed;cannot assert same initial RNG weights')
    rows={};initials={}
    for seed in config['seeds']:
        name=f'seed{seed}_tcn_gru';row=replay['exports'][name]
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Frozen original seed predictor changed')
        torch.manual_seed(seed);model=make_network('tcn_gru',48)
        shared={k:v for k,v in model.state_dict().items() if not k.startswith('temporal')}
        digest=lambda state:hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in state.items())).hexdigest()
        if digest(shared)!=original['training'][name]['shared_initial_sha256']:raise ValueError('Original shared initialization not reproducible')
        initials[seed]=digest(model.state_dict());rows[f'seed{seed}_control']=row
    return replay_path,replay,rows,initials
