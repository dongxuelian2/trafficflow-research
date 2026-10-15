"""Replay the small CPU attention-scope probe stored with the graph audit."""
import argparse
import json
from pathlib import Path
import torch
from trafficflow.adaptive_graph_state import AdaptiveSpatialBlock
from trafficflow.runtime import sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--note',type=Path,required=True);args=parser.parse_args();note=json.loads(args.note.read_text())
    if sha256_file('src/trafficflow/adaptive_graph_state.py')!=note['attention_source_sha256']:raise ValueError('Attention implementation changed')
    torch.set_num_threads(4);torch.manual_seed(note['probe_seed']);model=AdaptiveSpatialBlock(48,2);captured={}
    hook=model.attention.register_forward_pre_hook(lambda module,args,kwargs:captured.update(mask=kwargs['attn_mask'].detach().clone()),with_kwargs=True)
    x=torch.randn(1,48,2,3,requires_grad=True);model(x,torch.zeros(3,3),torch.zeros(3,3))[0,:,0,0].sum().backward();hook.remove()
    value=float(x.grad[0,:,0,1].abs().sum())
    if captured['mask'].numel()!=note['attention_mask_cells'] or int(torch.isfinite(captured['mask']).sum())!=note['finite_attention_mask_cells'] or abs(value-note['gradient_to_other_isolated_node_L1'])>1e-12:raise ValueError('Attention scope probe changed')
    print('global_attention_scope_probe_exact','other_isolated_node_gradient_L1',value,flush=True)


if __name__=='__main__':main()
