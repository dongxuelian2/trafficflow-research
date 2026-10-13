"""Compare independently retrained common A01/A09 model tensors exactly."""
import json
from pathlib import Path
import torch
from trafficflow.runtime import sha256_file

if __name__=='__main__':
    first=Path('artifacts/research/A01_graph_run01')
    second=Path('artifacts/research/A09_public_density_increment')
    rows=[]
    for name in ('temporal','graph','graph_FD'):
        a=torch.load(first/(name+'.pt'),map_location='cpu',weights_only=True)
        b=torch.load(second/(name+'.pt'),map_location='cpu',weights_only=True)
        if a.keys()!=b.keys():raise ValueError('Model parameters differ')
        maximum=max(float((a[key]-b[key]).abs().max()) for key in a)
        if maximum!=0:raise ValueError('Repeated model weights differ')
        rows.append({'model':name,'maximum_parameter_difference':maximum,
                     'first_sha256':sha256_file(first/(name+'.pt')),
                     'second_sha256':sha256_file(second/(name+'.pt'))})
    result={'experiment':'A01_independent_process_training_repeat','models':rows,
        'validation':'Exact tensor parity after independent training, same seed and cached public-train inputs.',
        'limitation':'This repeats training from cached inputs, not a complete independent full-data reprocessing and seven-million-row reproduction.'}
    Path('reports/research/A01_training_repeat.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)
