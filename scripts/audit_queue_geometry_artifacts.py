"""Rebuild all six frozen Queue operators solely from public network geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.data import Release
from trafficflow.state import neighbors
from trafficflow.queue_graph_geometry import travel_time_neighborhoods
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite geometry integrity evidence')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Frozen_Queue_geometry_independent_public_reconstruction',
        'scope':'C32/C34 direct,15min,30min operator and FD arrays; no traffic labels or refit',
        'reason':'Legacy artifact manifest included networks but omitted these six geometry NPZs; register current bytes after exact public rebuild',
        'formal_Queue':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip')
    release=Release('data/kaggle_public');source_path=Path('reports/research/C38_full_queue_probability_cache.json')
    source=json.loads(source_path.read_text());rows=[];sources={}
    for record in source['source_records']:
        study=Path(record['report']).stem;producer=Path('artifacts/research')/study
        run=json.loads(Path(record['report']).read_text());panel=run['registered']['protocol']['panel']
        cal=next(r for r in source['full_probability_caches'] if r['study']==study and r['phase']=='selection')
        if sha256_file(cal['path'])!=cal['sha256']:raise ValueError('Node-order reference changed')
        links=np.load(cal['path'],allow_pickle=False)['links'].tolist();net=release.network(panel);topology=release.topology(panel)
        for name in ['links.csv','fd_parameters.csv','lwr_mainline_topology.csv']:
            path=release.root/'corridors'/panel/'network'/name;sources[str(path)]=sha256_file(path)
        lane=net.lanes.reindex(links).to_numpy(float)
        geometry=np.stack([net.free_speed_kmh.reindex(links),net.capacity_vph.reindex(links)/lane,
                           net.k_jam.reindex(links)/lane,net.v_cut.reindex(links)]).astype(np.float32)
        for name,horizon in [('direct',None),('wave_15min',15.),('wave_30min',30.)]:
            if horizon is None:
                operators=[]
                for direction in ['incoming','outgoing']:
                    matrix=np.zeros((len(links),len(links)),np.float32)
                    for j,connected in enumerate(neighbors(topology,links,direction)):
                        if connected:matrix[j,connected]=1/len(connected)
                    operators.append(matrix)
            else:operators=travel_time_neighborhoods(links,net,topology,horizon)
            path=producer/(name+'_geometry.npz');saved=dict(np.load(path,allow_pickle=False))
            expected={'incoming':operators[0],'outgoing':operators[1],'geometry':geometry}
            if set(saved)!=set(expected) or not all(np.array_equal(saved[k],expected[k]) for k in saved):
                raise ValueError('Geometry arrays do not reproduce: '+str(path))
            rows.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path),
                'source_study':study,'n_nodes':len(links),'all_arrays_exact_from_public_network':True,
                'previous_manifest_missing':True})
    result={'registered':registered,'source_snapshot':snapshot,'source_probability_report_sha256':sha256_file(source_path),
        'public_network_files_sha256':sources,'rows':rows,'formal_Queue':None,'formal_Total':None,
        'scope':'Current NPZ bytes and exact public geometry content; no retrospective claim of prior byte hashes. Frozen model files and predictions unchanged.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8')
    (folder/'experiment.json').write_text(text,encoding='utf-8');print('publicly_rebuilt_geometry_files',len(rows),flush=True)


if __name__=='__main__':main()
