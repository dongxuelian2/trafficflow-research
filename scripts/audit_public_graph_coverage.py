"""Static released graph coverage;read geometry columns only,never traffic labels."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.state import neighbors
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite graph audit')
    registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Published_measured_graph_coverage_and_static_geometry','data':'Released network CSVs and only link_id/milepost columns from first public TRAIN partition',
        'graph':'Exact existing neighbors() traversal;undirected components use union of directional measured adjacency',
        'interpretation':'Coverage diagnostic only. Geometry order does not establish physical connection. No graph change/model fitting or predictive validation.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();release=Release('data/kaggle_public');rows=[];hashes={}
    for panel in release.panels():
        first=release.files(panel,'train')[0];meta=pd.read_parquet(first,columns=['link_id','milepost']);meta.link_id=meta.link_id.astype(str)
        position=meta.assign(milepost=pd.to_numeric(meta.milepost,errors='coerce')).groupby('link_id').milepost.mean();links=sorted(position.index);topology=release.topology(panel)
        for p in [first]+sorted((release.root/'corridors'/panel/'network').glob('*.csv')):hashes[p.as_posix()]=sha256_file(p)
        incoming=neighbors(topology,links,'incoming');outgoing=neighbors(topology,links,'outgoing');adj=[set(a+b) for a,b in zip(incoming,outgoing)]
        for j,ns in enumerate(adj):
            for k in list(ns):adj[k].add(j)
        remaining=set(range(len(links)));components=[]
        while remaining:
            pending=[remaining.pop()];component=[]
            while pending:
                j=pending.pop();component.append(j);fresh=adj[j]&remaining;remaining-=fresh;pending.extend(fresh)
            components.append(component)
        external=set()
        for link in links:
            if link not in topology.index:continue
            for field in ['incoming_link_ids','outgoing_link_ids']:
                for text in str(topology.loc[link,field]).split(';'):
                    if text and text!='nan' and text not in topology.index:external.add(text)
        isolated=[j for j,a,b in zip(range(len(links)),incoming,outgoing) if not(a or b)];ordered=position.sort_values();gaps=np.diff(ordered.to_numpy(float))
        rows.append({'panel':panel,'n_measured_links':len(links),'n_measured_in_published_topology':sum(k in topology.index for k in links),
            'n_isolated_measured_links':len(isolated),'isolated_fraction':len(isolated)/len(links),'n_missing_incoming':sum(not x for x in incoming),
            'n_missing_outgoing':sum(not x for x in outgoing),'n_directional_measured_edges':sum(map(len,incoming))+sum(map(len,outgoing)),
            'n_undirected_components':len(components),'component_sizes':sorted(map(len,components),reverse=True),'n_external_referenced_link_ids':len(external),
            'n_distinct_mileposts':int(position.nunique()),'adjacent_geometry_gap_milepost':{'median':float(np.median(gaps)) if len(gaps) else None,'max':float(np.max(gaps)) if len(gaps) else None},
            'isolated_examples':[{'link_id':links[j],'milepost':float(position.loc[links[j]]),'order_index':str(topology.loc[links[j],'order_index']) if links[j] in topology.index else None} for j in isolated[:5]]})
    result={'registered':registered,'source_snapshot':snapshot,'panels':rows,'source_files_sha256':hashes,'seconds':time.perf_counter()-start,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'limits':'A disconnected released measured graph is not proof of a reader defect or missing physical roads. Static inferred edges would be a separate statistical-model hypothesis.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'audit.json').write_text(text,encoding='utf-8')
    print('public_graph_audit',[(r['panel'],r['n_isolated_measured_links'],r['n_measured_links']) for r in rows],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
