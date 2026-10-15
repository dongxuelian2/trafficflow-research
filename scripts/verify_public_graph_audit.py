"""Check published graph counts through direct ID edges and union-find."""
import argparse
import json
from pathlib import Path
import time
import pandas as pd
from trafficflow.runtime import sha256_file,capture_source_snapshot
from trafficflow.research_eval import register_protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    folder=Path('artifacts/research')/args.output.stem
    if folder.exists() or args.output.exists():raise FileExistsError('Never overwrite graph verification')
    source=json.loads(args.study.read_text());registered=register_protocol(args.output.with_suffix('.registered.json'),{
        'id':'Direct_public_graph_ID_and_components_verification','source_study_sha256':sha256_file(args.study),
        'scope':'All10published graphs. Direct CSV ID filtering and union-find,without neighbors() helper. Metadata only,no performance claim.',
        'formal_State':None,'formal_Physics':None,'formal_Total':None})
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();checks=[]
    for path,digest in source['source_files_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Published static sources changed')
    for record in source['panels']:
        panel=record['panel'];root=Path('data/kaggle_public/corridors')/panel
        first=sorted((root/'train/mainline_states_masked').rglob('*.parquet'))[0];meta=pd.read_parquet(first,columns=['link_id','milepost']);meta.link_id=meta.link_id.astype(str)
        position=meta.assign(milepost=pd.to_numeric(meta.milepost,errors='coerce')).groupby('link_id').milepost.mean();links=set(position.index)
        topology=pd.read_csv(root/'network/lwr_mainline_topology.csv',dtype=str,keep_default_na=False).set_index('link_id')
        if links!=set(topology.index):raise ValueError('Direct verifier assumes every topology row is measured;requires traversal if release changes')
        parent={k:k for k in links}
        def find(k):
            while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
            return k
        incoming={};outgoing={};external=set()
        for k in links:
            for field,store in [('incoming_link_ids',incoming),('outgoing_link_ids',outgoing)]:
                ids={text.strip() for text in topology.loc[k,field].split(';') if text.strip() and text.strip()!='nan'}
                store[k]=ids&links;external|=ids-links
                for other in store[k]:parent[find(other)]=find(k)
        sizes={}
        for k in links:sizes[find(k)]=sizes.get(find(k),0)+1
        got={'n_measured_links':len(links),'n_measured_in_published_topology':len(links),
            'n_isolated_measured_links':sum(not(incoming[k] or outgoing[k]) for k in links),
            'n_missing_incoming':sum(not v for v in incoming.values()),'n_missing_outgoing':sum(not v for v in outgoing.values()),
            'n_directional_measured_edges':sum(map(len,incoming.values()))+sum(map(len,outgoing.values())),
            'n_undirected_components':len(sizes),'component_sizes':sorted(sizes.values(),reverse=True),'n_external_referenced_link_ids':len(external),
            'n_distinct_mileposts':int(position.nunique())}
        if any(record[k]!=v for k,v in got.items()):raise ValueError('Direct graph metadata differs')
        checks.append({'panel':panel,'all_direct_edge_isolation_component_counts_exact':True})
    result={'registered':registered,'source_snapshot':snapshot,'n_panels_verified':len(checks),'checks':checks,'seconds':time.perf_counter()-start,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'scope':'Metadata count verification,not proof of physical connections or neural attention restrictions.'}
    text=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(text,encoding='utf-8');(folder/'verification.json').write_text(text,encoding='utf-8')
    print('public_graph_verified',len(checks),'panels','seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
