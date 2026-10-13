"""Public incidence equivalence classes and K01 unmeasured-load movement."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from trafficflow.data import Release,csv
from trafficflow.odme import operator
from trafficflow.research_eval import register_protocol


if __name__=='__main__':
    output=Path('reports/research/D05_public_OD_identifiability.json')
    if output.exists():raise FileExistsError('Never overwrite structural evidence')
    registered=register_protocol(output.with_suffix('.registered.json'),{
        'id':'D05_public_OD_structure','data':'Public network and measured counts, our own frozen and rejected predictions',
        'hypothesis':'Repeated incidence columns and low row-space leverage leave path allocations dependent on prior assumptions.',
        'formal_ODME':None,'unmeasured_link_truth':'unavailable'})
    release=Release('data/kaggle_public')
    old=pd.read_csv('artifacts/submission_best/odme.csv',dtype={'panel':str,'path_id':str,'departure_time':str})
    new=pd.read_csv('artifacts/research/K01_OD_only_bayesian/odme.csv',dtype={'panel':str,'path_id':str,'departure_time':str})
    rows=[]
    for panel in release.panels():
        for split in ('validation','private'):
            A,c,b,paths=operator(release,panel,split)
            dense=A.toarray()
            packed=np.packbits(dense.astype(bool),axis=0).T
            _,representative,groups,counts=np.unique(packed,axis=0,return_index=True,return_inverse=True,return_counts=True)
            G=(A@A.T).toarray()
            pinv=np.linalg.pinv(G,rcond=1e-10)
            leverage=np.sum(dense*(pinv@dense),axis=0)
            B=dense[:,representative]
            gp=np.linalg.pinv(B@B.T,rcond=1e-10)
            group_leverage=np.sum(B*(gp@B),axis=0)
            counted=csv(release.root/'task4'/panel/split/'synthetic_link_counts.csv').link_id.tolist()
            incidence=csv(release.root/'corridors'/panel/'network'/'path_link_incidence.csv').drop_duplicates(['link_id','path_id'])
            links=sorted(incidence.link_id.unique())
            li={l:i for i,l in enumerate(links)}
            pi={p:i for i,p in enumerate(paths.path_id)}
            full=sparse.csr_matrix((np.ones(len(incidence)),(incidence.link_id.map(li),incidence.path_id.map(pi))),shape=(len(links),len(paths)))
            tokens=release.template(4,panel,split).departure_time.unique()
            vectors=[]
            for frame in (old,new):
                sub=frame[(frame.panel==panel)&frame.departure_time.isin(tokens)]
                vectors.append(sub.set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float))
            f0,f1=vectors
            if not np.isfinite(f0).all() or not np.isfinite(f1).all():raise ValueError('Incomplete prediction vectors')
            unmeasured=np.array([l not in counted for l in links])
            load0,load1=full@f0,full@f1
            equivalents=pd.DataFrame({'group':groups,'OD':paths.origin_zone.astype(str)+'|'+paths.destination_zone.astype(str)})
            rows.append({'panel':panel,'split':split,'n_paths':len(paths),'n_counted_links':A.shape[0],
                'numerical_rank':int(np.linalg.matrix_rank(G,tol=np.linalg.norm(G,2)*1e-10)),
                'path_coordinate_identifiable_at_tolerance_1e6':int((leverage>=1-1e-6).sum()),
                'max_path_leverage':float(leverage.max()),'n_zero_incidence_paths':int((dense.sum(axis=0)==0).sum()),
                'n_unique_counted_incidence_patterns':len(counts),'largest_equivalence_class':int(counts.max()),
                'paths_in_repeated_patterns':int(counts[counts>1].sum()),
                'patterns_spanning_multiple_OD_pairs':int((equivalents.groupby('group').OD.nunique()>1).sum()),
                'identifiable_pattern_total_count_at_tolerance_1e6':int((group_leverage>=1-1e-6).sum()),
                'n_full_incidence_links':len(links),'n_unmeasured_full_links':int(unmeasured.sum()),
                'K01_unmeasured_load_relative_L1_change':float(np.abs(load1-load0)[unmeasured].sum()/max(load0[unmeasured].sum(),1)),
                'K01_measured_load_relative_L1_change':float(np.abs(A@(f1-f0)).sum()/max(c.sum(),1)),
                'formal_ODME':None})
    result={'registered':registered,'rows':rows,
            'interpretation':'Coordinate leverage tests unconstrained linear identifiability. Nonnegativity can identify some particular boundary solutions but does not make general path flows recoverable. Full-link load movement has no unmeasured truth for scoring; it is a rejected-model behavior diagnostic.'}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby('split').mean(numeric_only=True).to_string())
