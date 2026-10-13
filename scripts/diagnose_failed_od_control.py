"""Public-only diagnostics after K01 failure; no reference path truth."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from trafficflow.data import Release
from trafficflow.odme import operator
from trafficflow.metrics import od_link_score

if __name__=='__main__':
    release=Release('data/kaggle_public')
    original=pd.read_csv('artifacts/submission_best/odme.csv',dtype={'panel':str,'path_id':str,'departure_time':str})
    candidate=pd.read_csv('artifacts/research/K01_OD_only_bayesian/odme.csv',dtype={'panel':str,'path_id':str,'departure_time':str})
    rows=[]
    for panel in release.panels():
        for split in ('validation','private'):
            A,c,b,paths=operator(release,panel,split)
            tokens=release.template(4,panel,split).departure_time.unique()
            vectors=[]
            for frame in (original,candidate):
                sub=frame[(frame.panel==panel)&frame.departure_time.isin(tokens)]
                vectors.append(sub.set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float))
            old,new=vectors
            delta=new-old
            G=(A@A.T).toarray()
            row_component=A.T@np.linalg.pinv(G,rcond=1e-10)@(A@delta)
            null_component=delta-row_component
            prior_change_old=np.abs(old-b).sum()
            rows.append({'panel':panel,'split':split,'formal_ODME':None,
                'prior_S_link':od_link_score(A,b,c),'ridge_S_link':od_link_score(A,old,c),'bayesian_S_link':od_link_score(A,new,c),
                'ridge_relative_prior_L1_change':float(prior_change_old/max(b.sum(),1)),
                'bayesian_relative_prior_L1_change':float(np.abs(new-b).sum()/max(b.sum(),1)),
                'prior_deviation_ratio_bayes_to_ridge':float(np.abs(new-b).sum()/max(prior_change_old,1e-9)),
                'L2_change_nullspace_share':float(np.linalg.norm(null_component)/max(np.linalg.norm(delta),1e-9)),
                'nullspace_link_load_error_relative':float(np.linalg.norm(A@null_component)/max(np.linalg.norm(c),1)),
                'prediction_L1_difference_relative':float(np.abs(delta).sum()/max(old.sum(),1)),
                'bayes_to_ridge_total_demand_ratio':float(new.sum()/max(old.sum(),1))})
    result={'experiment':'K01_public_failure_diagnosis','data':'released counts/priors/network plus our own two prediction files',
        'formal_feedback':{'baseline_total':.82940,'candidate_total':.76474},'rows':rows,
        'interpretation':'Nullspace movement is unobserved by link counts. These diagnostics characterize the rejected model; they cannot identify the hidden reference or establish the precise cause of score loss.'}
    Path('reports/research/K01_public_failure_diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby('split').mean(numeric_only=True).to_string())
