"""Count legal released at-T inputs; do not inspect any horizon labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from trafficflow.data import Release, csv, eligible
from trafficflow.causal_queue import released_origin_rows

if __name__=='__main__':
    release=Release('data/kaggle_public')
    rows=[]
    for panel in release.panels():
        if panel.startswith('D12_I405'):continue
        for split in ('validation','private'):
            index=csv(release.root/'task2'/panel/split/'window_index.csv')
            for window in index.itertuples(index=False):
                frame=released_origin_rows(release,panel,split,window.forecast_origin)
                rows.append({'panel':panel,'split':split,'window_id':window.window_id,'condition':window.condition,
                    'origin':window.forecast_origin,'released_origin_eligible_share':float(eligible(frame).mean()),
                    'n_origin_rows':len(frame),'any_timestamp_after_origin_read':False})
    result={'experiment':'C04_legal_origin_availability','source':'public masked observations at T only, timestamp-filtered parquet',
        'rule':'Published Rules: forecast may use released timestamps at or before T',
        'rows':rows,'formal_Queue':None,'horizon_labels_read':False}
    Path('reports/research/C04_legal_origin_availability.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby(['panel','split']).released_origin_eligible_share.mean().to_string())
