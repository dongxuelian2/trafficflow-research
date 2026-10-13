"""Collect measured fixture experiments. Does not invent official scores."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from trafficflow.runtime import sha256_file

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--compare',type=Path)
    ap.add_argument('--output',type=Path,default=Path('reports/fixture'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    state_rows,queue_rows,od_rows=[],[],[]
    for path in sorted(args.models.glob('*/experiment.json')):
        data=json.loads(path.read_text())
        if data['dataset']!='synthetic_fixture':
            raise ValueError('This report script is for clearly separated fixture evidence')
        dest=args.output/path.parent.name;dest.mkdir(exist_ok=True)
        shutil.copyfile(path,dest/'experiment.json')
        for method,row in data['state'].items():
            state_rows.append({'panel':data['panel'],'method':method,
               'calibration_S_state':row.get('calibration',{}).get('S_state'),
               'test_S_state':row.get('test',{}).get('S_state'),'seconds':row['seconds'],
               'selected':method==data['selection']['state'],'error':row.get('error','')})
            for detail in row.get('test',{}).get('details',[]):
                state_rows.append({'panel':data['panel'],'method':method+':'+detail['regime'],
                 'test_S_state':detail['S_state'],'n_targets':detail['n'],
                 'rmse_speed_kmh':detail.get('rmse_speed_kmh'),'rmse_flow_per_lane':detail.get('rmse_flow_per_lane')})
        for method,row in data['queue'].items():
            queue_rows.append({'panel':data['panel'],'method':method,
               'calibration_proxy_IoU':row['calibration']['proxy_IoU'],'test_proxy_IoU':row['test']['proxy_IoU'],
               'seconds':row['seconds'],'threshold':row['tuning']['selected_threshold'],
               'selected':method==data['selection']['queue']})
        for method,row in data['odme'].items():
            od_rows.append({'panel':data['panel'],'method':method,'S_link':row.get('S_link'),
              'prior_L1_change':row.get('prior_L1_change'),'seconds':row['seconds'],
              'S_ODME':None,'error':row.get('error','')})
    pd.DataFrame(state_rows).to_csv(args.output/'state_ablation.csv',index=False)
    pd.DataFrame(queue_rows).to_csv(args.output/'queue_ablation.csv',index=False)
    pd.DataFrame(od_rows).to_csv(args.output/'odme_ablation.csv',index=False)
    report={'dataset':'artificial fixture only','formal_total':None,'official_download_status':'HTTP 401',
            'submissions':{},'reproduction':{}}
    for name in ['state.csv','queue.csv','odme.csv','submission.csv']:
        path=args.predictions/name
        report['submissions'][name]={'bytes':path.stat().st_size,'sha256':sha256_file(path),
                                     'rows':sum(len(c) for c in pd.read_csv(path,chunksize=100000))}
        if args.compare:
            other=args.compare/name
            equal=sha256_file(path)==sha256_file(other)
            left=pd.read_csv(path);right=pd.read_csv(other)
            if left.shape!=right.shape or not left.columns.equals(right.columns):
                raise ValueError('Reproduction shape mismatch')
            differences={}
            for column in ['speed_kmh','flow_vph','queue_pred','path_flow']:
                if column in left:
                    differences[column]=float(np.max(np.abs(left[column].to_numpy()-right[column].to_numpy())))
            report['reproduction'][name]={'byte_identical':equal,'maximum_absolute_differences':differences}
    if (args.predictions/'prediction_report.json').exists():
        shutil.copyfile(args.predictions/'prediction_report.json',args.output/'prediction_report.json')
    (args.output/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
