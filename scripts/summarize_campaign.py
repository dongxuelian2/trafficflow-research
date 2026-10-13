"""Publish compact measured reports from a complete official-data campaign."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from trafficflow.data import Release

def family_mean(values):
    s=pd.Series(values,dtype=float)
    return float(s.groupby(s.index.map(lambda x:x.rsplit('_',1)[0])).mean().mean())

def compact_state(d):
    out={k:v for k,v in d.items() if k!='physics_diagnostics'}
    phys=d.get('physics_diagnostics',[])
    n=sum(x['observable_transitions'] for x in phys)
    possible=sum(x['possible_transitions'] for x in phys)
    out['physics_diagnostic_summary']={'formal_S_physics':None,'observable_transitions':n,
       'possible_transitions':possible,'coverage':n/max(possible,1),
       'conservation_RMS_vehicles':float(np.sqrt(sum(x['observable_transitions']*x['conservation_RMS_vehicles']**2 for x in phys if x['conservation_RMS_vehicles'] is not None)/n)) if n else None,
       'FD_mean_absolute_residual_per_lane_mean_days':float(np.mean([x['FD_mean_absolute_residual_per_lane'] for x in phys if x['FD_mean_absolute_residual_per_lane'] is not None]))}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,default=Path('artifacts/campaign_v1'))
    ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--submission',type=Path,default=Path('artifacts/submission_v1'))
    ap.add_argument('--queue-refinement',type=Path)
    ap.add_argument('--output',type=Path,default=Path('reports/official'));args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);panels=Release(args.release_root).panels()
    state_rows=[];queue_rows=[];od_rows=[];strata=[];summary={'panels':{},'formal_scores':{t:None for t in ['state','queue','physics','odme','total']}}
    for panel in panels:
        d=json.loads((args.campaign/panel/'experiment.json').read_text())
        if not d.get('complete'):raise ValueError('Incomplete campaign')
        selected=d['selection']['state'];s=d['state'][selected]['test'];baseline=d['state']['historical']['test']
        state_rows.append({'panel':panel,'historical_test':baseline['S_state'],'interpolation_calibration':d['state']['interpolation']['calibration']['S_state'],
            'selected_calibration':d['state'][selected]['calibration']['S_state'],'selected_test':s['S_state'],
            'gain_vs_historical_test':s['S_state']-baseline['S_state'],'selected':selected,'n_test_targets':s['n_targets']})
        for row in s['details']:strata.append({**row,'kind':'mask_regime'})
        for label in ['congested_observation_proxy','free_observation_proxy']:
            for row in (s.get(label) or {}).get('details',[]):strata.append({**row,'kind':label})
        sd={**d,'state':{k:{**v,**{split:compact_state(v[split]) for split in ['calibration','test'] if split in v}} for k,v in d['state'].items()}}
        summary['panels'][panel]=sd
        if d['queue']:
            name=d['selection']['queue'];q=d['queue'][name]
            final=q;final_name=name;threshold=str(q['tuning']['selected_threshold'])
            if args.queue_refinement:
                ref=json.loads((args.queue_refinement/panel/'research.json').read_text());candidate=ref['results'][ref['selected']]
                summary['panels'][panel]['queue_refinement']=ref
                if candidate['calibration']['proxy_IoU']>=q['calibration']['proxy_IoU']-1e-12:
                    final=candidate;final_name=ref['selected'];threshold=json.dumps(candidate['tuning']['selected']['thresholds'])
            queue_rows.append({'panel':panel,'persistence_test_proxy':d['queue']['persistence']['test']['proxy_IoU'],
               'initial_calibration_proxy':q['calibration']['proxy_IoU'],'initial_test_proxy':q['test']['proxy_IoU'],
               'selected_calibration_proxy':final['calibration']['proxy_IoU'],'selected_test_proxy':final['test']['proxy_IoU'],
               'threshold':threshold,'selected':final_name,'n_test_windows':q['n_test_windows'],
               'onset_test_proxy':final['test']['conditions'].get('queue_onset'),'ongoing_test_proxy':final['test']['conditions'].get('queue_ongoing')})
        for name,o in d['odme'].items():
            od_rows.append({'panel':panel,'method':name,'success':o.get('success'), 'S_link':o.get('S_link'),
              'heldout_S_link':o.get('sensor_holdout',{}).get('heldout_S_link'),'prior_L1_change':o.get('prior_L1_change'),
              'seconds':o['seconds'],'iterations':o.get('iterations'),'KKT_residual':o.get('KKT_max_absolute_residual'),'error':o.get('error')})
    st=pd.DataFrame(state_rows);q=pd.DataFrame(queue_rows);od=pd.DataFrame(od_rows)
    st.to_csv(args.output/'state_holdout.csv',index=False);q.to_csv(args.output/'queue_proxy_holdout.csv',index=False)
    od.to_csv(args.output/'odme_ablation.csv',index=False);pd.DataFrame(strata).to_csv(args.output/'state_strata.csv',index=False)
    summary['aggregate']={'State_holdout':family_mean(st.set_index('panel').selected_test.to_dict()),
      'Historical_holdout':family_mean(st.set_index('panel').historical_test.to_dict()),
      'Queue_proxy_IoU':family_mean(q.set_index('panel').selected_test_proxy.to_dict()),
      'Queue_persistence_proxy_IoU':family_mean(q.set_index('panel').persistence_test_proxy.to_dict()),
      'OD_ridge_S_link':family_mean(od[od.method=='ridge:0.05'].set_index('panel').S_link.to_dict())}
    summary['caveat']='D12_I5_N is development corridor; proxy/diagnostics/link fit never replace formal metrics.'
    summary['submission']=json.loads((args.submission/'verification.json').read_text())
    (args.output/'campaign_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    shutil.copyfile(args.submission/'verification.json',args.output/'submission_verification.json')
    # A static research figure with data available beside it.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    x=np.arange(len(st));axes[0].bar(x-.18,st.historical_test,.36,label='Chronological historical profile')
    axes[0].bar(x+.18,st.selected_test,.36,label='Selected model');axes[0].set_ylim(0,1)
    axes[0].set_xticks(x,st.panel,rotation=65,ha='right');axes[0].set_title('State: public train date holdout');axes[0].legend(fontsize=8)
    x=np.arange(len(q));axes[1].bar(x-.18,q.persistence_test_proxy,.36,label='Persistence')
    axes[1].bar(x+.18,q.selected_test_proxy,.36,label='Selected causal model');axes[1].set_ylim(0,1)
    axes[1].set_xticks(x,q.panel,rotation=65,ha='right');axes[1].set_title('Queue: observation-proxy IoU (NOT true S_queue)');axes[1].legend(fontsize=8)
    fig.savefig(args.output/'holdout_comparison.png',dpi=170);plt.close(fig)
    print(summary['aggregate']);print(summary['submission']['rows'])

if __name__=='__main__':main()
