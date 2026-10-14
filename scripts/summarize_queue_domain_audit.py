"""Correct paired comparisons while retaining old Queue choices and reports."""
import argparse
import json
from pathlib import Path
from compare_queue_events import comparison
from trafficflow.runtime import sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite corrected Queue evidence')
    audit=json.loads(args.audit.read_text());results=[]
    for source in audit['source_records']:
        path=Path(source['report']);old=json.loads(path.read_text());study=path.stem
        rows=[r for r in audit['rows'] if r['study']==study];new={'dates':old['dates'],'assessment':[]}
        for run in old['assessment']:
            phase='assessment_'+str(run['seed'])
            new['assessment'].append({'seed':run['seed'],'models':{r['model']:r['full_domain_then_score'] for r in rows if r['phase']==phase}})
        selected=old['selected']
        results.append({'study':study,'source_report_sha256':sha256_file(path),
            'old_cal_selected_model_retained':selected,
            'old_selected_model_predictions_unchanged':all(r['changed_scored_decisions']==0 for r in rows if r['model']==selected),
            'cal_candidate_model_ranking_reselected':False,
            'old_selected_comparison':old['selected_comparison'],
            'corrected_selected_comparison':comparison(new,'shared_threshold',selected),
            'corrected_all_candidate_comparisons':{name:comparison(new,'shared_threshold',name) for name in old['selection'] if name!='shared_threshold'}})
    result={'source_audit_sha256':sha256_file(args.audit),'studies':results,
        'changed_scored_predictions_across_replayed_model_phase_calls':sum(r['changed_scored_decisions'] for r in audit['rows']),
        'legacy_proxy_scores_independently_replayed_exact':all(r['legacy_proxy_replay_exact'] for r in audit['rows']),
        'formal_Queue':None,'formal_Total':None,
        'limitations':'Fixed old models/thresholds and origin sets; corrected decision domain only. No cal reselection, refit, fresh split or actual latent Queue truth. Future experiments must predict full public template before score eligibility.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print([(r['study'],r['corrected_selected_comparison']['paired_macro_gain'],r['corrected_selected_comparison']['ci95']) for r in results],flush=True)


if __name__=='__main__':main()
