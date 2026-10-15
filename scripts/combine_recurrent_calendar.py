"""Secondary contiguous-calendar summary with one frozen source policy."""
import argparse
import json
from pathlib import Path
from trafficflow.research_eval import state_from_sufficient_stats,paired_block_interval
from trafficflow.runtime import sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite calendar aggregation')
    original_path=Path('reports/research/A72b_recurrent_I210_W_autumn_full_day.json')
    extension_path=Path('reports/research/A81b_I210_W_public_outage_stress.json')
    original=json.loads(original_path.read_text());extension=json.loads(extension_path.read_text())
    if extension['registered']['protocol']['source_study_sha256']!=sha256_file(original_path):raise ValueError('Frozen model differs')
    roles=extension['registered']['protocol']['roles'];results={};combined={}
    for role in roles:
        first=original['assessment'][role]['days'];second=[r for r in extension['official_control_daily_stats'] if r['role']==role]
        dates=[r['date'] for r in first+second]
        if len(dates)!=28 or dates!=sorted(set(dates)):raise ValueError('Dates overlap or unordered')
        combined[role]=first+second;results[role]=state_from_sufficient_stats(combined[role])
    comparisons={role:{'pooled_State_gain_vs_incumbent':results[role]['S_state']-results['incumbent']['S_state'],
        'paired_day_mean_interval':paired_block_interval([r['daily_State'] for r in combined['incumbent']],
            [r['daily_State'] for r in combined[role]])} for role in roles if role!='incumbent'}
    result={'original_sha256':sha256_file(original_path),'additional_control_report_sha256':sha256_file(extension_path),
        'dates':[r['date'] for r in combined['incumbent']],'source_primary':original['selection']['primary'],
        'same_weights_and_blends':True,'metrics':results,'comparisons':comparisons,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,'selection_changed':False,
        'scope':'Secondary28day pooled summary of original14day assessment plus14new-date public-template controls. '
        'Additional synthetic-outage target populations are excluded. Already studied public calendars,not a final/private test. '
        'No aggregation result overrides the original model choice or project multi-direction requirements.'}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print('28day_metrics',{k:v['S_state'] for k,v in results.items()},'comparisons',comparisons,flush=True)


if __name__=='__main__':main()
