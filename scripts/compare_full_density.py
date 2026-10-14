"""Report matched density-loss ablations; retain the original cal selections."""
import argparse
import json
from pathlib import Path
from trafficflow.research_eval import paired_block_interval
from trafficflow.runtime import sha256_file


STUDIES=['A66_full_density_I210_W_autumn','A68_full_density_I5_N_winter']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite full density comparison')
    sources,rows=[],[]
    for study in STUDIES:
        source=Path('reports/research')/(study+'.json');run=json.loads(source.read_text())
        reference=run['assessment']['state_only']
        if len({r['sample_sequence_sha256'] for r in run['training'].values()})!=1:
            raise ValueError('Matched patch sampling differs')
        sources.append({'report':source.as_posix(),'sha256':sha256_file(source),
                        'shared_initialization_fingerprint':run['shared_initialization_fingerprint']})
        for variant,measured in run['assessment'].items():
            selected=run['selected']['joint'] if variant=='joint_challenger' else run['primary_per_model'][variant]
            state_interval=paired_block_interval([r['daily_State'] for r in reference['days']],
                                                 [r['daily_State'] for r in measured['days']])
            proxy_interval=paired_block_interval([-r['error_L1']/max(r['observation_increment_L1'],1e-9) for r in reference['public_increment_days']],
                                                [-r['error_L1']/max(r['observation_increment_L1'],1e-9) for r in measured['public_increment_days']])
            rows.append({'study':study,'panel':run['prepared']['panel'],'variant':variant,
                'cal_checkpoint_epoch':selected['epoch'],'cal_alpha':selected['alpha'],
                'is_primary_cal_choice':variant==run['selected']['primary']['name'],
                'S_state':measured['S_state'],'strong_LGB_State':run['baseline_assessment']['S_state'],
                'pooled_State_gain_vs_state_only':measured['S_state']-reference['S_state'],
                'paired_day_State_interval':state_interval,
                'public_coverage_increment_relative_L1':measured['public_coverage_increment_relative_L1'],
                'macro_proxy_error_reduction_vs_state_only':reference['public_coverage_increment_relative_L1']-measured['public_coverage_increment_relative_L1'],
                'paired_day_proxy_error_reduction_interval':proxy_interval,
                'legacy_target_pair_relative_L1':measured['legacy_target_pair_relative_L1'],
                'formal_State':None,'formal_Physics':None,'formal_Total':None})
    result={'source_reports':sources,'rows':rows,'selection_changed_after_assessment':False,
        'scope':'Same strict12/vq architecture, initialization, batch order and budget. Constant lambda=.01 means normalization also changes effective penalty strength. Compare cal-selected strategies, not fixed checkpoint/alpha.',
        'formal_Total':None,'limitations':'Two panel/fold cases, one fit seed, old public-train development dates. Macro-regime increment ratios and day-mean bootstrap intervals differ. No actual organizer boundary flux or formal Physics evidence.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print([(r['study'],r['variant'],r['pooled_State_gain_vs_state_only'],r['macro_proxy_error_reduction_vs_state_only']) for r in rows],flush=True)


if __name__=='__main__':main()
