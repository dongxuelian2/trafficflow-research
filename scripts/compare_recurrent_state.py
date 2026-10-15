"""Two-fold model comparison retaining every original cal-selected policy."""
import argparse
import json
from pathlib import Path
import numpy as np
from trafficflow.research_eval import paired_block_interval
from trafficflow.runtime import sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite State architecture comparisons')
    rows=[]
    for path in args.study:
        study=json.loads(path.read_text());config=study['registered']['protocol'];folder=Path('artifacts/research')/path.stem
        if sha256_file(folder/'selection_sealed.json')!=study['selection_seal_sha256']:raise ValueError('Cal choice changed')
        pairs=[]
        for seed in config['seeds']:
            left,right=f'seed{seed}_tcn',f'seed{seed}_tcn_gru';a,b=study['training'][left],study['training'][right]
            if a['shared_initial_sha256']!=b['shared_initial_sha256'] or a['patch_sequence_sha256']!=b['patch_sequence_sha256']:
                raise ValueError('Neural comparisons are not matched')
            for name in [left,right]:
                if sha256_file(folder/(name+'_exposure.npz'))!=study['training'][name]['exposure_sha256']:
                    raise ValueError('Exposure counts changed')
            with np.load(folder/(left+'_exposure.npz')) as exp_a,np.load(folder/(right+'_exposure.npz')) as exp_b:
                if exp_a.files!=exp_b.files or any(not np.array_equal(exp_a[k],exp_b[k]) for k in exp_a.files):
                    raise ValueError('Paired architectures saw different targets')
            x,y=study['assessment'][left],study['assessment'][right]
            strata={}
            for key in x['diagnostic_strata']:
                dx,dy=x['diagnostic_strata'][key],y['diagnostic_strata'][key]
                if dx['n']!=dy['n']:raise ValueError('Diagnostic target population differs')
                strata[key]={'n':dx['n'],'GRU_minus_TCN_RMSE_speed':dy['rmse_speed']-dx['rmse_speed'] if dx['n'] else None,
                    'GRU_minus_TCN_RMSE_lane_flow':dy['rmse_lane_flow']-dx['rmse_lane_flow'] if dx['n'] else None}
            pairs.append({'seed':seed,'cal_alpha_TCN':study['selection']['choices'][left]['alpha'],
                'cal_alpha_GRU':study['selection']['choices'][right]['alpha'],
                'selected_epoch_TCN':max(a['history'],key=lambda r:r['cal_State'])['epoch'],
                'selected_epoch_GRU':max(b['history'],key=lambda r:r['cal_State'])['epoch'],
                'TCN_State':x['S_state'],'GRU_State':y['S_state'],'GRU_minus_TCN_gain':y['S_state']-x['S_state'],
                'paired_day_diagnostic':paired_block_interval([r['daily_State'] for r in x['days']],[r['daily_State'] for r in y['days']]),
                'regime_gains':{u['regime']:v['S_state']-u['S_state'] for u,v in zip(x['details'],y['details'])},
                'stratum_RMSE_differences':strata,'parameters_TCN':a['parameters'],'parameters_GRU':b['parameters'],
                'seconds_TCN':a['seconds'],'seconds_GRU':b['seconds'],
                'target_grid_cells_supervised':a['unique_target_grid_cells_supervised'],
                'target_grid_cells_available':a['total_available_target_grid_cells'],'actual_exposures_exactly_matched':True})
        metrics=study['assessment'];primary=study['selection']['primary']
        rows.append({'study':path.stem,'source_report_sha256':sha256_file(path),'panel':config['panel'],'fold':config['fold'],
            'primary':primary,'incumbent_State':metrics['incumbent']['S_state'],'primary_State':metrics[primary]['S_state'],
            'primary_gain':metrics[primary]['S_state']-metrics['incumbent']['S_state'],'seed_pairs':pairs,
            'architecture_seed_means':{kind:metrics[kind+'_seed_mean']['S_state'] for kind in ['tcn','tcn_gru']},
            'incumbent_context_only_effect':{'cal':study['selection']['cal_metrics']['incumbent_windowed']['S_state']-
                study['selection']['cal_metrics']['incumbent']['S_state'],
                'assessment':metrics['incumbent_windowed']['S_state']-metrics['incumbent']['S_state']},
            'seconds':study['seconds'],'GPU_peak_allocated_MiB':study['gpu_peak_allocated_MiB']})
    result={'studies':rows,'all_original_selections_preserved':True,'new_formal_model_promoted':False,
        'formal_State':None,'formal_Physics':None,'formal_Total':None,
        'limits':'Two seeds share days. Cal checkpoints/alphas are individually selected policies;not a fixed-checkpoint gradient experiment. '
        'GRU adds21313parameters. Fixed context versus full-day incumbent is separately diagnosed. '
        'Intervals are day-mean conditional diagnostics,not formal-score or model-selection uncertainty.'}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print([(r['panel'],r['primary'],r['primary_gain'],[p['GRU_minus_TCN_gain'] for p in r['seed_pairs']]) for r in rows],flush=True)


if __name__=='__main__':main()
