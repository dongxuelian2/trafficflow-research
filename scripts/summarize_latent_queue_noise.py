"""Checked seed uncertainty and contract/phase/noise summaries; no selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from trafficflow.runtime import sha256_file,capture_source_snapshot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();folder=Path('artifacts/research')/args.output.stem
    if args.output.exists() or folder.exists():raise FileExistsError('Never overwrite Queue noise summary')
    source_path=Path('reports/research/C62_CTM_Queue_hard_soft_teachers.json');audit_path=Path('reports/research/C63_CTM_Queue_hard_soft_independent_verification.json')
    source=json.loads(source_path.read_text());audit=json.loads(audit_path.read_text())
    if audit['registered']['protocol']['source_study_sha256']!=sha256_file(source_path) or audit['n_regenerated_scenes']!=268 or not audit['primary_selection_exact']:raise ValueError('Independent source verification missing')
    folder.mkdir(parents=True);snapshot=capture_source_snapshot(folder/'source_snapshot.zip');start=time.perf_counter();checks=[];rows=[];populations=[]
    for phase,arms in source['assessment'].items():
        record=next(r for r in source['datasets'] if r['phase']==phase)
        with np.load(record['path']) as data:
            valid=data['available'];populations.append({'phase':phase,'windows':len(valid),'condition_counts':record['conditions'],
                'eligible_cells':int(valid.sum()),'latent_observed_disagreement_cells':int(((data['latent']!=data['observed'])&valid).sum()),
                'zero_observed_positive_windows':record['n_zero_observed_positive_windows']})
        for contract in ['observed','latent']:
            base=pd.DataFrame(arms['observation_hard']['observed_cal'][contract]['scenes']).set_index(['seed','case','sigma'])
            for arm in ['observation_soft','latent_supervised']:
                candidate=pd.DataFrame(arms[arm]['observed_cal'][contract]['scenes']).set_index(['seed','case','sigma'])
                if not candidate.index.equals(base.index):raise ValueError('Seed diagnostic populations differ')
                gains=(candidate.macro_available_conditions-base.macro_available_conditions).groupby('seed').mean().sort_index();rng=np.random.default_rng(947211)
                samples=gains.to_numpy()[rng.integers(0,len(gains),(4000,len(gains)))].mean(axis=1);ci=np.quantile(samples,[.025,.975]);wanted=source['comparisons'][phase][arm][contract]
                if abs(gains.mean()-wanted['seed_mean_gain'])>1e-12 or not np.allclose(ci,wanted['ci95_seed_bootstrap'],atol=1e-12,rtol=0):raise ValueError('Seed uncertainty differs')
                checks.append({'phase':phase,'candidate':arm,'label_contract':contract,'seed_mean_gain':float(gains.mean()),'ci95_seed_bootstrap':ci.tolist(),'four_seed_diagnostic_exact':True})
        for arm,policies in arms.items():
            for contract,metric in policies['observed_cal'].items():
                frame=pd.DataFrame(metric['scenes'])
                for sigma,group in frame.groupby('sigma'):
                    rows.append({'phase':phase,'arm':arm,'label_contract':contract,'noise_kmh':float(sigma),
                        'equal_scene_available_condition_IoU':float(group.macro_available_conditions.mean()),'n_scenes_with_events':len(group),
                        'n_scenes_with_onset':sum('onset' in row for row in group.condition_IoU),'n_scenes_with_ongoing':sum('ongoing' in row for row in group.condition_IoU)})
    result={'source_study_sha256':sha256_file(source_path),'independent_audit_sha256':sha256_file(audit_path),'source_snapshot':snapshot,
        'cal_primary':source['selection']['primary'],'observed_cal_soft_gain':source['selection']['calibration_observed_gain_soft_vs_hard'],
        'n_seed_intervals_independently_verified':len(checks),'seed_diagnostics':checks,'noise_scene_summaries':rows,'population_diagnostics':populations,
        'all_observed_vs_latent_cal_threshold_choices_identical':all(p['observed']['selected']['setting']==p['latent']['selected']['setting'] for p in source['selection']['choices'].values()),
        'seconds':time.perf_counter()-start,'formal_Queue':None,'formal_Total':None,
        'scope':'Post-fit descriptive noise breakdown and independent4seed uncertainty,not additional selection or official score. Scene macro differs from window/condition macro. Latent-supervised label control is not a mathematical upper bound or a competition candidate.'}
    raw=json.dumps(result,indent=2,allow_nan=False);args.output.write_text(raw,encoding='utf-8');(folder/'experiment.json').write_text(raw,encoding='utf-8')
    print('latent_queue_summary','verified_intervals',len(checks),'rows',len(rows),'primary',result['cal_primary'],flush=True)


if __name__=='__main__':main()
