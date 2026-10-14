"""Fair cross-direction report for preregistered disjoint Queue phase studies."""
import argparse
import json
from pathlib import Path
from trafficflow.runtime import sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration',type=Path,default=Path('reports/research/C55_C58_five_phase_campaign.registered.json'));args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite campaign comparisons')
    registration=args.registration
    config=json.loads(registration.read_text());rows=[]
    for job in config['protocol']['jobs']:
        path=Path('reports/research')/(job['name']+'.json');run=json.loads(path.read_text())
        worker_file=next(r for r in run['source_snapshot']['captured_files'] if r['path']=='scripts/queue_five_phase_experiment.py')
        if worker_file['sha256']!=config['protocol']['worker_source_sha256']:
            raise ValueError('Campaign worker source changed across studies')
        if 'sampler_source_sha256' in config['protocol']:
            sampler_file=next(r for r in run['source_snapshot']['captured_files'] if r['path']=='src/trafficflow/queue_events.py')
            if sampler_file['sha256']!=config['protocol']['sampler_source_sha256']:
                raise ValueError('Campaign sampler source changed across studies')
        counts={}
        for record in run['cache_records']:
            counts.setdefault(record['phase'],[]).append({'seed':record['seed'],
                'requested_per_condition':record['sampler']['requested_per_condition'],
                'candidate_counts':record['sampler']['candidate_counts'],
                'selected_counts':record['sampler']['selected_counts']})
        selected=run['comparisons']['selected_vs_incumbent'];challenger=run['comparisons']['decision_selected_vs_incumbent']
        rows.append({'report':path.as_posix(),'report_sha256':sha256_file(path),'panel':job['panel'],'fit_end':job['fit_end'],
            'dates':run['dates'],'incumbent':run['incumbent'],'decision_selected':run['decision_selected'],'selected':run['selected'],
            'selection_gate':run['selection_gate'],'selected_vs_incumbent':selected,
            'decision_selected_vs_incumbent':challenger,'graph_mean_vs_incumbent':run['comparisons']['graph_mean_vs_incumbent'],
            'sample_counts':counts,'wall_seconds':run['seconds'],'gpu_peak_allocated_MiB':run['gpu_peak_allocated_MiB'],
            'formal_Queue':None})
    promoted=[r for r in rows if r['selected_vs_incumbent']['paired_macro_gain']>=.01 and
              r['selected_vs_incumbent']['ci95'][0]>0]
    passed=len({r['panel'] for r in promoted})>=2 and len({r['fit_end'] for r in promoted})>=2 and all(
        r['selected_vs_incumbent']['paired_macro_gain']>=-.01 for r in rows)
    result={'campaign_registered_sha256':sha256_file(registration),'rows':rows,
        'local_proxy_promotion_evidence_gate_passed':passed,'formal_Queue':None,'formal_Total':None,
        'limitations':'Four registered mechanisms,public noisy scored-positive proxy only. Independent mechanism phases do not make previously inspected public dates a final blind test. Three optimizer seeds share events. A passing proxy gate cannot establish official Queue or Total; no automatic deployment/upload.'}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    for r in rows:print(r['panel'],r['fit_end'],r['selected'],r['selected_vs_incumbent']['paired_macro_gain'],
        r['selected_vs_incumbent']['ci95'],'checkpoint_onset',r['sample_counts']['checkpoint'][0]['selected_counts']['queue_onset'],flush=True)


if __name__=='__main__':main()
