"""Freeze local evidence, retained failures and checksums for this phase.

No network, submission, publication, or goal-completion side effects.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def main():
    report = Path('reports/research')
    outputs = [report/'model_artifacts_checkpoint_ctm_od_v1.json',
               report/'ctm_od_verification_record_v1.json', report/'research_checkpoint_ctm_od_v1.json',
               report/'A46_D12_I405_S_autumn_cache.json']
    if any(p.exists() for p in outputs):
        raise FileExistsError('Never overwrite a phase checkpoint')
    old_path = report/'model_artifacts_checkpoint_replication_v4.json'
    old = json.loads(old_path.read_text())
    artifacts = []
    for row in old['artifacts']:
        path = Path(row['path'])
        if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise ValueError('Previous preserved artifact changed: '+str(path))
        artifacts.append(dict(row))
    classes = {
        'A46_D12_I405_S_autumn_cache': 'Completed public-train feature cache',
        'A47_adaptive_attention_D12_I405_S': 'Completed development model, not promoted',
        'B07_frozen_adaptive_CTM_stress': 'Completed frozen-model simulation',
        'B07_frozen_adaptive_CTM_repeat': 'Completed independent-process replay',
        'D09_entropy_OD_uncertainty_mismatch': 'Preserved failed aggregation; not a completed assessment result',
        'D10_entropy_OD_uncertainty_mismatch_repaired': 'Completed uncertainty/point-recovery research diagnostic'}
    for folder, kind in classes.items():
        for path in sorted((Path('artifacts/research')/folder).rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.pkl', '.pt', '.npz', '.zip'}:
                artifacts.append({'path': path.as_posix(), 'bytes': path.stat().st_size,
                    'sha256': sha256_file(path), 'artifact_class': kind})
    formal_digest = '43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1'
    preserved = []
    for name, digest in [
        ('artifacts/frozen_baselines/57138874/submission.csv', formal_digest),
        ('artifacts/submission_best/submission.csv', formal_digest),
        ('artifacts/research/K03_I5_N_attention_State_Physics/submission.csv', 'a4645b6a5100b71b138afc37ca89bf167dd0026e7243398bba29e69d00ffd694'),
        ('artifacts/research/K04_I5_N_attention_density_State_Physics/submission.csv', '404e059f8169aa71c0a5387a40a544c346b889c98711ea03299def4d7a8560b6')]:
        path = Path(name)
        if sha256_file(path) != digest:
            raise ValueError('Preserved complete submission changed: '+name)
        preserved.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest})
    locks = {name: sha256_file(name) for name in old['dependency_locks']}
    if locks != old['dependency_locks']:
        raise ValueError('Dependency locks changed')
    stamp = datetime.now(timezone.utc).isoformat()
    manifest = {'purpose': 'Local model/config/cache/source integrity, with failed-run preservation explicitly marked',
        'created_UTC': stamp, 'formal_best_ref': 57138874, 'formal_best_total': .82940,
        'formal_best_submission_sha256': formal_digest, 'previous_manifest_sha256': sha256_file(old_path),
        'n_previous_artifacts_rehashed_unchanged': len(old['artifacts']),
        'artifacts': artifacts, 'n_artifacts': len(artifacts), 'preserved_complete_submissions': preserved,
        'dependency_locks': locks,
        'limitations': 'Hashes are records, not redistributed weights or data. Cache and calibration arrays are included for this phase; old manifest scope unchanged. No new formal score.'}
    cache = json.loads(Path('artifacts/research/A46_D12_I405_S_autumn_cache/prepared.json').read_text())
    a = json.loads((report/'A47_adaptive_attention_D12_I405_S.json').read_text())
    b = json.loads((report/'B07_frozen_adaptive_CTM_verified.json').read_text())
    d = json.loads((report/'D10_entropy_OD_uncertainty_verified.json').read_text())
    if not b['independent_process_registered_rows_summary_exact'] or d['max_score_drift'] != 0 or d['optimizer_failures'] != 0:
        raise ValueError('Phase independent verification incomplete')
    gain = a['selected_assessment']['S_state']-a['baseline_assessment']['S_state']
    if gain >= .002 or a['paired_daily_interval']['ci95'][0] > 0:
        raise ValueError('Expected A47 non-promotion evidence changed; manually audit policy')
    verification = {'created_UTC': stamp, 'tests': {
        'research': {'passed': 47, 'seconds': 12.36,
            'command': '.venv-research/Scripts/python.exe -m pytest -q --basetemp E:/ieee/.cache/pytest_ctm_uncertainty_phase_research'},
        'CPU': {'passed': 42, 'skipped_optional_torch_modules': 3, 'seconds': 7.07,
            'command': '.venv-repro/Scripts/python.exe -m pytest -q --basetemp E:/ieee/.cache/pytest_ctm_uncertainty_phase_cpu'},
        'scope': 'Observed terminal test results; meaningful adapter contamination, independent Hessian inverse and equal-case calibration checks added.'},
        'compileall': 'src/scripts/tests completed; checkpoint writer compiled separately after creation',
        'ctm_independent_verification_sha256': sha256_file(report/'B07_frozen_adaptive_CTM_verified.json'),
        'OD_independent_verification_sha256': sha256_file(report/'D10_entropy_OD_uncertainty_verified.json'),
        'A47_deployment_verification_sha256': sha256_file(report/'A47_adaptive_attention_D12_I405_S.deployment.json'),
        'A47_deployable_model_sha256': a['deployable_model_sha256'],
        'formal_best_submission_rehashed': True, 'formal_Kaggle_queried_or_uploaded_this_phase': False,
        'limitations': 'Automated tests and numerical replays do not prove hidden competition performance.'}
    checkpoint = {'created_UTC': stamp, 'git_base_before_phase_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'objective_status': 'active; formal 0.95000+ target remains unachieved',
        'previous_goal_turn_classification': 'no progress: interrupted after read-only inventory; no new training started. This phase took the next safe modeling actions.',
        'current_goal_turn_classification': 'progress',
        'formal_best_ref': 57138874, 'formal_best_Total': .82940,
        'platform_policy': 'Latest human instruction: focus on models, defer publication, GitHub push and formal uploads. Local Git commits only.',
        'completed_this_phase': ['A46 cache and A47 fourth-family State model, calibration-only selection, standalone replay and matched ablation',
            'B07 54 independent CTM configurations and exact separate-process replay, filtered observation inputs and explicit flux/noise scope',
            'D09 retained aggregation failure; D10 sealed empirical curvature calibration and three new mismatch regimes',
            '210 calibration arrays exact; 420 old point scores exact; 600 distinct assessment method cases with no optimizer failure',
            '47 research tests / 42 CPU tests with three optional-Torch skips, documentation and reproducible commands'],
        'terminal_job_records': [{'session_id': sid, 'task': task, 'exit_code': code} for sid, task, code in [
            (18496, 'B07', 0), (95134, 'B07 repeat', 0), (93303, 'D09 failed aggregation', 1),
            (81656, 'A46', 0), (97228, 'A47', 0), (2112, 'D10', 0),
            (12356, 'research tests', 0), (16571, 'A47 export', 0), (48524, 'A47 ablation', 0)]],
        'active_jobs_at_checkpoint': [], 'preserved_complete_submissions': preserved,
        'A47_pooled_State_gain': gain, 'A47_promoted': False,
        'new_simulator_model_promoted': False, 'new_formal_candidate_score': None,
        'next_research_priorities': ['Sparse/zero-path-preserving OD prior and count noise on development distributions, ten-regime gate; no assessment-only floor tuning',
            'State D7_I210 and opposite-direction strong controls; calibration-sealed fallback rules',
            'Latent/noisy CTM targets and missing learned profile ablations; recovery and mild-case degradation before changing dynamics',
            'Queue cross-direction/seed wave controls; previous unselected assessment winner cannot be promoted'],
        'final_simulator_seeds': {'seeds': [7907, 8017, 9109], 'generated': False},
        'limitations': 'Current formal best unchanged. Fourth-family NN gain is below gate. CTM-flux losses and sparse entropy failure contradict global gains; marginal wide intervals do not identify paths.'}
    for path, value in zip(outputs, [manifest, verification, checkpoint, cache]):
        write_new(path, value)
    print('artifacts', len(artifacts), 'previous_unchanged', len(old['artifacts']), 'formal_best_SHA_verified', True, flush=True)


if __name__ == '__main__':
    main()
