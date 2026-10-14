"""Archive observed full-domain Queue experiments without overwriting evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from trafficflow.runtime import sha256_file


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked(row):
    path = Path(row['path'])
    if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
        raise ValueError('Previous artifact changed: ' + str(path))
    return dict(row)


def main():
    reports = Path('reports/research')
    names = ['model_artifacts_checkpoint_full_queue_v1.json',
             'full_queue_test_record_v1.json', 'research_checkpoint_full_queue_v1.json',
             'K06_I5_N_full_domain_Queue_LOCAL.controlled.json']
    if any((reports / name).exists() for name in names):
        raise FileExistsError('Never overwrite a checkpoint')
    parent_path = reports / 'model_artifacts_checkpoint_full_density_v1.json'
    parent = read(parent_path)
    artifacts = [checked(row) for row in parent['artifacts']]
    folders = ['C38_full_queue_probability_cache', 'C39_full_domain_Queue_calibration',
               'C39_full_domain_Queue_calibration.deployment',
               'C40_synthetic_Queue_selection_bias', 'K06_I5_N_full_domain_Queue_LOCAL']
    for name in folders:
        folder = Path('artifacts/research') / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.pkl', '.pt', '.npz', '.zip', '.csv'}:
                artifacts.append({'path': path.as_posix(), 'bytes': path.stat().st_size,
                                  'sha256': sha256_file(path), 'phase': 'full_queue_calibration'})
    if len({row['path'] for row in artifacts}) != len(artifacts):
        raise ValueError('Duplicate artifacts')
    preserved = [checked(row) for row in parent['preserved_complete_submissions']]
    if {name: sha256_file(name) for name in parent['dependency_locks']} != parent['dependency_locks']:
        raise ValueError('Dependency locks changed')
    cache = read(reports / 'C38_full_queue_probability_cache.json')
    if len(cache['full_probability_caches']) != 8 or not all(
        row['legacy_replay_exact'] and row['fresh_causal_inputs_exact'] for row in cache['rows']
    ):
        raise ValueError('Incomplete fixed-weight probability replay')
    for row in cache['full_probability_caches']:
        if sha256_file(row['path']) != row['sha256']:
            raise ValueError('Probability cache changed')
    producer = Path('artifacts/research/C39_full_domain_Queue_calibration/experiment.json')
    run = read(producer)
    replay_path = reports / 'C39_full_domain_Queue_calibration.deployment.json'
    replay = read(replay_path)
    comparison = read(reports / 'C41_full_domain_Queue_model_comparison.json')
    if replay['source_experiment_sha256'] != sha256_file(producer):
        raise ValueError('Deployment source changed')
    if comparison['source_experiment_sha256'] != sha256_file(producer) or comparison[
        'source_probability_report_sha256'] != sha256_file(reports / 'C38_full_queue_probability_cache.json'):
        raise ValueError('Comparison source changed')
    if len(replay['models']) != 4 or sum(row['n_cal_windows'] for row in replay['models']) != 220:
        raise ValueError('Incomplete deployment verification')
    if not all(row['all_full_template_predictions_exact'] and row[
        'history_rebuilt_from_public_le_T'] for row in replay['models']):
        raise ValueError('Full-template causal deployment replay failed')
    verified_models = {row['model_sha256'] for row in replay['models']}
    for study in run['studies']:
        for model in study['models'].values():
            if sha256_file(model['path']) != model['sha256'] or model['sha256'] not in verified_models:
                raise ValueError('Unverified exported model')
    candidate_path = Path('artifacts/research/K06_I5_N_full_domain_Queue_LOCAL/submission.csv')
    diff_path = candidate_path.parent / 'controlled_diff.json'
    diff = read(diff_path)
    if not diff['full_row_diff_passed'] or diff['rows'] != 6980503 or diff[
        'changed_rows'] != {'queue': 415} or diff['allowed_panel'] != 'D12_I5_N':
        raise ValueError('Controlled full-row diff failed')
    candidate = {'path': candidate_path.as_posix(), 'bytes': candidate_path.stat().st_size,
                 'sha256': sha256_file(candidate_path)}
    if candidate['sha256'] != diff['candidate_sha256'] or diff['baseline_sha256'] != parent[
        'formal_best_submission_sha256']:
        raise ValueError('Submission checksum changed')
    preserved.append(candidate)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {'created_UTC': now, 'parent_manifest_sha256': sha256_file(parent_path),
                'n_previous_artifacts_rehashed_unchanged': len(parent['artifacts']),
                'n_artifacts': len(artifacts), 'artifacts': artifacts,
                'dependency_locks': parent['dependency_locks'],
                'preserved_complete_submissions': preserved, 'formal_best_Total': .82940,
                'formal_best_submission_sha256': parent['formal_best_submission_sha256'],
                'purpose': 'Model/cache/config/source and complete LOCAL candidate integrity; hashes only',
                'limitations': 'No model promotion, new fit seed or new formal score. K06 is an unuploaded controlled candidate.'}
    tests = {'created_UTC': now,
             'research_pytest': {'passed': 67, 'seconds': 14.06, 'session_id': 41531,
                 'basetemp': 'E:/ieee/.cache/pytest_full_queue_deployment_phase_research'},
             'CPU_pytest': {'passed': 59, 'skipped_optional_torch_modules': 3, 'seconds': 7.84,
                 'basetemp': 'E:/ieee/.cache/pytest_full_queue_deployment_phase_cpu'},
             'compileall_passed': True, 'full_probability_caches_verified': 8,
             'independent_deployment_report_sha256': sha256_file(replay_path),
             'independent_full_template_cal_window_calls_exact': 220,
             'complete_candidate_full_row_diff_sha256': sha256_file(diff_path),
             'complete_candidate_rows': diff['rows'], 'changed_queue_values': 415,
             'formal_baseline_rehashed': True, 'formal_Kaggle_actions_this_phase': False,
             'scope': 'Observed tests, causal deployment replay and complete-file diff; no hidden-score evidence.'}
    checkpoint = {'created_UTC': now,
        'git_base_before_phase_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'objective_status': 'active; formal 0.95000+ unachieved',
        'previous_goal_turn_classification': 'progress: b245741, density controls and Queue decision domain audit',
        'current_goal_turn_classification': 'progress', 'formal_best_Total': .82940,
        'platform_policy': 'User directs local model research; no publication, push or formal upload',
        'completed': ['C38 eight complete-domain probabilities from frozen causal inputs/weights',
            'C39 calibration sealed before assessment; public timing before/after control has zero independent gain',
            'Four portable full-template models,220 independent cal-window exact replays',
            'C40 135000 independently generated known-truth events; selection-bias diagnostic only',
            'C41 fair full-domain tree comparison, both seasonal gain intervals include zero',
            'K06 complete Queue-only LOCAL candidate;415 values changed, all other fields identical',
            '67 research/59 CPU tests,3 optional Torch skips; model/artifact/report integrity'],
        'terminal_jobs': [{'session_id': sid, 'task': task, 'exit_code': 0} for sid, task in [
            (49579, 'C38'), (17820, 'C39'), (91864, 'intermediate66 research tests'),
            (41531, 'final67 research tests'), (88303, 'C39 independent deployment'),
            (21913, 'validation prediction'), (99872, 'private prediction'),
            (62195, 'K06 complete build'), (23959, 'K06 full-row diff')]],
        'active_jobs_at_checkpoint': [], 'new_model_promoted': False,
        'new_formal_score': None, 'formal_best_preserved': True,
        'final_simulator_seeds': {'seeds': [7907, 8017, 9109], 'generated': False},
        'next_research_priorities': ['Lawful TRAIN event sampling and observation-vs-latent selection bias',
            'Cross-direction Queue and new fit seeds before generalization claims',
            'Event-level joint probability/dependence, not another same-grid threshold search',
            'State observation-noise dynamics and OD prior/count mismatch with retained negative evidence'],
        'limitations': 'Old public TRAIN development periods and one fit seed. Synthetic law is not organizer law. True Queue/Physics/OD/Total unknown.'}
    for name, value in zip(names, [manifest, tests, checkpoint, diff]):
        with (reports / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    print('previous_unchanged', len(parent['artifacts']), 'total_artifacts', len(artifacts),
          'deployment_calls_exact', tests['independent_full_template_cal_window_calls_exact'], flush=True)


if __name__ == '__main__':
    main()
