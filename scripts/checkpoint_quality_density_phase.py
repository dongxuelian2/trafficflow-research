"""Archive completed local studies and preserve exact previous artifact hashes.

No network, model fitting, submission, publication, or goal-status changes.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from trafficflow.runtime import sha256_file


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    reports = Path('reports/research')
    names = ['model_artifacts_checkpoint_quality_density_v1.json',
             'quality_density_test_record_v1.json',
             'quality_density_metadata_corrections_v1.json',
             'research_checkpoint_quality_density_v1.json']
    if any((reports / n).exists() for n in names):
        raise FileExistsError('Never overwrite checkpoint evidence')
    parent_path = reports / 'model_artifacts_checkpoint_sparse_profile_v1.json'
    parent = json.loads(parent_path.read_text())
    artifacts = []
    for row in parent['artifacts']:
        path = Path(row['path'])
        if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise ValueError('Previous artifact changed: ' + str(path))
        artifacts.append(dict(row))
    folders = ['A51_quality_D7_I210_W_autumn_cache', 'A52_quality_density_D7_I210_W',
               'A54_density_coordinates_D7_I210_W',
               'A55_observed_increment_coverage_diagnostic',
               'A55_observed_increment_coverage_v2',
               'A57_frozen_State_temporal_extension']
    for name in folders:
        folder = Path('artifacts/research') / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.pkl', '.pt', '.npz', '.zip'}:
                artifacts.append({'path': path.as_posix(), 'bytes': path.stat().st_size,
                                  'sha256': sha256_file(path), 'phase': 'quality_density'})
    if len({r['path'] for r in artifacts}) != len(artifacts):
        raise ValueError('Duplicate manifest artifacts')
    formal_digest = parent['formal_best_submission_sha256']
    preserved = []
    for name, digest in [
        ('artifacts/frozen_baselines/57138874/submission.csv', formal_digest),
        ('artifacts/submission_best/submission.csv', formal_digest),
        ('artifacts/research/K05_OD_expert_LOCAL/submission.csv',
         parent['complete_LOCAL_candidate']['submission_sha256'])]:
        path = Path(name)
        if sha256_file(path) != digest:
            raise ValueError('Preserved complete submission changed: ' + name)
        preserved.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest})
    locks = {n: sha256_file(n) for n in parent['dependency_locks']}
    if locks != parent['dependency_locks']:
        raise ValueError('Dependency locks changed')
    control = json.loads((reports/'A52_exact_training_control.json').read_text())
    if not control['four_model_patch_sequence_identical'] or not all(
            r['all_parameter_tensors_exact'] for r in control['controls']):
        raise ValueError('Matched optimization control not verified')
    verification = {}
    for study in ['A52_quality_density_D7_I210_W', 'A54_density_coordinates_D7_I210_W']:
        path = reports/(study+'.deployment.json')
        d = json.loads(path.read_text())
        source = reports/(study+'.json')
        if d['source_report_sha256'] != sha256_file(source):
            raise ValueError('Deployment source report changed')
        for role, row in d['roles'].items():
            if row['n_calibration_days'] != 14 or not row['independent_masked_frame_inputs_exact']:
                raise ValueError('Independent deployment replay incomplete')
            model = Path('artifacts/research')/study/('State_'+role+'.pkl')
            if sha256_file(model) != row['model_sha256']:
                raise ValueError('Deployed weights changed')
        verification[path.name] = sha256_file(path)
    extension = json.loads((reports/'A57_frozen_State_temporal_extension.json').read_text())
    if sum(extension['regime_day_counts'].values()) != 30:
        raise ValueError('Incomplete additional time replication')
    now = datetime.now(timezone.utc).isoformat()
    manifest = {'created_UTC': now, 'purpose': 'Artifact integrity, hashes only; no data or model redistribution',
        'parent_manifest_sha256': sha256_file(parent_path),
        'n_previous_artifacts_rehashed_unchanged': len(parent['artifacts']),
        'formal_best_Total': .82940, 'formal_best_submission_sha256': formal_digest,
        'artifacts': artifacts, 'n_artifacts': len(artifacts),
        'dependency_locks': locks, 'preserved_complete_submissions': preserved,
        'limitations': 'Completed artifacts do not establish formal improvement. Final simulator seeds unopened.'}
    test_record = {'created_UTC': now,
        'research_pytest': {'passed': 53, 'seconds': 13.42,
            'basetemp': 'E:/ieee/.cache/pytest_quality_coordinates_research'},
        'CPU_pytest': {'passed': 46, 'skipped_optional_torch_modules': 3, 'seconds': 8.31,
            'basetemp': 'E:/ieee/.cache/pytest_quality_coordinates_cpu'},
        'compileall': 'src/scripts/tests passed; checkpoint writer compiled after creation',
        'independent_deployment_reports': verification,
        'exact_control_sha256': sha256_file(reports/'A52_exact_training_control.json'),
        'legacy_compatibility_sha256': sha256_file(reports/'A53_quality_context_legacy_audit.json'),
        'formal_baseline_rehashed': True, 'formal_Kaggle_actions_this_phase': False,
        'scope': 'Observed terminal tests, compatibility/input reconstruction and matched optimization controls; no hidden score'}
    metadata = {'created_UTC': now, 'preserved_reports': {
        n: sha256_file(reports/n) for n in ['A55_observed_increment_coverage_diagnostic.json',
        'A55_observed_increment_coverage_v2.json', 'A57_frozen_State_temporal_extension.json']},
        'correction': 'These original evidence() calls left gpu_used at default false. This field does not establish CPU-only execution.',
        'device_inference': 'Executed AdaptiveStateModel source selects CUDA when available; the same research environment had CUDA available. GPU inference is inferred, not separately logged by these runs.',
        'original_GPU_peak_MiB': None,
        'original_scores_and_source_snapshots_preserved': True,
        'future_fix': 'Diagnostic and temporal runner now inspect actual cached network parameter device for evidence(). No metric changes.',
        'joint_selection_text': 'Current source now states optional strict12 calibration reference. A54 executed source already used it; current primary exceeds control so original wording omission does not change this result.'}
    checkpoint = {'created_UTC': now,
        'git_base_before_phase_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'objective_status': 'Formal 0.95000+ unachieved; completed local modeling checkpoint',
        'app_goal_status_observed_at_archive': 'blocked',
        'goal_status_scope': 'Reported by get_goal; this script does not resume, pause, block or complete goals. Latest human direction authorizes local models and defers platform work.',
        'current_phase_classification': 'progress', 'formal_best_Total': .82940,
        'platform_policy': 'Local models and Git commits only; no forum/Notebook, push or formal upload',
        'completed': ['A51 legal quality18 cache', 'A52 four matched vq/quality/density-loss models and exact A49 control',
            'A53 masked visibility and old-pickle fallback replay', 'A54 two matched density-coordinate models',
            'A55 coverage-aligned public-train increment proxy, both original and bounded versions preserved',
            'A56 matched State/proxy paired-day intervals', 'A57 frozen 30-day time extension, no re-selection',
            '53 research tests /46 CPU tests, three optional Torch skips, all deployment roles replayed'],
        'terminal_jobs': [{'session_id': sid, 'task': task, 'exit_code': 0} for sid, task in [
            (48835, 'A51'), (64753, 'A52'), (53778, 'A53'), (22272, 'A54'),
            (32942, 'A52 deployment'), (71695, 'A54 deployment'), (55334, 'A55 v1'),
            (47409, 'A55 v2'), (31936, 'research pytest'), (52032, 'A57')]],
        'active_jobs_at_checkpoint': [], 'new_State_model_promoted': False,
        'new_formal_score': None, 'best_formal_preserved': True,
        'final_simulator_seeds': {'seeds': [7907, 8017, 9109], 'generated': False},
        'next_research_priorities': ['Explicit observation uncertainty gating with strict12 control',
            'Cross-season density/state assimilation and latent boundary/noise modeling',
            'Causal Queue cross-direction controls and robust OD count/prior shifts'],
        'limitations': 'Quality and density State effects are small/unstable. Coverage proxy is not formal Physics. One direction and fit seed; 30-day extension is public development.'}
    for name, value in zip(names, [manifest, test_record, metadata, checkpoint]):
        write_new(reports/name, value)
    print('previous_unchanged', len(parent['artifacts']), 'total_artifacts', len(artifacts),
          'formal_best_preserved', True, flush=True)


if __name__ == '__main__':
    main()
