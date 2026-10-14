"""Instrument one preserved D06 failed solve without changing its algorithm."""
import inspect
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from trafficflow.data import Release
from trafficflow.odme import operator
from trafficflow.entropy_od import entropy_inverse
from trafficflow.runtime import sha256_file
from entropy_od_experiment import generated, REGIMES


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=Path('reports/research/D06_entropy_OD_broad_prior.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/research/D06_entropy_failed_case_trace.json'))
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError('Never overwrite a failed-case trace')
    original = json.loads(args.study.read_text())
    failed = next(r for r in original['development'] if not r['success'])
    settings = original['registered']['protocol']['grid'][int(failed['method'].rsplit('_', 1)[1])]
    A, _, b, paths = operator(Release('data/kaggle_public'), failed['panel'], 'train')
    _, c, prior = generated(A, b, paths, failed['seed']+100*REGIMES.index(failed['regime']), failed['regime'])
    lines, start = inspect.getsourcelines(entropy_inverse)
    trace_line = start+next(i for i, line in enumerate(lines) if line.strip() == 'current = evaluate(dual)')
    records = []
    def trace(frame, event, arg):
        if frame.f_code is entropy_inverse.__code__ and event == 'line' and frame.f_lineno == trace_line:
            local = frame.f_locals
            exponent = -np.asarray(A.T@local['dual'])
            records.append({'iteration': local['iteration'], 'min_log_flow_ratio': float(exponent.min()),
                'max_log_flow_ratio': float(exponent.max()), 'previous_primal_stationarity': local.get('certificate'),
                'previous_objective': local.get('value'), 'previous_line_length': local.get('length'),
                'previous_dual_gradient_max': float(np.max(np.abs(local['gradient']))) if 'gradient' in local else None,
                'previous_Newton_step_max': float(np.max(np.abs(local['step']))) if 'step' in local else None})
        return trace
    old_trace = sys.gettrace()
    error = None
    sys.settrace(trace)
    try:
        entropy_inverse(A, c, prior, **settings)
    except RuntimeError as failure:
        error = str(failure)
    finally:
        sys.settrace(old_trace)
    result = {'original_failed_case': failed, 'settings': settings, 'trace': records,
        'reproduced_error': error, 'solver_source_sha256': sha256_file(Path(inspect.getsourcefile(entropy_inverse))),
        'scope': 'Exact known-truth generator/settings from a preserved failed development case; no model retuning or hidden data'}
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(error, records[:3], records[-4:], flush=True)
