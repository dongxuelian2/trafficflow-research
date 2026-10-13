"""Known-truth independent OD experiments; never reads public_reference files."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from trafficflow.data import Release
from trafficflow.odme import operator, solve
from trafficflow.bayesian_od import posterior
from trafficflow.metrics import od_link_score
from trafficflow.runtime import evidence


def generate(A, b, paths, seed, regime):
    rng = np.random.default_rng(seed)
    f = np.maximum(b, 2.).copy()
    if regime == 'independent_paths':
        f *= rng.lognormal(-.5 * .25**2, .25, len(b))
    elif regime == 'hierarchical_demand':
        for key in ('origin_zone', 'destination_zone'):
            labels = paths[key].to_numpy(str)
            factors = {k: rng.lognormal(-.5 * .15**2, .15) for k in sorted(set(labels))}
            f *= np.array([factors[k] for k in labels])
        f *= rng.lognormal(-.5 * .12**2, .12, len(b))
    elif regime == 'prior_total_bias':
        f *= rng.uniform(.75, 1.3)
        f *= rng.lognormal(-.5 * .20**2, .20, len(b))
    else:
        raise ValueError(regime)
    clean = A @ f
    c = np.maximum(0, clean + rng.normal(0, .02 * np.maximum(clean, 10.)))
    return f, c


def scores(A, c, b, truth, prediction, paths):
    sod = max(0., 1 - np.abs(prediction - truth).sum() / truth.sum())
    slink = od_link_score(A, prediction, c)
    dtrue = np.abs(truth - b).sum()
    dev = np.exp(-abs(np.abs(prediction - b).sum() / max(dtrue, 1e-9) - 1))
    labels = paths.destination_zone.to_numpy(str)
    truth_attr = np.array([truth[labels == key].sum() for key in sorted(set(labels))]) / truth.sum()
    pred_attr = np.array([prediction[labels == key].sum() for key in sorted(set(labels))]) / max(prediction.sum(), 1e-9)
    attr = max(0., 1 - .5 * np.abs(truth_attr - pred_attr).sum())
    return {'simulated_S_od': float(sod), 'simulated_S_link': float(slink),
            'simulated_S_dev': float(dev), 'simulated_S_attr': float(attr),
            'simulated_S_ODME': float(.45*sod + .25*slink + .15*dev + .15*attr),
            'relative_path_L1': float(np.abs(prediction-truth).sum()/truth.sum()),
            'formal_ODME': None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', action='append')
    parser.add_argument('--all-panels', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new experiment output; never overwrite evidence')
    started = time.perf_counter()
    release = Release('data/kaggle_public')
    protocol = json.loads(Path('reports/research/protocol_v2_registered.json').read_text())
    grid = [{'path_cv': p, 'group_cv': g, 'noise_cv': .02} for p, g in
            [(.12, 0.), (.2, 0.), (.2, .08), (.2, .2)]]
    results = {'experiment': 'D01_probabilistic_OD', 'protocol_sha256': protocol['protocol_sha256'],
        'truth': 'independently generated lognormal path/group perturbations; public network and weak prior only',
        'formal_ODME': None, 'panels': {}, 'grid': grid, 'source_at_start': evidence()}
    regimes = ['independent_paths', 'hierarchical_demand', 'prior_total_bias']
    for panel in (release.panels() if args.all_panels else args.panel or ['D12_I5_N', 'D7_I210_W']):
        A, _, b, paths = operator(release, panel, 'train')
        development = []
        for seed in protocol['protocol']['simulator_development_seeds']:
            for ri, regime in enumerate(regimes):
                truth, c = generate(A, b, paths, seed + 100 * ri, regime)
                for gi, settings in enumerate(grid):
                    tic = time.perf_counter()
                    try:
                        f, sd, info = posterior(A, c, b, paths, **settings)
                        row = scores(A, c, b, truth, f, paths)
                        row.update(info)
                    except RuntimeError as error:
                        row = {'error': str(error), 'simulated_S_ODME': None}
                    development.append({'seed': seed, 'regime': regime, 'grid': gi,
                                        'seconds': time.perf_counter()-tic, **row})
        means = [np.mean([r['simulated_S_ODME'] for r in development if r['grid'] == gi
                         and r['simulated_S_ODME'] is not None]) for gi in range(len(grid))]
        best = int(np.argmax(means))
        assessment = []
        for seed in protocol['protocol']['simulator_assessment_seeds']:
            for ri, regime in enumerate(regimes):
                truth, c = generate(A, b, paths, seed + 100 * ri, regime)
                for method in ('prior', 'ridge_0.05', 'selected_bayesian'):
                    tic = time.perf_counter()
                    if method == 'prior':
                        f, sd, info = b, None, {}
                    elif method == 'ridge_0.05':
                        f, info = solve(A, c, b, 'ridge', .05)
                        sd = None
                    else:
                        f, sd, info = posterior(A, c, b, paths, **grid[best])
                    row = scores(A, c, b, truth, f, paths)
                    if sd is not None:
                        row['approximate_95pct_path_interval_coverage'] = float((np.abs(f-truth) <= 1.96*sd).mean())
                    assessment.append({'seed': seed, 'regime': regime, 'method': method,
                                       'seconds': time.perf_counter()-tic, **row, **info})
        public, _, public_info = posterior(A, operator(release, panel, 'train')[1], b, paths, **grid[best])
        results['panels'][panel] = {'matrix_shape': list(A.shape), 'selected_grid': best,
            'development_mean_by_grid': means, 'development': development, 'assessment': assessment,
            'public_train_stability': {'S_link_only': od_link_score(A, public, operator(release, panel, 'train')[1]),
            'prior_relative_L1_change': float(np.abs(public-b).sum()/max(b.sum(), 1)), **public_info}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(panel, {method: np.mean([r['simulated_S_ODME'] for r in assessment if r['method']==method])
                      for method in ('prior', 'ridge_0.05', 'selected_bayesian')}, flush=True)
    results['seconds'] = time.perf_counter()-started
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
