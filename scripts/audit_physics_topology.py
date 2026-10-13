"""Audit a structural gate in the public scorer with invented ideal states.

The result is conditional on this public scorer being used unchanged in the
competition; it is not a measured formal score or a probe of hidden data.
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from trafficflow.data import Release


def main():
    source = Path('vendor/official/src/task3/score_task3.py').resolve()
    spec = importlib.util.spec_from_file_location('public_task3_audit', source)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    release = Release('data/kaggle_public')
    cfg = json.loads(Path('vendor/official/config/task3_lwr_modes.json').read_text())
    rows = []
    for panel in release.panels():
        raw = release.read(release.files(panel, 'train', False)[0])
        sensors = set(raw.link_id)
        topo = release.topology(panel)
        topology_ids = set(topo.index)
        observable = []
        for link in sensors:
            if link not in topo.index:
                continue
            neighbors = set()
            for key in ('incoming_link_ids', 'outgoing_link_ids'):
                neighbors.update(x.strip() for x in str(topo.loc[link, key]).split(';') if x.strip())
            if (neighbors & topology_ids) <= sensors:
                observable.append(link)
        net = scorer.network_parameters(release.root / 'corridors' / panel).set_index('link_id').loc[sorted(sensors)]
        pieces = []
        for t in (0, 1):
            q = net.capacity_vph.to_numpy() * (.4 + .005 * t)
            v = net.free_speed_kmh.to_numpy()
            n = q / v * net.length_km.to_numpy()
            dn = .005 * net.capacity_vph.to_numpy() / v * net.length_km.to_numpy()
            pieces.append(pd.DataFrame({'panel': panel, 'timestamp': pd.Timestamp('2040-01-01', tz='UTC') + pd.Timedelta(minutes=5*t),
                'link_id': net.index, 'mask_regime': 'R1', 'speed_kmh': v, 'flow_vph': q,
                'density_vpkm': q/v, 'accumulation_N': n, 'is_target': 1.,
                'inflow_vph': 5000. + dn * 12, 'outflow_vph': 5000.,
                'on_ramp_flow_vph': 0., 'off_ramp_flow_vph': 0., 'on_ramp_valid': 1., 'off_ramp_valid': 1.}))
        measured = scorer.score_panel(panel, pd.concat(pieces), release.root, cfg, 'R1')
        rows.append({'panel': panel, 'sensor_links': len(sensors), 'observable_sensor_links': len(observable),
            'fraction': len(observable)/len(sensors), 'invented_ideal_gate_check': measured})
    zeros = sum(r['observable_sensor_links'] == 0 for r in rows)
    upper = 1 - (zeros / len(rows)) * (2/3) * .15
    result = {'source_commit': 'c88cddf533bbf0afa4ff1fc6c031d760a08e7a31',
        'source': 'public evaluator + public topology + invented two-time ideal state',
        'formal_score': None, 'zero_transition_panels': zeros,
        'conditional_total_upper_bound': upper,
        'qualification': 'Only if the formal evaluator uses this exact structural gate and equal ten-panel Physics averaging. Hidden evaluator behavior has not been verified.',
        'panels': rows}
    output = Path('reports/research/physics_topology_audit.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
