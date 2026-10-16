"""Independent implementation of the vendored c88cddf public score contract.

Raw train observations are evaluator arguments only. The topology fallback is
an executable public diagnostic, never the organizer-held Physics score.
"""
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['timestamp', 'station_id', 'link_id']
VALUES = ['speed_kmh', 'flow_vph']
REGIMES = ('R1', 'R2', 'R3')


def canonical(frame):
    f = frame.copy()
    f['timestamp'] = pd.to_datetime(f.timestamp, utc=True)
    for c in ['station_id', 'link_id']:
        if c in f:
            f[c] = f[c].astype(str)
    return f


def state_targets(raw, masked):
    """Task1 implementation universe; separately compare published template."""
    raw, masked = canonical(raw), canonical(masked)
    blanks = masked.loc[masked[VALUES].isna().all(axis=1), KEY]
    return raw.loc[raw.is_score_eligible.astype(bool)].merge(blanks, on=KEY, validate='one_to_one')


def anchored_frame(raw, template, prediction):
    """Task3 anchors: eligible raw train except published Task1 targets.

    Average station speeds and flows separately, then compute q/v downstream.
    An is_target value is a station fraction, not an any-target flag.
    """
    raw, template, prediction = map(canonical, [raw, template, prediction])
    eligible = raw.loc[raw.is_score_eligible.astype(bool), KEY+VALUES].copy()
    targets = template[KEY].assign(_target=True)
    p = prediction[KEY+VALUES].rename(columns={c: 'pred_'+c for c in VALUES})
    selected = targets.merge(p, on=KEY, how='left', validate='one_to_one')
    full = eligible.merge(selected, on=KEY, how='left', validate='one_to_one')
    flag = full['_target'].fillna(False).astype(bool)
    missing = int(full.loc[flag, ['pred_'+c for c in VALUES]].isna().sum().sum())
    for c in VALUES:
        full.loc[flag, c] = full.loc[flag, 'pred_'+c]
    full['is_target'] = flag.astype(float)
    return full.groupby(['timestamp', 'link_id'], as_index=False)[VALUES+['is_target']].mean(), missing


def network_parameters(panel_dir):
    n = pd.read_csv(Path(panel_dir)/'network/links.csv', dtype={'link_id': str}).drop_duplicates('link_id')
    p = n[['link_id', 'length_km', 'free_speed_kmh', 'capacity_vph', 'lanes']].copy()
    p['length_km'] = pd.to_numeric(p.length_km, errors='coerce').fillna(1.)
    p['free_speed_kmh'] = pd.to_numeric(p.free_speed_kmh, errors='coerce').fillna(100.)
    original_lanes = pd.to_numeric(n.lanes, errors='coerce').fillna(4.)
    p['lanes'] = pd.to_numeric(p.lanes, errors='coerce').fillna(1.).clip(lower=1.)
    p['capacity_vph'] = pd.to_numeric(p.capacity_vph, errors='coerce').fillna(1800*original_lanes)
    fd_path = Path(panel_dir)/'network/fd_parameters.csv'
    if fd_path.exists():
        f = pd.read_csv(fd_path, dtype={'link_id': str})
        cols = [c for c in ['free_speed_kmh', 'capacity_vph', 'lanes', 'k_jam'] if c in f]
        fd = f.groupby('link_id')[cols].mean()
        for c in cols:
            mapped = p.link_id.map(fd[c])
            p[c] = mapped.fillna(p[c]) if c in p else mapped
    p['critical_density'] = p.capacity_vph/p.free_speed_kmh.clip(lower=1.)
    if 'k_jam' not in p:
        p['k_jam'] = np.nan
    p['k_jam'] = np.maximum(p.k_jam.fillna(100*p.lanes.clip(lower=1)), 1.05*p.critical_density)
    return p


def ramp_flows(panel_dir, split='train'):
    root = Path(panel_dir)
    cols = ['timestamp', 'link_id', 'on_ramp_flow_vph', 'off_ramp_flow_vph', 'on_ramp_valid', 'off_ramp_valid']
    attachment = root/'network/ramp_attachment_map.csv'
    if not attachment.exists():
        return pd.DataFrame(columns=cols)
    amap = pd.read_csv(attachment, dtype=str).rename(columns={'nearest_mainline_link_id': 'link_id'})
    pieces = []
    for path in sorted((root/split/'ramp_states').rglob('*.parquet')):
        f = pd.read_parquet(path)
        good = f.is_score_eligible.astype(bool) & (pd.to_numeric(f.pct_observed, errors='coerce') >= 75) & pd.to_numeric(f.flow_vph, errors='coerce').notna()
        f = f.loc[good].copy()
        f['ramp_link_id'] = f.ramp_link_id.astype(str)
        f['timestamp'] = pd.to_datetime(f.timestamp, utc=True)
        f = f.merge(amap[['ramp_link_id', 'link_id']], on='ramp_link_id')
        flow = pd.to_numeric(f.flow_vph)
        for kind, labels in [('on', ['OR', 'ON']), ('off', ['FR', 'OFF'])]:
            flag = f.ramp_type.astype(str).str.upper().isin(labels)
            f[kind+'_ramp_valid'] = flag.astype(int)
            f[kind+'_ramp_flow_vph'] = np.where(flag, flow, 0.)
        pieces.append(f[cols])
    if not pieces:
        return pd.DataFrame(columns=cols)
    return pd.concat(pieces).groupby(['timestamp', 'link_id'], as_index=False).agg(
        on_ramp_flow_vph=('on_ramp_flow_vph', 'sum'), off_ramp_flow_vph=('off_ramp_flow_vph', 'sum'),
        on_ramp_valid=('on_ramp_valid', 'max'), off_ramp_valid=('off_ramp_valid', 'max'))


def topology_flows(state, topology):
    pivot = state.pivot_table(index='timestamp', columns='link_id', values='flow_vph', aggfunc='mean')
    known, rows = set(pivot.columns), []
    for link in state.link_id.drop_duplicates():
        f = pd.DataFrame({'timestamp': pivot.index, 'link_id': link})
        for side, col in [('incoming', 'inflow_vph'), ('outgoing', 'outflow_vph')]:
            ids = str(topology.loc[link, side+'_link_ids']).split(';') if link in topology.index else []
            ids = [i for i in ids if i in known]
            # pandas sum skips absent times and returns zero, exactly as upstream.
            f[col] = (pivot[ids].sum(axis=1) if ids else pivot[link]).to_numpy()
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def physics_frame(state, panel, regime, params, topology, ramps, boundary=None):
    s = state.merge(params, on='link_id', how='left').dropna(subset=VALUES+['length_km', 'capacity_vph', 'critical_density', 'k_jam'])
    s['density_vpkm'] = s.flow_vph/s.speed_kmh.clip(lower=1.)
    s['accumulation_N'] = s.density_vpkm*s.length_km
    if len(ramps):
        s = s.merge(ramps, on=['timestamp', 'link_id'], how='left')
    for kind in ['on', 'off']:
        flow, valid = kind+'_ramp_flow_vph', kind+'_ramp_valid'
        if flow not in s:
            s[flow] = 0.
            s[valid] = 0.
        s[flow] = pd.to_numeric(s[flow], errors='coerce').fillna(0.)
        attached = set(topology.index[topology[kind+'_ramp_link_ids'].fillna('').str.len() > 0]) if len(topology) else set(s.link_id)
        s[valid] = (pd.to_numeric(s[valid], errors='coerce').fillna(0).astype(bool) | ~s.link_id.isin(attached)).astype(int)
    boundary = topology_flows(s, topology) if boundary is None or boundary.empty else canonical(boundary)
    s = s.merge(boundary, on=['timestamp', 'link_id'], how='left')
    for c in ['inflow_vph', 'outflow_vph']:
        s[c] = s[c].fillna(s.flow_vph)
    s['panel'], s['mask_regime'] = panel, regime
    return s


def conservation_pairs(frame, topology, mode='A'):
    """Exactly the scorer's link-set coverage, start-ramp and dt guards."""
    x = frame.copy()
    if mode == 'C' and len(topology):
        attached = topology[['on_ramp_link_ids', 'off_ramp_link_ids']].fillna('').ne('').any(axis=1)
        x = x[~x.link_id.isin(topology.index[attached])]
    if len(topology):
        topo_ids, submitted = set(topology.index), set(x.link_id)
        good = {link: all(({i.strip() for i in str(row[c]).split(';') if i.strip()} & topo_ids) <= submitted
                         for c in ['incoming_link_ids', 'outgoing_link_ids']) for link, row in topology.iterrows()}
        x = x[x.link_id.map(good).fillna(False)].copy()
    x = x.sort_values(['link_id', 'timestamp']).copy()
    x['next_timestamp'] = x.groupby('link_id').timestamp.shift(-1)
    x['dN'] = x.groupby('link_id').accumulation_N.shift(-1)-x.accumulation_N
    x['dt'] = (x.next_timestamp-x.timestamp).dt.total_seconds()/3600
    x = x[(x.dt > 0) & (x.dt <= 5.1/60)]
    if mode in ['A', 'B']:
        x = x[(x.on_ramp_valid >= .5) & (x.off_ramp_valid >= .5)]
    return x.copy()


def physics_score(frame, topology, mode='A'):
    x = frame.dropna(subset=VALUES+['density_vpkm', 'inflow_vph', 'outflow_vph', 'on_ramp_flow_vph', 'off_ramp_flow_vph',
        'on_ramp_valid', 'off_ramp_valid', 'accumulation_N', 'length_km', 'free_speed_kmh', 'capacity_vph', 'critical_density', 'k_jam']).copy()
    if x.empty:
        return {'S_physics': 0., 'n_rows': 0}
    lanes = x.lanes.clip(lower=1)
    k, q, kc, kj, cap = x.density_vpkm/lanes, x.flow_vph/lanes, x.critical_density/lanes, x.k_jam/lanes, x.capacity_vph/lanes
    fd = np.where(k <= kc, x.free_speed_kmh*k, cap/np.maximum(kj-kc, 1e-9)*np.maximum(kj-k, 0))
    s_fd = max(0., 1-float(np.abs(q-fd).sum())/(float(np.abs(q).sum())+1e-9))
    submitted = x.is_target > .5
    if submitted.any() and (x.loc[submitted, 'flow_vph'] < 50).mean() > .2:
        s_fd = 0.
    pairs = conservation_pairs(x, topology, mode)
    rhs = pairs.dt*(pairs.inflow_vph+pairs.on_ramp_flow_vph-pairs.outflow_vph-pairs.off_ramp_flow_vph)
    error = float(np.abs(pairs.dN-rhs).sum()/(np.abs(rhs).sum()+1e-9)) if len(pairs) else 1.
    s_lwr = max(0., 1-min(1., error))
    return {'S_FD': s_fd, 'S_LWR': s_lwr, 'E_LWR': error, 'S_physics': (s_fd+2*s_lwr)/3, 'n_rows': len(pairs)}


def state_statistics(target, prediction, lanes):
    p = target[KEY+VALUES].rename(columns={c: 'true_'+c for c in VALUES}).merge(
        canonical(prediction)[KEY+VALUES], on=KEY, how='left', validate='one_to_one')
    y = p[['true_'+c for c in VALUES]].to_numpy(float)
    a = p[VALUES].to_numpy(float)
    valid = np.isfinite(y).all(axis=1)
    missing = ~np.isfinite(a).all(axis=1)
    a = np.where(np.isfinite(a), a, 0.)
    e = a-y
    return {'n': int(valid.sum()), 'speed_sse': float(np.square(e[valid, 0]).sum()),
        'lane_flow_sse': float(np.square(e[valid, 1]/np.asarray(lanes)[valid]).sum()), 'missing': int(missing[valid].sum())}


def state_macro(rows, families):
    """Regime pooled RMSE -> direction -> family -> overall. Empty groups 0."""
    details = []
    for panel in sorted(families):
        for regime in REGIMES:
            subset = [r for r in rows if r['panel'] == panel and r['regime'] == regime]
            n = sum(r['n'] for r in subset)
            rv = np.sqrt(sum(r['speed_sse'] for r in subset)/n) if n else None
            rq = np.sqrt(sum(r['lane_flow_sse'] for r in subset)/n) if n else None
            score = .54*max(0., 1-rv/25)+.46*max(0., 1-rq/600) if n else 0.
            details.append({'panel': panel, 'regime': regime, 'n': n, 'rmse_speed': rv, 'rmse_flow_lane': rq, 'S_state': score})
    panel_scores = {p: float(np.mean([r['S_state'] for r in details if r['panel'] == p])) for p in families}
    family_scores = {f: float(np.mean([s for p, s in panel_scores.items() if families[p] == f])) for f in set(families.values())}
    return {'S_state': float(np.mean(list(family_scores.values()))), 'details': details, 'panels': panel_scores, 'families': family_scores}
