"""Public-release reader; raw train labels are available only to training code."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
STATE_KEY = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
QUEUE_KEY = ['window_id', 'timestamp', 'link_id']
OD_KEY = ['panel', 'departure_time', 'path_id', 'origin_zone', 'destination_zone']
VALUES = ['speed_kmh', 'flow_vph']
SPLITS = ('train', 'validation', 'private')

def csv(path: Path) -> pd.DataFrame:
    # IDs and timestamp spellings are preserved exactly, including leading zeros.
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def eligible(frame: pd.DataFrame) -> np.ndarray:
    good = pd.to_numeric(frame.pct_observed, errors='coerce').ge(75).to_numpy(copy=True)
    for c in VALUES:
        good &= np.isfinite(pd.to_numeric(frame[c], errors='coerce').to_numpy(float))
    return good

def check_keys(frame: pd.DataFrame, keys: list[str], label: str):
    missing = set(keys) - set(frame.columns)
    if missing:
        raise ValueError(f'{label}: missing columns {sorted(missing)}')
    if frame[keys].isna().any().any() or frame[keys].astype(str).eq('').any().any():
        raise ValueError(f'{label}: blank key')
    if frame.duplicated(keys).any():
        raise ValueError(f'{label}: duplicate keys')
    if 'timestamp' in keys:
        if not frame.timestamp.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$').all():
            raise ValueError(f'{label}: timestamps must keep released UTC text')
        pd.to_datetime(frame.timestamp, utc=True, errors='raise')

class Release:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not (self.root / 'corridors').is_dir():
            raise FileNotFoundError(f'{self.root} is not a release root (expected corridors/). Download current official kaggle_public.')

    def panels(self) -> list[str]:
        return sorted(p.name for p in (self.root / 'corridors').iterdir() if (p / 'network').is_dir())

    def files(self, panel: str, split: str, masked: bool = True) -> list[Path]:
        if split not in SPLITS or panel not in self.panels():
            raise ValueError('Unknown split or panel')
        if not masked and split != 'train':
            raise ValueError('Only public train has raw observations; hidden-label access is prohibited')
        layer = 'mainline_states_masked' if masked else 'mainline_states'
        return sorted((self.root / 'corridors' / panel / split / layer).rglob('*.parquet'))

    def read(self, path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        for c in ('timestamp', 'date', 'station_id', 'link_id', 'mask_regime'):
            if c in frame:
                frame[c] = frame[c].astype(str)
        for c in VALUES + ['pct_observed', 'milepost']:
            if c in frame:
                frame[c] = pd.to_numeric(frame[c], errors='coerce')
        check_keys(frame, ['timestamp', 'station_id', 'link_id'], str(path))
        return frame

    def template(self, task: int, panel: str, split: str) -> pd.DataFrame:
        name = {1:'state', 2:'queue', 4:'path_flow'}[task]
        f = csv(self.root / f'task{task}' / panel / split / f'sample_submission_{name}.csv')
        if task == 2:
            index = csv(self.root / 'task2' / panel / split / 'window_index.csv')
            # Some queue templates enumerate more than one split for convenience.
            f = f[f.window_id.isin(index.window_id)].copy()
        key = {1:STATE_KEY, 2:QUEUE_KEY, 4:OD_KEY}[task]
        check_keys(f, key, f'task{task} template')
        return f

    def network(self, panel: str) -> pd.DataFrame:
        folder = self.root / 'corridors' / panel / 'network'
        links = csv(folder / 'links.csv')
        links = links.drop_duplicates('link_id').set_index('link_id')
        required = ['length_km', 'lanes', 'free_speed_kmh', 'capacity_vph']
        if not set(required).issubset(links):
            raise ValueError('Network geometry/FD fields missing')
        for c in required:
            links[c] = pd.to_numeric(links[c], errors='raise')
        fd_path = folder / 'fd_parameters.csv'
        if fd_path.exists():
            fd = csv(fd_path)
            for c in ['lanes', 'free_speed_kmh', 'capacity_vph', 'k_jam', 'v_cut']:
                if c in fd:
                    values = pd.to_numeric(fd[c], errors='coerce').groupby(fd.link_id).mean()
                    links[c] = values.reindex(links.index).fillna(links[c] if c in links else np.nan)
        links['critical_density'] = links.capacity_vph / links.free_speed_kmh
        if 'k_jam' not in links:
            links['k_jam'] = 100 * links.lanes
        if 'v_cut' not in links:
            links['v_cut'] = .60 * links.free_speed_kmh
        if (links[required] <= 0).any().any():
            raise ValueError('Non-positive lanes or geometry/FD parameters')
        return links

    def station_lanes(self, panel: str, frame: pd.DataFrame) -> np.ndarray:
        net = self.network(panel)
        lane = frame.link_id.map(net.lanes)
        fd = csv(self.root / 'corridors' / panel / 'network' / 'fd_parameters.csv')
        if 'station_id' in fd and 'lanes' in fd:
            st = pd.to_numeric(fd.lanes, errors='coerce').groupby(fd.station_id).mean()
            lane = frame.station_id.map(st).fillna(lane)
        if lane.isna().any() or (lane <= 0).any():
            raise ValueError('Target station/link has no valid lane count')
        return lane.to_numpy(float)

    def topology(self, panel: str) -> pd.DataFrame:
        p = self.root / 'corridors' / panel / 'network' / 'lwr_mainline_topology.csv'
        return csv(p).set_index('link_id') if p.exists() else pd.DataFrame()

    def ramps(self, panel: str, split: str, timestamps: list[str], links: list[str]) -> tuple[np.ndarray, np.ndarray]:
        # Known structural zeros and missing attached ramps are distinct.
        shape = (len(timestamps), len(links))
        on, off = np.zeros(shape), np.zeros(shape)
        topo = self.topology(panel)
        folder = self.root / 'corridors' / panel
        attached = {'on':{}, 'off':{}}
        for j, link in enumerate(links):
            for kind in attached:
                field = f'{kind}_ramp_link_ids'
                ids = str(topo.loc[link, field]).split(';') if link in topo.index and field in topo else []
                ids = [i for i in ids if i and i != 'nan']
                attached[kind][link] = ids
                if ids:
                    (on if kind == 'on' else off)[:, j] = np.nan
        pieces = []
        wanted_days = {t[:10] for t in timestamps}
        for p in sorted((folder / split / 'ramp_states').rglob('*.parquet')):
            # Read only matching dates if filename carries a date; fallback to filtering.
            match = re.search(r'(20\d\d)[_-](\d\d)[_-](\d\d)', p.name)
            if match and '-'.join(match.groups()) not in wanted_days:
                continue
            d = pd.read_parquet(p)
            d = d[d.timestamp.astype(str).str[:10].isin(wanted_days)].copy()
            if len(d):
                pieces.append(d)
        if not pieces:
            return on, off
        d = pd.concat(pieces, ignore_index=True)
        d['flow_vph'] = pd.to_numeric(d.flow_vph, errors='coerce')
        d.loc[pd.to_numeric(d.pct_observed, errors='coerce') < 75, 'flow_vph'] = np.nan
        pivot = d.pivot_table(index='timestamp', columns='ramp_link_id', values='flow_vph', aggfunc='mean').reindex(timestamps)
        for j, link in enumerate(links):
            for kind, out in [('on',on), ('off',off)]:
                ids = attached[kind][link]
                if ids and set(ids).issubset(pivot.columns):
                    out[:,j] = pivot[ids].sum(axis=1, min_count=len(ids)).to_numpy()
        return on, off

    def inventory(self, output: Path, full_hash: bool = False) -> dict:
        rows = []
        for p in sorted(self.root.rglob('*')):
            if not p.is_file():
                continue
            entry = {'path':p.relative_to(self.root).as_posix(), 'bytes':p.stat().st_size}
            if full_hash:
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda:f.read(4*1024**2), b''):
                        h.update(block)
                entry['sha256'] = h.hexdigest()
            rows.append(entry)
        result = {'release_root':str(self.root), 'panels':self.panels(), 'files':rows,
                  'content_hashes':full_hash, 'synthetic_fixture':(self.root/'FIXTURE_ONLY.json').exists()}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
