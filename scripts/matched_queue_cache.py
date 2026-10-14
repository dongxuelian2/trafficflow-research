"""Read-only causal sequence cache paired with a preserved public proxy study."""
import json
from pathlib import Path
import numpy as np
from trafficflow.data import Release
from trafficflow.experiments import partition_date, raw_for
from trafficflow.queue import QueueModel
from trafficflow.queue_events import sample_events
from trafficflow.causal_queue import released_origin_rows
from trafficflow.research_eval import canonical_hash
from trafficflow.runtime import sha256_file


class MatchedQueueCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.report = json.loads((self.directory/'experiment.json').read_text())
        self.config = self.report['registered']['protocol']
        self.release = Release('data/kaggle_public')
        self.panel = self.config['panel']
        self.net, self.topo = self.release.network(self.panel), self.release.topology(self.panel)
        self.baseline_path = Path('artifacts/research')/Path(self.config['matched_study']).stem/'shared_threshold.pkl'
        if sha256_file(self.baseline_path) != self.config['baseline_sha256']:
            raise ValueError('Matched tree model changed')
        self.baseline = QueueModel.load(self.baseline_path)
        paths = self.release.files(self.panel, 'train')
        self.parts = {key: [p for p in paths if partition_date(self.release, p) in dates] for key, dates in self.report['dates'].items()}
        self.origin_cache = {}
        raw = next(raw_for(self.release, self.panel, self.parts['fit'][:1]))
        self.links = raw.link_id.drop_duplicates().tolist()

    def origin(self, origin):
        token = str(origin)
        if token not in self.origin_cache:
            self.origin_cache[token] = released_origin_rows(self.release, self.panel, 'train', origin)
        return self.origin_cache[token]

    def fit(self):
        path = self.directory/('fit_'+str(self.config['seed'])+'.npz')
        if sha256_file(path) != self.report['cache_sha256']['fit']:
            raise ValueError('Fit sequence cache changed')
        arrays = dict(np.load(path, allow_pickle=False))
        self.validate_shape(arrays)
        return arrays

    def validate_shape(self, arrays):
        if arrays['x'].shape[1:] != (13, len(self.links), 9) or not np.isfinite(arrays['x']).all():
            raise ValueError('Causal sequence node shape or finite-input contract changed')

    def sample(self, part, seed):
        if part == 'selection':
            expected_meta = self.report['samplers']['selection']
            expected_hash = self.report['cache_sha256']['selection']
        elif part == 'assessment':
            previous = next(r for r in self.report['assessment'] if r['seed'] == seed)
            expected_meta, expected_hash = previous['sampler'], previous['cache_sha256']
        else:
            raise ValueError('Only selection or assessment proxies may be sampled here')
        frames = list(raw_for(self.release, self.panel, self.parts[part]))
        if any(frame.link_id.drop_duplicates().tolist() != self.links for frame in frames):
            raise ValueError('Public sequence node order changed')
        data, metadata = sample_events(frames, self.net, self.topo, self.panel, seed,
            per_condition=30, spacing_minutes=360, origin_reader=self.origin, onset_last_step=True)
        if canonical_hash(metadata) != canonical_hash(expected_meta):
            raise ValueError('Matched proxy events differ')
        path = self.directory/(part+'_'+str(seed)+'.npz')
        if sha256_file(path) != expected_hash:
            raise ValueError('Sequence cache changed')
        arrays = dict(np.load(path, allow_pickle=False))
        self.validate_shape(arrays)
        origins = [r['origin'] for r in sorted(metadata['windows'], key=lambda r: r['origin'])]
        if arrays['origins'].tolist() != origins:
            raise ValueError('Matched origin order changed')
        y = np.concatenate([label.ravel()[valid.ravel()] for label, valid in zip(arrays['labels'], arrays['available'])])
        if not np.array_equal(y, data.y):
            raise ValueError('Cached future train labels differ from independently reconstructed proxies')
        return data, arrays, metadata
