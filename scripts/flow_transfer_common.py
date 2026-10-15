"""Shared I5_N fixed-channel transfer roles and public evaluation inputs."""
import json
from pathlib import Path
import numpy as np
from trafficflow.data import Release,VALUES
from trafficflow.state import StateModel,StateFeatures
from trafficflow.experiments import partition_date,day_template,target_truth
from trafficflow.adaptive_graph_state import reconstruction_input
from trafficflow.runtime import sha256_file
from recurrent_state_experiment import transition_masks


def context(study_path):
    study=json.loads(Path(study_path).read_text());cfg=study['registered']['protocol']
    folder=Path('artifacts/research')/Path(study_path).stem
    views=json.loads((Path(cfg['views'])/'cache.json').read_text())
    original=json.loads(Path(views['registered']['protocol']['source_study']).read_text())
    return study,cfg,folder,views,original


def compose(values,primary):
    out=dict(values);out['flow_seed_mean']=(out['flow_seed20261013']+out['flow_seed20261029'])/2
    for key,flow in [('channel_seed20261013','flow_seed20261013'),('channel_seed20261029','flow_seed20261029'),('transfer_fixed','flow_seed_mean')]:
        a=out['nn_primary'].copy();a[:,1]=out[flow][:,1];out[key]=a
    if primary is not None:out['primary_export']=out[primary]
    return out


class PublicDays:
    def __init__(self,panel,profile,version):
        self.release=Release('data/kaggle_public');self.panel=panel;self.profile=profile;self.version=version
        self.net,self.topology=self.release.network(panel),self.release.topology(panel)
        self.masked={partition_date(self.release,p):p for p in self.release.files(panel,'train')}
        self.raw={p.name:p for p in self.release.files(panel,'train',False)};self.template=self.release.template(1,panel,'train')
        self.hashes={};self.factor=None

    def load(self,date,cache=None):
        path=self.masked[date];raw_path=self.raw[path.name]
        for p in [path,raw_path]:self.hashes[p.as_posix()]=sha256_file(p)
        frame=self.release.read(path);raw=self.release.read(raw_path)
        f=StateFeatures(frame,self.net,self.topology,self.profile,self.version)
        target=day_template(self.release,self.panel,'train',frame,self.template)
        lanes=target.link_id.map(self.net.lanes).to_numpy(float)
        teacher=target_truth(self.release,self.panel,target,raw)[VALUES].to_numpy(float);teacher[:,1]/=lanes
        if cache is not None:
            day=dict(np.load(cache))
            if not np.array_equal(reconstruction_input(f),day['x']) or not np.array_equal(teacher,day['station_truth']) or not np.array_equal(f.lookup(target),day['target_indices']):raise ValueError('Original public inputs/teacher differ')
        else:
            indices=f.lookup(target);shape=(len(f.times),len(f.links));y=np.zeros((*shape,2),np.float32);mask=np.zeros(shape,bool)
            # Grid labels are only used for the separate legacy diagnostic.
            import pandas as pd
            grouped=pd.DataFrame({'i':indices,'v':teacher[:,0],'q':teacher[:,1]}).groupby('i')[['v','q']].mean()
            y.reshape(-1,2)[grouped.index]=grouped.to_numpy()/np.array([25.,600.]);mask.ravel()[indices]=True
            day={'x':reconstruction_input(f),'y':y,'mask':mask,'station_truth':teacher,'target_indices':indices,
                'target_link_lanes':lanes,'station_lanes':self.release.station_lanes(self.panel,target),'regime':frame.mask_regime.iloc[0]}
        day.update(date=date,features=f,template=target);day['diagnostic_masks']=transition_masks(day,target,raw)
        self.factor=(self.net.lanes*self.net.length_km).reindex(f.links).to_numpy(np.float32)
        return day


def load_exports(study):
    result={}
    for role,row in study['exports'].items():
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('Frozen export changed')
        result[role]=StateModel.load(Path(row['path']))
    return result
