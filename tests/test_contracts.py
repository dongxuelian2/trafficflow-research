import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.optimize import nnls
from trafficflow.data import STATE_KEY,Release
from trafficflow.metrics import state_score,total_score,queue_iou,grouped_iou
from trafficflow.odme import solve
from trafficflow.submission import merge_files,validate_task
from trafficflow.queue import queue_features
from trafficflow.state import Profile,StateFeatures,StateModel,kalman_residual
from trafficflow.fixture import make_fixture
from trafficflow.physics import conservation_operator
from trafficflow.queue_calibration import decisions

def truth_frame():
    return pd.DataFrame({'panel':['D7_I10_E']*3,'timestamp':['2030-06-01T00:00:00Z']*3,
         'station_id':['0001','0002','0003'],'link_id':['L1','L2','L3'],'mask_regime':['R1','R2','R3'],
         'speed_kmh':[100.,100.,100.],'flow_vph':[3000.,3000.,3000.]})

def test_state_per_lane_macro_and_missing_penalty():
    truth=truth_frame();pred=truth.copy();pred.flow_vph+=600
    result=state_score(truth,pred,np.full(3,3.))
    assert result['S_state']==pytest.approx(.54+.46*(1-200/600))
    assert state_score(truth,pred.iloc[:2],np.full(3,3.))['S_state']<result['S_state']
    assert state_score(truth,pred.iloc[:0],np.full(3,3.))['S_state']==0

def test_unknown_is_not_zero_or_formal_total():
    result=total_score({'state':.96,'queue':None,'physics':None,'odme':None})
    assert result['S_total'] is None
    assert result['bounds']==pytest.approx([.336,.986])
    assert total_score({'state':.96,'queue':.89,'physics':.94,'odme':.98})['S_total']==pytest.approx(.94)
    with pytest.raises(ValueError):
        total_score({'state':np.nan})

def test_iou():
    assert queue_iou([0,0],[0,0])==1
    assert queue_iou([1,0],[1,1])==.5

def test_vectorized_iou_matches_window_reference():
    rng=np.random.default_rng(655);windows=np.array(['w3','w1','w2','w0'])
    codes=rng.integers(0,4,350);window=windows[codes];condition=np.where(codes<2,'queue_onset','queue_ongoing')
    y=rng.random(350)<.3;pred=rng.random(350)<.4
    # Include an empty union to preserve the specified IoU=1 edge case.
    y[codes==3]=False;pred[codes==3]=False
    got=grouped_iou(y,pred,window,condition)
    values={w:queue_iou(y[window==w],pred[window==w]) for w in windows}
    expected={str(c):np.mean([values[w] for w in windows if condition[window==w][0]==c]) for c in np.unique(condition)}
    assert got['conditions']==pytest.approx(expected)
    assert got['proxy_IoU']==pytest.approx(np.mean(list(expected.values())))

def test_conditional_decisions_keep_window_boundaries():
    p=np.array([.1,.2,.4,.3,.6,.4])
    windows=np.array(['a','a','b','b','c','c'])
    conditions=np.array(['queue_onset']*2+['queue_ongoing']*4)
    cut={'global':.5,'queue_onset':.15,'queue_ongoing':.5}
    assert decisions(p,windows,conditions,cut).tolist()==[0,1,0,0,1,0]
    assert decisions(p,windows,conditions,cut,True).tolist()==[0,1,1,0,1,0]

def test_offline_kalman_missing_column_and_terminal_measurement():
    prior=np.tile([10.,20.],(3,1));observed=np.full((3,2),np.nan);observed[2,1]=30.
    actual=kalman_residual(observed,prior)
    last=10.*1.075/(1.075+.08);middle=last*1.05/1.075;first=middle*1.025/1.05
    assert actual[:,0]==pytest.approx([10.,10.,10.])
    assert actual[:,1]==pytest.approx(20.+np.array([first,middle,last]))

def test_sparse_nnls_prior_missing_connectors():
    A=sparse.csr_matrix([[1.,1.,0.],[0.,1.,1.]])
    f,info=solve(A,np.array([100.,150.]),np.array([30.,70.,80.]),'ridge',.05)
    assert info['success'] and np.all(f>=0)
    assert np.linalg.norm(A@f-[100,150])<1e-5

def test_measurement_space_nnls_matches_independent_dense_reference():
    rng=np.random.default_rng(921)
    for iteration in range(16):
        A=(rng.random((7,23))<.45).astype(float)
        b=rng.uniform(0,30,23);c=rng.uniform(0,120,7)
        regularization=[.005,.05,5.][iteration%3]
        method='relative' if iteration%2 else 'ridge'
        weight=1/np.maximum(b,10.) if method=='relative' else np.ones_like(b)
        aa=np.vstack([A,np.sqrt(regularization)*np.diag(weight)])
        yy=np.r_[c,np.sqrt(regularization)*b*weight]
        expected,_=nnls(aa,yy)
        actual,info=solve(sparse.csr_matrix(A),c,b,method,regularization)
        assert np.all(actual>=0)
        assert np.max(np.abs(actual-expected))<1e-6
        assert info['KKT_max_absolute_residual']<5e-6

@pytest.fixture(scope='module')
def fixture_release(tmp_path_factory):
    root=tmp_path_factory.mktemp('release')
    make_fixture(root,days=21,panels=['D7_I10_E'])
    return Release(root)

def test_queue_rejects_future_and_historical_cutoff(fixture_release):
    r=fixture_release;p=r.panels()[0]
    history=pd.read_parquet(r.root/'task2'/p/'train'/'window_history.parquet')
    wid=history.window_id.iloc[0];h=history[history.window_id==wid].copy()
    target=r.template(2,p,'train');target=target[target.window_id==wid]
    origin=pd.Timestamp(h.timestamp.max())+pd.Timedelta(minutes=5)
    x=queue_features(h,target,r.network(p),r.topology(p),origin)
    assert len(x)==len(target)
    leaked=h.copy();leaked.iloc[-1,leaked.columns.get_loc('timestamp')]=origin.strftime('%Y-%m-%dT%H:%M:%SZ')
    with pytest.raises(ValueError,match='leakage'):
        queue_features(leaked,target,r.network(p),r.topology(p),origin)
    # The current rules allow exactly T, while later timestamps stay forbidden.
    at_origin=queue_features(leaked,target,r.network(p),r.topology(p),origin,allow_origin=True)
    assert len(at_origin)==len(target)
    leaked.iloc[-1,leaked.columns.get_loc('timestamp')]=(origin+pd.Timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    with pytest.raises(ValueError,match='leakage'):
        queue_features(leaked,target,r.network(p),r.topology(p),origin,allow_origin=True)

def test_masked_features_cannot_access_target_truth(fixture_release):
    r=fixture_release;p=r.panels()[0]
    raw=r.read(r.files(p,'train',False)[0]);profile=Profile.fit([raw])
    masked=r.read(r.files(p,'train',True)[0]);net,topo=r.network(p),r.topology(p)
    features=StateFeatures(masked,net,topo,profile)
    template=r.template(1,p,'train');target=template[template.timestamp.isin(masked.timestamp)]
    idx=features.lookup(target)
    assert np.isnan(features.raw.reshape(-1,2)[idx]).all()
    prediction=StateModel(profile,'interpolation').predict(features,target)
    assert np.isfinite(prediction[['speed_kmh','flow_vph']]).all().all()
    # Extra blank horizon cells are excluded by the published template.
    assert len(target)<(masked.speed_kmh.isna()&masked.flow_vph.isna()&masked.is_score_eligible).sum()

def test_strict_merge_rejects_missing_duplicate_and_nonbinary(tmp_path):
    truth=truth_frame();state=truth.iloc[:1].copy()
    q=pd.DataFrame({'window_id':['w'],'timestamp':['2030-06-01T00:05:00Z'],'link_id':['L1'],'queue_pred':[1]})
    od=pd.DataFrame({'panel':['D7_I10_E'],'departure_time':['TOKEN'],'path_id':['P1'],
                     'origin_zone':['0001'],'destination_zone':['0002'],'path_flow':[100.]})
    frames={'state':state,'queue':q,'odme':od};paths={}
    key=pd.concat([state.assign(task='state'),q.assign(task='queue'),od.assign(task='odme')],ignore_index=True).fillna('')
    key['submission_id']=['0','1','2'];key.to_csv(tmp_path/'key.csv',index=False)
    for task,frame in frames.items():
        paths[task]=tmp_path/f'{task}.csv';frame.to_csv(paths[task],index=False)
    result=merge_files(paths,tmp_path/'key.csv',tmp_path/'out.csv',chunk_size=1)
    assert result['complete'] and result['rows']==3
    out=pd.read_csv(tmp_path/'out.csv');assert not out.isna().any().any()
    q.iloc[:0].to_csv(paths['queue'],index=False)
    with pytest.raises(ValueError,match='missing'):
        merge_files(paths,tmp_path/'key.csv',tmp_path/'bad.csv',chunk_size=1)
    with pytest.raises(ValueError,match='binary'):
        validate_task(q.assign(queue_pred=.5),'queue')

def test_no_raw_hidden_split(fixture_release):
    with pytest.raises(ValueError,match='prohibited'):
        fixture_release.files('D7_I10_E','private',masked=False)

def test_missing_ramp_and_latent_flux_are_not_zero():
    times=['2030-06-01T00:00:00Z','2030-06-01T00:05:00Z']
    links=['a','b','c']
    q=np.full((2,3),1000.);v=np.full((2,3),100.)
    net=pd.DataFrame({'link_id':links,'length_km':[1.,1.,1.]}).set_index('link_id')
    topo=pd.DataFrame({'link_id':links,'incoming_link_ids':['','a','b'],
                       'outgoing_link_ids':['b','c','']}).set_index('link_id')
    on=np.zeros_like(q);off=np.zeros_like(q)
    B,rhs=conservation_operator(times,links,v,q,net,topo,on,off)
    assert len(rhs)==3 and np.allclose(B@q.ravel()-rhs,0)
    on[0,1]=np.nan
    B,rhs=conservation_operator(times,links,v,q,net,topo,on,off)
    assert len(rhs)==2
    on[0,1]=0;topo.loc['b','incoming_link_ids']='unobserved_latent'
    topo.loc['unobserved_latent']=['a','b']
    B,rhs=conservation_operator(times,links,v,q,net,topo,on,off)
    assert len(rhs)==2

def test_master_key_duplicate_across_chunks_is_rejected(tmp_path):
    state=truth_frame().iloc[:1].copy()
    q=pd.DataFrame({'window_id':['w'],'timestamp':['2030-06-01T00:05:00Z'],'link_id':['L1'],'queue_pred':[1]})
    od=pd.DataFrame({'panel':['D7_I10_E'],'departure_time':['TOKEN'],'path_id':['P1'],
                    'origin_zone':['O'],'destination_zone':['D'],'path_flow':[100.]})
    key=pd.concat([state.assign(task='state'),state.assign(task='state'),q.assign(task='queue'),od.assign(task='odme')],ignore_index=True).fillna('')
    key['submission_id']=['0','1','2','3'];key.to_csv(tmp_path/'key.csv',index=False)
    files={}
    for name,d in [('state',state),('queue',q),('odme',od)]:
        files[name]=tmp_path/f'{name}.csv';d.to_csv(files[name],index=False)
    with pytest.raises(ValueError,match='Duplicate natural keys'):
        merge_files(files,tmp_path/'key.csv',tmp_path/'submission.csv',chunk_size=1)
