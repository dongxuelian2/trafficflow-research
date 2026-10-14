from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from trafficflow.queue_selection_uncertainty import select_with_cal_uncertainty
from trafficflow.full_domain_queue import FullDomainQueueModel
from trafficflow.ensemble_queue import EnsembleFullDomainQueueModel
from trafficflow.queue import QueueModel


def candidates(values):
    rows=[]
    for j,ious in enumerate(values):
        windows=[{'window':f'p:2030-10-{i+1:02d}T00:00:00Z:411','condition':'queue_ongoing','IoU':v} for i,v in enumerate(ious)]
        rows.append({'name':['tree','network'][j],'complexity':[j,.25*j], 'metrics':{'windows':windows}})
    return rows


def test_cal_sensitivity_keeps_uniform_gain_when_paired_uncertainty_is_zero():
    calendar=[f'2030-10-{i:02d}' for i in range(1,5)]
    result=select_with_cal_uncertainty(candidates([[.5]*4,[.6]*4]),calendar,block_days=1)
    assert result['argmax']==result['selected']=='network'
    assert result['candidates'][0]['paired_gap_bootstrap_standard_error']<1e-10


def test_cal_sensitivity_can_prefer_simpler_model_for_one_day_gain():
    calendar=[f'2030-10-{i:02d}' for i in range(1,9)]
    result=select_with_cal_uncertainty(candidates([[.5]*4,[1.,.4,.4,.4]]),calendar,block_days=1)
    assert result['argmax']=='network' and result['selected']=='tree'
    assert result['days_with_events']==4 and result['calendar_days']==8
    assert result['candidates'][0]['within_one_gap_SE']


def test_cal_sensitivity_rejects_candidate_window_misalignment():
    rows=candidates([[.5]*4,[.6]*4]);rows[1]['metrics']['windows'][0]['condition']='queue_onset'
    with pytest.raises(ValueError,match='disagree'):
        select_with_cal_uncertainty(rows,[f'2030-10-{i:02d}' for i in range(1,5)])


class Baseline:
    seed=1;iterations=1;threads=1
    def probability(self,features):return np.full(len(features),.5)


def test_score_and_binary_api_preserve_complete_template_under_reordering(monkeypatch):
    def features(history,target,net,topology,origin,**kwargs):
        lead=(pd.to_datetime(target.timestamp,utc=True)-origin).dt.total_seconds()/60
        return pd.DataFrame({'lead_minutes':lead})
    monkeypatch.setattr('trafficflow.full_domain_queue.queue_features',features)
    setting={'thresholds':{'global':.6},'require_nonempty':False,'timing':'before'}
    model=FullDomainQueueModel(Baseline(),['a','b'],setting,.5,{}, {})
    p=np.full((6,2),.9);p[0,0]=.3
    monkeypatch.setattr(model,'_graph_probability',lambda *args:p)
    origin=pd.Timestamp('2030-10-01T00:00:00Z')
    times=[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]]
    canonical=pd.MultiIndex.from_product([times,['a','b']],names=['timestamp','link_id']).to_frame(index=False)
    target=canonical.iloc[::-1].reset_index(drop=True)
    history=pd.DataFrame({'timestamp':[str(origin)]})
    scores=model.forecast_probability(history,target,None,None,origin,'queue_ongoing').queue_score
    binary=model.predict_window(history,target,None,None,origin,'queue_ongoing').queue_pred
    np.testing.assert_allclose(scores,np.ravel((p+.5)/2)[::-1])
    np.testing.assert_array_equal(binary,(scores>=.6).astype(int))


def test_ensemble_save_discards_live_networks_and_restores_current_process_cache(tmp_path):
    setting={'thresholds':{'global':.5},'require_nonempty':False,'timing':'before'}
    model=EnsembleFullDomainQueueModel(Baseline(),['a'],setting,.25,[{},{}],{})
    live=[SimpleNamespace(marker='transient')];model._networks=live;model._network='legacy_transient'
    path=tmp_path/'ensemble.pkl';model.save(path);loaded=QueueModel.load(path)
    assert loaded._networks is None and loaded._network is None
    assert model._networks is live and model._network=='legacy_transient'


def test_portable_tree_uses_only_registered_occupancy_and_public_condition(monkeypatch):
    observed=[]
    def features(history,target,net,topology,origin,**kwargs):
        observed.append(kwargs['include_occupancy'])
        return pd.DataFrame({'lead_minutes':[5,10,15,20,25,30], 'condition_ongoing':0.,'occupancy_current':.1})
    class CheckBaseline(Baseline):
        uses_occupancy_features=True;uses_published_condition=True
        def probability(self,x):
            assert (x.condition_ongoing==1.).all()
            return super().probability(x)
    monkeypatch.setattr('trafficflow.full_domain_queue.queue_features',features)
    setting={'thresholds':{'global':.5},'require_nonempty':False,'timing':'before'}
    model=FullDomainQueueModel(CheckBaseline(),['a'],setting)
    origin=pd.Timestamp('2030-10-01T00:00:00Z')
    target=pd.DataFrame({'timestamp':[(origin+pd.Timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in [5,10,15,20,25,30]],'link_id':'a'})
    model.predict_window(pd.DataFrame({'timestamp':[str(origin)]}),target,None,None,origin,'queue_ongoing')
    assert observed==[True]
