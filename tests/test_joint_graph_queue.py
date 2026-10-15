import numpy as np
import pytest
torch=pytest.importorskip('torch')
from trafficflow.graph_queue import CausalGraphQueue
from trafficflow.joint_graph_queue import JointGraphQueue,joint_dynamics_loss


def tensors():
    torch.manual_seed(71);x=torch.zeros(2,9,13,4);x[:,0]=3.;x[:,1]=1.;x[:,2:4]=1.;x[:,2:4,-1,0]=0;x[:,:2,-1,0]=0
    graph=[torch.eye(4),torch.eye(4),torch.tensor([[100.]*4,[2000.]*4,[150.]*4,[60.]*4])]
    return x,graph


def test_initial_joint_predictions_preserve_legacy_and_visible_state():
    x,g=tensors();old=CausalGraphQueue();model=JointGraphQueue(rollout=False);model.load_state_dict(old.state_dict(),strict=False);old.eval();model.eval()
    with torch.no_grad():expected=old(x,*g);p,v,q,a=model(x,*g)
    torch.testing.assert_close(p,expected[0],atol=0,rtol=0);torch.testing.assert_close(v,expected[1],atol=0,rtol=0)
    torch.testing.assert_close(q,torch.ones_like(q));torch.testing.assert_close(a[:,0],torch.full_like(a[:,0],3.));torch.testing.assert_close(a[:,1],torch.ones_like(a[:,1]))
    model.analysis_output.bias.data[:]=10.
    with torch.no_grad():a=model(x,*g)[3]
    torch.testing.assert_close(a[:,:,1:],x[:,:2,-1,1:]);assert (a[:,:,0]>10).all()


def test_flow_supervision_and_latent_rollout_have_finite_nonzero_gradients():
    x,g=tensors();model=JointGraphQueue(rollout=True);torch.nn.init.normal_(model.transition_output.weight,0,.01);outputs=model(x,*g)
    a={'labels':torch.ones(2,6,4),'truth_speed':torch.full((2,6,4),2.),'truth_flow':torch.full((2,6,4),2.),'available':torch.ones(2,6,4,dtype=torch.bool),'truth_current':torch.full((2,2,4),2.),'current_domain':torch.ones(2,2,4,dtype=torch.bool),'ongoing':torch.tensor([False,True])}
    loss=joint_dynamics_loss(outputs,a,.1,event=True);loss.backward()
    assert torch.isfinite(loss) and model.flow_output.weight.grad.abs().sum()>0 and model.transition.weight_hh.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    noflow=JointGraphQueue();joint_dynamics_loss(noflow(x,*g),a,0.).backward();assert noflow.flow_output.weight.grad.abs().sum()==0


def test_unknown_future_flow_and_current_labels_have_no_loss_effect():
    x,g=tensors();model=JointGraphQueue();outputs=model(x,*g);mask=torch.ones(2,6,4,dtype=torch.bool);mask[:,5]=False;current=torch.ones(2,2,4,dtype=torch.bool);current[:,:,3]=False
    a={'labels':torch.zeros(2,6,4),'truth_speed':torch.ones(2,6,4),'truth_flow':torch.ones(2,6,4),'available':mask,'truth_current':torch.ones(2,2,4),'current_domain':current}
    before=joint_dynamics_loss(outputs,a);a['truth_flow'][:,5]=999;a['truth_speed'][:,5]=999;a['labels'][:,5]=1;a['truth_current'][:,:,3]=999
    torch.testing.assert_close(joint_dynamics_loss(outputs,a),before,atol=0,rtol=0)


def test_portable_state_condition_uses_same_scaled_threshold_at_boundary():
    import pandas as pd
    from trafficflow.joint_queue_model import JointDynamicsQueueModel
    model=JointDynamicsQueueModel.__new__(JointDynamicsQueueModel);model.links=['a'];conditions=[]
    def predict(history,net,origin,condition):conditions.append(condition);return {'analysis':np.zeros((2,1))}
    model.predict_dynamics=predict;origin=pd.Timestamp('2030-08-01T01:00:00Z');history=pd.DataFrame({'timestamp':pd.date_range(origin-pd.Timedelta(minutes=60),origin,freq='5min'),'link_id':'a','speed_kmh':67.86,'flow_vph':600.,'occupancy':.1,'pct_observed':100.});net=pd.DataFrame({'lanes':[2],'v_cut':[67.86]},index=['a'])
    model.reconstruct_current(history,net,origin)
    assert conditions==['queue_ongoing']
