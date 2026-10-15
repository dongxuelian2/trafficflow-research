import pytest
torch=pytest.importorskip('torch')
from trafficflow.state_control_model import state_only_loss


def test_state_only_loss_matches_current_component_and_masks_unknown_gradients():
    p=torch.zeros(2,2,3,requires_grad=True);truth=torch.tensor([[[1.,3.,999.],[2.,999.,999.]],[[4.,999.,999.],[1.,3.,999.]]]);mask=truth<100
    loss=state_only_loss(p,truth,mask);assert float(loss.detach())==pytest.approx(.5)
    loss.backward();assert not p.grad[~mask].any() and p.grad[mask].lt(0).all()


def test_portable_state_disables_TF32_locally_and_restores_backend(monkeypatch):
    import numpy as np
    from trafficflow.joint_graph_queue import JointGraphQueue
    from trafficflow.state_control_model import StateControlModel
    original=JointGraphQueue.analyze;seen=[]
    def checked(self,*args):
        seen.append(torch.backends.cudnn.allow_tf32);return original(self,*args)
    monkeypatch.setattr(JointGraphQueue,'analyze',checked);graph={'geometry':np.array([[100.]*3,[2000.]*3,[150.]*3,[60.]*3],np.float32),'incoming':np.eye(3,dtype=np.float32),'outgoing':np.eye(3,dtype=np.float32)};network=JointGraphQueue();model=StateControlModel(['a','b','c'],graph,states=[network.state_dict()]);x=np.zeros((1,13,3,9),np.float32)
    before=torch.backends.cudnn.allow_tf32;torch.backends.cudnn.allow_tf32=True
    try:
        one=model.predict_tensor(x);many=model.predict_tensor(np.repeat(x,48,axis=0));assert torch.backends.cudnn.allow_tf32 is True
        assert seen and not any(seen);np.testing.assert_allclose(one[0],many[0],atol=1e-5,rtol=0)
    finally:torch.backends.cudnn.allow_tf32=before
