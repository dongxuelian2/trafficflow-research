import numpy as np
import pytest
torch=pytest.importorskip('torch')
from trafficflow.speed_jump_loss import speed_jump_pair_loss,speed_jump_objective
from trafficflow.adaptive_graph_state import patch_score_loss


def test_one_supervised_jump_has_hand_calculated_endpoint_gradients_only():
    truth=torch.tensor([[[[3.],[1.],[100.]],[[1.],[1.],[100.]]]],dtype=torch.float64)
    prediction=torch.tensor([[[[2.],[1.],[-100.]],[[7.],[8.],[-100.]]]],dtype=torch.float64,requires_grad=True)
    target=torch.tensor([[[True],[True],[False]]]);loss=speed_jump_pair_loss(prediction,truth,target);loss.backward()
    scale=np.sqrt(1+1e-8);assert loss.item()==pytest.approx(scale,abs=1e-12)
    np.testing.assert_allclose(prediction.grad.numpy(),[[[[-1/scale],[1/scale],[0]],[[0],[0],[0]]]],atol=1e-12)


def test_unpaired_targets_produce_zero_auxiliary_loss_and_exact_control_objective():
    y=torch.tensor([[[[0.],[2.],[0.]],[[1.],[1.],[1.]]]])
    p=torch.ones_like(y,requires_grad=True);mask=torch.tensor([[[False],[True],[False]]])
    auxiliary=speed_jump_pair_loss(p,y,mask);assert auxiliary.item()==0.;auxiliary.backward()
    assert not p.grad.any();assert torch.equal(speed_jump_objective(p,y,mask),patch_score_loss(p,y,mask))
