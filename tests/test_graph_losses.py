"""Optional torch tests: physical anchors and missing pairs must be respected."""
import pytest
torch=pytest.importorskip('torch')
from trafficflow.graph_state import density_change_loss,public_density_change_loss


def test_density_increment_anchors_and_ignores_missing_pairs():
    inputs=torch.zeros(1,12,3,1)
    inputs[:,0]=4.
    inputs[:,1]=2.
    inputs[:,2:4]=1.
    target=torch.tensor([[[False],[True],[False]]])
    truth=inputs[:,:2].clone()
    prediction=truth.clone()
    factor=torch.tensor([4.])
    assert density_change_loss(prediction,truth,target,inputs,factor)==0
    prediction[:,1,1]=3.
    assert density_change_loss(prediction,truth,target,inputs,factor)>0
    # Unavailable non-target endpoints carry no physical evidence.
    inputs[:,2:4]=0.
    assert density_change_loss(prediction,truth,target,inputs,factor)==0


def test_public_density_coverage_anchors_and_all_pair_denominator_hand_calculation():
    truth=torch.tensor([[[[4.],[4.],[4.],[4.]],[[1.],[2.],[6.],[8.]]]])
    target=torch.tensor([[[False],[True],[False],[False]]])
    prediction=truth.clone();prediction[:,1,1]=3.
    prediction[:,:,0]=999.;prediction[:,:,2:]=999.
    prediction.requires_grad_(True)
    factor=torch.tensor([1.])
    # True accumulations 6/12/36/48, estimate6/18/36/48. The final
    # non-target pair has zero error but contributes12 to denominator42.
    all_pairs=torch.ones(1,3,1,dtype=torch.bool)
    torch.testing.assert_close(public_density_change_loss(prediction,truth,target,all_pairs,factor),torch.tensor(12./42.))
    # Exclude middle pair by public coverage: error6, denominator6+12.
    coverage=torch.tensor([[[True],[False],[True]]])
    loss=public_density_change_loss(prediction,truth,target,coverage,factor)
    torch.testing.assert_close(loss,torch.tensor(1./3.))
    loss.backward()
    torch.testing.assert_close(prediction.grad[0,:,1,0],torch.tensor([-.25,1./3.]))
    assert torch.count_nonzero(prediction.grad[:,:,0])==0
    assert torch.count_nonzero(prediction.grad[:,:,2:])==0
