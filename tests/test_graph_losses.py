"""Optional torch tests: physical anchors and missing pairs must be respected."""
import pytest
torch=pytest.importorskip('torch')
from trafficflow.graph_state import density_change_loss


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
