import numpy as np
import pytest
torch=pytest.importorskip('torch')
from trafficflow.recurrent_state import make_network,window_plan,windowed_prediction,training_window


def test_shared_initialization_zero_head_and_zero_gain_match_control():
    torch.set_num_threads(1);torch.manual_seed(7);control=make_network('tcn',8)
    torch.manual_seed(7);candidate=make_network('tcn_gru',8)
    for key,value in control.state_dict().items():assert torch.equal(value,candidate.state_dict()[key])
    x=torch.randn(2,12,9,3);a=torch.eye(3);fd=torch.ones(3,3)
    np.testing.assert_array_equal(candidate(x,a,a,fd).detach().numpy(),x[:,6:8].numpy())
    with torch.no_grad():
        control.output.weight.fill_(.01);candidate.output.weight.copy_(control.output.weight);candidate.temporal_gain.zero_()
    assert torch.equal(candidate(x,a,a,fd),control(x,a,a,fd))
    candidate.temporal_gain.data.fill_(.1);loss=candidate(x,a,a,fd).square().mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in candidate.temporal.parameters())
    candidate.eval();changed=x.clone();changed[1]+=100.
    with torch.no_grad():
        first=candidate(x,a,a,fd)[0];other=candidate(changed,a,a,fd)[0]
    assert torch.equal(first,other), 'Recurrent state must not mix different batch examples'


def test_fixed_window_prediction_matches_manual_context_and_covers_every_cell():
    class SumContext(torch.nn.Module):
        def forward(self,x,*graph):return x[:,6:8]+x[:,0:1].sum(dim=2,keepdim=True)
    x=np.zeros((11,2,12),np.float32);x[:,:,0]=np.arange(11)[:,None];x[:,:,6:8]=2
    plan=window_plan(11,6,2);coverage=np.zeros(11,int);expected=np.empty((11,2,2),np.float32)
    for left,start,end in plan:
        coverage[start:end]+=1;expected[start:end]=2+x[left:left+6,:,0].sum(axis=0)[None,:,None]
    actual=windowed_prediction(SumContext(),x,[],torch.device('cpu'),6,2)
    np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(coverage,np.ones(11))
    short=windowed_prediction(SumContext(),x[:3],[],torch.device('cpu'),6,2)
    np.testing.assert_array_equal(short,np.full((3,2,2),5.,np.float32))


def test_fixed_context_rejects_invalid_shape_and_geometry():
    with pytest.raises(ValueError,match='context'):window_plan(10,5,2)
    with pytest.raises(ValueError,match='strict12'):windowed_prediction(None,np.ones((4,2,13)),[],torch.device('cpu'))
    net=make_network('tcn_gru',8)
    with pytest.raises(ValueError,match='geometry'):net(torch.ones(1,12,4,2),torch.eye(3),torch.eye(2),torch.ones(3,2))


def test_core_first_training_covers_every_day_boundary_without_changing_context():
    coverage=np.zeros(288,int)
    for center in range(288):
        left,start,end=training_window(288,center,128,64)
        assert 0<=left<=start<end<=left+128<=288 and end-start==64
        coverage[start:end]+=1
    assert (coverage>0).all()
    assert training_window(288,0)==(0,0,64)
    assert training_window(288,287)==(160,224,288)
