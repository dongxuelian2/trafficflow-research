import numpy as np
import pytest
torch=pytest.importorskip('torch')
from trafficflow.dense_queue_loss import dense_forecast_loss


def test_dense_loss_supervises_all_leads_and_ignores_unknown_cells():
    logits=torch.zeros((2,6,2),requires_grad=True);speed=torch.zeros_like(logits,requires_grad=True);labels=torch.zeros_like(logits);labels[0,0,0]=1;truth=torch.ones_like(logits);mask=torch.ones_like(logits,dtype=torch.bool);mask[:,5]=False
    loss=dense_forecast_loss(logits,speed,labels,truth,mask);expected=((13/10+1)*np.log(2)/2)+.1
    assert float(loss.detach())==pytest.approx(expected)
    loss.backward();assert logits.grad[0,0,0]<0 and not logits.grad[:,5].any() and not speed.grad[:,5].any()
    other=labels.clone();other[:,5]=1
    assert float(dense_forecast_loss(logits.detach(),speed.detach(),other,truth,mask))==pytest.approx(expected)
