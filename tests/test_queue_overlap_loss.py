import itertools
import math
import pytest
torch = pytest.importorskip('torch')
from trafficflow.queue_overlap_loss import lovasz_binary_hinge, queue_window_overlap


def test_lovasz_vertices_match_independently_enumerated_jaccard_set_loss():
    for bits in itertools.product([0., 1.], repeat=3):
        y = torch.tensor(bits)
        for errors in itertools.product([0., 1.], repeat=3):
            mistake = torch.tensor(errors)
            score = (2*y-1)*(1-mistake)
            predicted = torch.logical_xor(y.bool(), mistake.bool())
            union = (y.bool() | predicted).sum().item()
            inter = (y.bool() & predicted).sum().item()
            expected = 1-inter/union if union else 0.
            assert lovasz_binary_hinge(score, y).item() == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize('kind', ['lovasz', 'soft_iou'])
def test_overlap_omits_unavailable_and_known_empty_early_onset_cells(kind):
    logits = torch.zeros((1, 6, 3), requires_grad=True)
    labels = torch.zeros_like(logits); labels[0, 5, 0] = 1.
    valid = torch.ones_like(logits, dtype=torch.bool); valid[0, 5, 2] = False
    loss = queue_window_overlap(logits, labels, valid, torch.tensor([False]), kind)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :5]) == 0
    assert logits.grad[0, 5, 2] == 0
    assert logits.grad[0, 5, 0] < 0
    assert torch.isfinite(logits.grad).all()


def test_soft_overlap_equal_window_weight_and_empty_foreground_extension():
    raw = torch.full((2, 6, 2), math.log(4.))
    y = torch.zeros_like(raw); y[0, 5, 0] = 1.
    valid = torch.zeros_like(raw, dtype=torch.bool); valid[0, 5] = True; valid[1, 5, 0] = True
    # First window: 1 - .5/1.5 = 2/3. Second empty window: mean p = .5.
    value = queue_window_overlap(raw, y, valid, torch.tensor([False, False]), 'soft_iou')
    assert value.item() == pytest.approx((2/3+.5)/2)
