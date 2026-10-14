import pytest
torch = pytest.importorskip('torch')
from trafficflow.adaptive_graph_state import AdaptiveGraphState


@pytest.mark.parametrize('attention',[False,True])
def test_adaptive_graph_supports_variable_nodes_and_finite_gradient(attention):
    torch.set_num_threads(2)
    model = AdaptiveGraphState(hidden=16, heads=2,attention=attention)
    for nodes in (5, 9):
        x = torch.randn(2, 12, 8, nodes)
        graph = torch.eye(nodes)
        fd = torch.stack([torch.full((nodes,), 100.), torch.full((nodes,), 2000.), torch.full((nodes,), 100.)])
        p = model(x, graph, graph, fd)
        assert p.shape == (2, 2, 8, nodes)
        torch.testing.assert_close(p, x[:, 6:8], rtol=0, atol=0)
        (p.square().mean()).backward()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        model.zero_grad(set_to_none=True)
    with pytest.raises(ValueError, match='does not match'):
        model(x, graph[:-1], graph, fd)
