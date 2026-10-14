import pytest
torch = pytest.importorskip('torch')
from trafficflow.adaptive_graph_state import AdaptiveGraphState
from trafficflow.quality_state import expand_shared_state


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


def test_zero_extra_initialization_is_nested_and_can_learn_added_inputs():
    torch.manual_seed(119)
    original = AdaptiveGraphState(hidden=16)
    torch.nn.init.normal_(original.output.weight, std=.02)
    extended = AdaptiveGraphState(channels=18, hidden=16)
    shared = original.state_dict()
    expanded = expand_shared_state(shared, 18)
    extended.load_state_dict(expanded)
    x = torch.randn(1, 12, 8, 5)
    extra = torch.randn(1, 6, 8, 5)
    graph = torch.eye(5)
    fd = torch.tensor([[100.]*5, [2000.]*5, [100.]*5])
    a, b = original(x, graph, graph, fd), extended(torch.cat([x, extra], dim=1), graph, graph, fd)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    b.square().mean().backward()
    assert torch.isfinite(extended.input.weight.grad).all()
    assert extended.input.weight.grad[:, 12:].abs().sum() > 0
    torch.testing.assert_close(shared['input.weight'], expanded['input.weight'][:, :12], rtol=0, atol=0)


def test_density_coordinates_keep_initial_state_and_decouple_density_from_speed():
    model = AdaptiveGraphState(hidden=16, output_coordinates='density')
    x = torch.zeros(1, 12, 4, 5)
    x[:, 6], x[:, 7] = 4., 1.
    graph = torch.eye(5)
    fd = torch.tensor([[100.]*5, [2000.]*5, [150.]*5])
    torch.testing.assert_close(model(x, graph, graph, fd), x[:, 6:8], rtol=0, atol=0)
    with torch.no_grad():
        model.output.bias.copy_(torch.tensor([-.8, .5]))
    output = model(x, graph, graph, fd)
    density = output[:, 1]*600/(output[:, 0]*25).clamp(min=1)
    torch.testing.assert_close(density, torch.full_like(density, 9.), rtol=1e-6, atol=1e-6)
    density.sum().backward()
    assert model.output.bias.grad[0].abs() < 1e-5
    assert model.output.bias.grad[1] > 0
    assert torch.isfinite(model.output.weight.grad).all()
