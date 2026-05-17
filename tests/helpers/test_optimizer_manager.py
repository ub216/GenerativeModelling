import pytest
import torch
import torch.nn as nn

from helpers.optimizer_manager import OptimizerManager


@pytest.fixture
def simple_model():
    m = nn.Linear(4, 2)
    torch.manual_seed(0)
    nn.init.ones_(m.weight)
    nn.init.zeros_(m.bias)
    return m


def _make_manager(model, accumulate_steps=1, use_scaler=False):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    return OptimizerManager(
        {"all": opt},
        model=model,
        use_scaler=use_scaler,
        accumulate_steps=accumulate_steps,
    )


def test_zero_grad_clears_all_gradients(simple_model):
    manager = _make_manager(simple_model)
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    manager.zero_grad()
    for p in simple_model.parameters():
        assert p.grad is None


def test_step_updates_parameters(simple_model):
    manager = _make_manager(simple_model)
    params_before = [p.data.clone() for p in simple_model.parameters()]
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    manager.step()
    params_after = [p.data for p in simple_model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))


def test_step_skipped_before_accumulation(simple_model):
    manager = _make_manager(simple_model, accumulate_steps=2)
    params_before = [p.data.clone() for p in simple_model.parameters()]
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    manager.step()  # step_count=1, accumulate_steps=2 → should NOT update
    params_after = [p.data for p in simple_model.parameters()]
    assert all(torch.equal(b, a) for b, a in zip(params_before, params_after))


def test_step_executes_at_accumulation_boundary(simple_model):
    manager = _make_manager(simple_model, accumulate_steps=2)
    params_before = [p.data.clone() for p in simple_model.parameters()]
    for _ in range(2):
        x = torch.rand(2, 4)
        simple_model(x).sum().backward()
        manager.step()
    params_after = [p.data for p in simple_model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))


def test_step_force_updates_immediately(simple_model):
    """force=True bypasses accumulation logic and always runs the update."""
    manager = _make_manager(simple_model, accumulate_steps=10)
    params_before = [p.data.clone() for p in simple_model.parameters()]
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    manager.step(force=True)
    params_after = [p.data for p in simple_model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))


def test_backward_divides_loss_by_accumulate_steps(simple_model):
    """Gradient should be halved when accumulate_steps=2 vs accumulate_steps=1."""

    def get_grad(accumulate_steps):
        m = nn.Linear(4, 2)
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
        opt = torch.optim.Adam(m.parameters(), lr=0.0)  # no update
        manager = OptimizerManager({"all": opt}, model=m, use_scaler=False, accumulate_steps=accumulate_steps)
        torch.manual_seed(1)
        x = torch.rand(2, 4)
        loss = m(x).sum()
        manager.backward({"all": loss})
        return m.weight.grad.clone()

    grad_1 = get_grad(1)
    grad_2 = get_grad(2)
    assert torch.allclose(grad_2, grad_1 / 2.0, atol=1e-6)


def test_state_dict_roundtrip(simple_model):
    manager = _make_manager(simple_model)
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    manager.step()
    state = manager.state_dict()

    opt2 = torch.optim.Adam(simple_model.parameters(), lr=1e-2)
    manager2 = OptimizerManager({"all": opt2}, model=simple_model, use_scaler=False)
    manager2.load_state_dict(state)

    for key in state:
        for k, v in state[key]["state"].items():
            for param_key, val in v.items():
                if isinstance(val, torch.Tensor):
                    assert torch.equal(val, manager2.state_dict()[key]["state"][k][param_key])


def test_step_returns_grad_norm_metrics(simple_model):
    manager = _make_manager(simple_model)
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    metrics = manager.step()
    assert "grads/all_norm_raw" in metrics
    assert "grads/all_norm_clipped" in metrics
