import pytest
import torch
import torch.nn as nn

from engine.optimizer_manager import OptimizerManager


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
        manager.backward({"all": loss}, model=m)
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

    for key in state["optimizers"]:
        for k, v in state["optimizers"][key]["state"].items():
            for param_key, val in v.items():
                if isinstance(val, torch.Tensor):
                    assert torch.equal(val, manager2.state_dict()["optimizers"][key]["state"][k][param_key])


def test_step_returns_grad_norm_metrics(simple_model):
    manager = _make_manager(simple_model)
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    metrics = manager.step()
    assert "grads/all_norm_raw" in metrics
    assert "grads/all_norm_clipped" in metrics


def test_backward_skips_no_grad_losses(simple_model):
    # MeanFlowMSELoss returns monitoring sub-losses (requires_grad=False) alongside the
    # backprop loss (requires_grad=True). backward() must call .backward() only on the
    # grad loss and silently skip the monitoring losses — no error, correct gradient.
    manager = _make_manager(simple_model)
    x = torch.rand(2, 4)
    grad_loss = simple_model(x).sum()  # requires_grad=True
    no_grad_loss = grad_loss.detach()  # requires_grad=False, simulates monitoring sub-loss

    assert grad_loss.requires_grad is True
    assert no_grad_loss.requires_grad is False

    manager.backward({"all": grad_loss, "monitoring": no_grad_loss}, model=simple_model)  # must not raise

    # Gradient should be present (from grad_loss) and finite.
    for p in simple_model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_no_sync_entered_only_on_non_final_accumulation_steps(simple_model):
    """With accumulate_steps=2, no_sync must be entered on step 1 but NOT on step 2.
    Step 2 is the final accumulation step where allreduce must fire."""
    import contextlib

    no_sync_enter_count = 0

    @contextlib.contextmanager
    def mock_no_sync():
        nonlocal no_sync_enter_count
        no_sync_enter_count += 1
        yield

    simple_model.no_sync = mock_no_sync
    manager = _make_manager(simple_model, accumulate_steps=2)

    for _ in range(2):
        x = torch.rand(2, 4)
        loss = simple_model(x).sum()
        manager.backward({"all": loss}, model=simple_model)
        manager.step()

    # no_sync entered once (step 1 only); step 2 lets allreduce fire
    assert no_sync_enter_count == 1, f"Expected no_sync entered once, got {no_sync_enter_count}"


def test_no_sync_not_entered_when_accumulate_steps_is_one(simple_model):
    """With accumulate_steps=1, every backward is the final step — no_sync must never be entered."""
    import contextlib

    no_sync_enter_count = 0

    @contextlib.contextmanager
    def mock_no_sync():
        nonlocal no_sync_enter_count
        no_sync_enter_count += 1
        yield

    simple_model.no_sync = mock_no_sync
    manager = _make_manager(simple_model, accumulate_steps=1)

    for _ in range(3):
        x = torch.rand(2, 4)
        loss = simple_model(x).sum()
        manager.backward({"all": loss}, model=simple_model)
        manager.step()

    assert no_sync_enter_count == 0, f"Expected no_sync never entered, got {no_sync_enter_count}"


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


def _make_manager_with_scheduler(model, accumulate_steps=1):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1.0)
    return OptimizerManager(
        {"all": opt},
        model=model,
        use_scaler=False,
        accumulate_steps=accumulate_steps,
        schedulers={"all": sched},
    )


def test_scheduler_steps_on_optimizer_update(simple_model):
    manager = _make_manager_with_scheduler(simple_model)
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    manager.step()
    # LambdaLR starts at last_epoch=0 before any step; after one step it is 1.
    assert manager.schedulers["all"].last_epoch == 1


def test_scheduler_not_stepped_on_accumulation_skip(simple_model):
    manager = _make_manager_with_scheduler(simple_model, accumulate_steps=2)
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    manager.step()  # step_count=1, not an update boundary → scheduler must NOT step
    assert manager.schedulers["all"].last_epoch == 0


def test_state_dict_new_format_keys(simple_model):
    manager = _make_manager_with_scheduler(simple_model)
    state = manager.state_dict()
    assert "optimizers" in state
    assert "schedulers" in state


def test_state_dict_roundtrip_with_scheduler(simple_model):
    manager = _make_manager_with_scheduler(simple_model)
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    manager.step()
    state = manager.state_dict()

    opt2 = torch.optim.Adam(simple_model.parameters(), lr=1e-2)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda=lambda step: 1.0)
    manager2 = OptimizerManager({"all": opt2}, model=simple_model, use_scaler=False, schedulers={"all": sched2})
    manager2.load_state_dict(state)
    assert manager2.schedulers["all"].last_epoch == manager.schedulers["all"].last_epoch


def test_load_state_dict_legacy_format_compat(simple_model):
    # Build a legacy-format state dict (old checkpoint without "optimizers" wrapper).
    opt = torch.optim.Adam(simple_model.parameters(), lr=1e-2)
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    opt.step()
    legacy_state = {"all": opt.state_dict()}

    manager = _make_manager(simple_model)
    manager.load_state_dict(legacy_state)  # must not raise


def test_load_state_dict_missing_scheduler_warns(simple_model):
    from unittest.mock import patch

    manager = _make_manager_with_scheduler(simple_model)
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    manager.step()

    # Build a new-format state dict but strip the scheduler key.
    state = manager.state_dict()
    del state["schedulers"]

    manager2 = _make_manager_with_scheduler(simple_model)
    with patch("engine.optimizer_manager.logger") as mock_logger:
        manager2.load_state_dict(state)
    mock_logger.warning.assert_called_once()
    assert "scheduler" in mock_logger.warning.call_args[0][0].lower()


def test_step_returns_lr_metrics(simple_model):
    manager = _make_manager_with_scheduler(simple_model)
    x = torch.rand(2, 4)
    simple_model(x).sum().backward()
    metrics = manager.step()
    assert "lr/all" in metrics
    assert isinstance(metrics["lr/all"], float)
