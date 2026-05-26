import pytest
import torch

from losses.mean_flow_mse import MeanFlowMSELoss


def _make_outputs(b=4, c=1, h=8, w=8, boundary_count=2):
    """Return (pred, target, same_time) with `boundary_count` True entries."""
    pred = torch.rand(b, c, h, w, requires_grad=True)
    target = torch.rand(b, c, h, w)
    same_time = torch.zeros(b, dtype=torch.bool)
    same_time[:boundary_count] = True
    return pred, target, same_time


@pytest.fixture
def loss_fn():
    return MeanFlowMSELoss(reduction="adaptive", adaptive_power=0.5)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    def test_all_key_always_present(self, loss_fn):
        outputs = _make_outputs()
        result = loss_fn(outputs)
        assert "all" in result

    def test_boundary_key_present_when_some_same_time(self, loss_fn):
        outputs = _make_outputs(boundary_count=2)
        result = loss_fn(outputs)
        assert "boundary" in result

    def test_meanflow_key_present_when_some_not_same_time(self, loss_fn):
        outputs = _make_outputs(boundary_count=2)
        result = loss_fn(outputs)
        assert "meanflow" in result

    def test_boundary_key_absent_when_no_same_time(self, loss_fn):
        pred, target, same_time = _make_outputs(boundary_count=0)
        result = loss_fn((pred, target, same_time))
        assert "boundary" not in result

    def test_meanflow_key_absent_when_all_same_time(self, loss_fn):
        pred, target, same_time = _make_outputs(b=4, boundary_count=4)
        result = loss_fn((pred, target, same_time))
        assert "meanflow" not in result

    def test_all_values_are_scalars(self, loss_fn):
        outputs = _make_outputs()
        result = loss_fn(outputs)
        for k, v in result.items():
            assert v.ndim == 0, f"{k} should be a scalar"


# ---------------------------------------------------------------------------
# Grad / detach contract
# ---------------------------------------------------------------------------


class TestGradContract:
    def test_all_has_grad(self, loss_fn):
        outputs = _make_outputs()
        result = loss_fn(outputs)
        assert result["all"].requires_grad is True

    def test_boundary_is_detached(self, loss_fn):
        outputs = _make_outputs(boundary_count=2)
        result = loss_fn(outputs)
        assert result["boundary"].requires_grad is False
        assert result["boundary"].grad_fn is None

    def test_meanflow_is_detached(self, loss_fn):
        outputs = _make_outputs(boundary_count=2)
        result = loss_fn(outputs)
        assert result["meanflow"].requires_grad is False
        assert result["meanflow"].grad_fn is None

    def test_backward_through_all_succeeds(self, loss_fn):
        pred, target, same_time = _make_outputs()
        result = loss_fn((pred, target, same_time))
        result["all"].backward()  # must not raise
        assert pred.grad is not None

    def test_requires_three_element_tuple(self, loss_fn):
        pred, target, _ = _make_outputs()
        with pytest.raises(ValueError):
            loss_fn((pred, target))


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


class TestNumerics:
    def test_all_loss_is_finite(self, loss_fn):
        outputs = _make_outputs()
        result = loss_fn(outputs)
        assert torch.isfinite(result["all"])

    def test_monitoring_losses_are_finite(self, loss_fn):
        outputs = _make_outputs(boundary_count=2)
        result = loss_fn(outputs)
        assert torch.isfinite(result["boundary"])
        assert torch.isfinite(result["meanflow"])

    def test_zero_when_pred_equals_target(self, loss_fn):
        x = torch.rand(4, 1, 8, 8, requires_grad=True)
        same_time = torch.zeros(4, dtype=torch.bool)
        result = loss_fn((x, x.detach(), same_time))
        assert torch.isclose(result["all"], torch.tensor(0.0), atol=1e-6)
