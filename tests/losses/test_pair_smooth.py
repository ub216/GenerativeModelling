import pytest
import torch
import torch.nn.functional as F
from losses.pair_smooth import PairSmoothLoss


@pytest.fixture
def loss_fn():
    return PairSmoothLoss()


class TestPairSmoothLoss:
    def test_loss_is_scalar(self, loss_fn):
        x, y = torch.rand(2, 1, 4, 4), torch.rand(2, 1, 4, 4)
        assert loss_fn((x, y)).ndim == 0

    def test_zero_when_identical(self, loss_fn):
        x = torch.rand(2, 1, 4, 4)
        assert torch.isclose(loss_fn((x, x)), torch.tensor(0.0), atol=1e-6)

    def test_positive_when_different(self, loss_fn):
        x = torch.zeros(2, 1, 4, 4)
        y = torch.ones(2, 1, 4, 4)
        assert loss_fn((x, y)) > 0

    def test_shape_mismatch_raises(self, loss_fn):
        with pytest.raises(AssertionError):
            loss_fn((torch.rand(2, 1, 4, 4), torch.rand(2, 1, 4, 8)))

    def test_invalid_reduction_raises(self):
        with pytest.raises(AssertionError):
            PairSmoothLoss(reduction="invalid")

    def test_mean_vs_sum_reduction(self):
        fn_mean = PairSmoothLoss(reduction="mean")
        fn_sum = PairSmoothLoss(reduction="sum")
        x = torch.rand(4, 1, 4, 4)
        y = torch.rand(4, 1, 4, 4)
        assert torch.isclose(fn_sum((x, y)), fn_mean((x, y)) * 4, atol=1e-5)

    def test_uses_smooth_l1_not_mse(self):
        """For large errors (> beta), smooth L1 grows linearly, not quadratically."""
        fn = PairSmoothLoss(beta=1.0)
        x = torch.zeros(1, 1, 1, 1)
        y = torch.tensor([[[[10.0]]]])  # large error
        # smooth_l1 at 10.0 with beta=1: linear part → |10| - 0.5 = 9.5
        expected = F.smooth_l1_loss(x, y, beta=1.0, reduction="mean")
        assert torch.isclose(fn((x, y)), expected, atol=1e-5)

    def test_weights_scale_contributions(self):
        fn = PairSmoothLoss(reduction="sum")
        x = torch.zeros(2, 1, 4, 4)
        y = torch.ones(2, 1, 4, 4) * 5.0  # large error → linear regime
        weights = torch.tensor([2.0, 1.0])
        loss_equal = fn((x, y, torch.ones(2)))
        loss_weighted = fn((x, y, weights))
        assert torch.isclose(loss_weighted / loss_equal, torch.tensor(1.5), atol=1e-5)
