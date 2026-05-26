import pytest
import torch

from losses.pair_mad import PairMADLoss


@pytest.fixture
def loss_fn():
    return PairMADLoss()


class TestPairMADLoss:
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
        with pytest.raises(ValueError):
            loss_fn((torch.rand(2, 1, 4, 4), torch.rand(2, 1, 4, 8)))

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError):
            PairMADLoss(reduction="invalid")

    def test_mean_vs_sum_reduction(self):
        fn_mean = PairMADLoss(reduction="mean")
        fn_sum = PairMADLoss(reduction="sum")
        x = torch.rand(4, 1, 4, 4)
        y = torch.rand(4, 1, 4, 4)
        assert torch.isclose(fn_sum((x, y)), fn_mean((x, y)) * 4, atol=1e-5)

    def test_uses_l1_not_l2(self):
        """MAD uses L1 (absolute) error, so loss = |error|, not error²."""
        fn = PairMADLoss(reduction="mean")
        x = torch.zeros(1, 1, 1, 1)
        y = torch.tensor([[[[3.0]]]])
        # L1 loss for a single element: |3 - 0| = 3.0
        assert torch.isclose(fn((x, y)), torch.tensor(3.0), atol=1e-6)

    def test_l1_less_than_mse_for_large_errors(self):
        """For large errors, L1 < MSE (no quadratic blow-up)."""
        from losses.pair_mse import PairMSELoss

        fn_mad = PairMADLoss(reduction="mean")
        fn_mse = PairMSELoss(reduction="mean")
        x = torch.zeros(2, 1, 4, 4)
        y = torch.ones(2, 1, 4, 4) * 10.0
        assert fn_mad((x, y)) < fn_mse((x, y))

    def test_weights_scale_contributions(self):
        fn = PairMADLoss(reduction="sum")
        x = torch.zeros(2, 1, 4, 4)
        y = torch.ones(2, 1, 4, 4)
        loss_equal = fn((x, y, torch.ones(2)))
        loss_doubled = fn((x, y, torch.tensor([2.0, 1.0])))
        assert torch.isclose(loss_doubled / loss_equal, torch.tensor(1.5), atol=1e-5)
