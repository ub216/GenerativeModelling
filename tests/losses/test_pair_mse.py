import pytest
import torch

from losses.pair_mse import PairMSELoss


@pytest.fixture
def loss_fn():
    return PairMSELoss()


class TestPairMSELoss:
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
        x = torch.rand(2, 1, 4, 4)
        y = torch.rand(2, 1, 4, 8)
        with pytest.raises(AssertionError):
            loss_fn((x, y))

    def test_invalid_reduction_raises(self):
        with pytest.raises(AssertionError):
            PairMSELoss(reduction="invalid")

    def test_mean_vs_sum_reduction(self):
        fn_mean = PairMSELoss(reduction="mean")
        fn_sum = PairMSELoss(reduction="sum")
        x = torch.rand(4, 1, 4, 4)
        y = torch.rand(4, 1, 4, 4)
        l_mean = fn_mean((x, y))
        l_sum = fn_sum((x, y))
        # sum over B samples = mean * B
        assert torch.isclose(l_sum, l_mean * 4, atol=1e-5)

    def test_weights_scale_contributions(self):
        """Doubling one sample's weight and halving the other's changes the loss predictably."""
        fn = PairMSELoss(reduction="sum")
        x = torch.zeros(2, 1, 4, 4)
        y = torch.ones(2, 1, 4, 4)
        # both samples have per-image MSE = 1.0
        loss_equal = fn((x, y, torch.ones(2)))
        loss_doubled = fn((x, y, torch.tensor([2.0, 1.0])))
        # equal: 1+1=2, doubled: 2+1=3  → ratio 3/2
        assert torch.isclose(loss_doubled / loss_equal, torch.tensor(1.5), atol=1e-5)

    def test_weight_of_zero_excludes_sample(self):
        """A weight of 0 completely excludes a sample from the loss."""
        fn = PairMSELoss(reduction="sum")
        x = torch.zeros(2, 1, 4, 4)
        y = torch.ones(2, 1, 4, 4)
        weights = torch.tensor([0.0, 1.0])
        loss = fn((x, y, weights))
        # only second sample contributes: MSE=1.0, sum=1.0
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


class TestPairMSEAdaptive:
    def test_output_is_scalar(self):
        fn = PairMSELoss(reduction="adaptive")
        x, y = torch.rand(2, 1, 4, 4), torch.rand(2, 1, 4, 4)
        assert fn((x, y)).ndim == 0

    def test_zero_when_identical(self):
        # With identical inputs ||Δ||²=0, weight = 1/eps^p, loss = 0 * weight = 0.
        fn = PairMSELoss(reduction="adaptive")
        x = torch.rand(2, 1, 4, 4)
        assert torch.isclose(fn((x, x)), torch.tensor(0.0), atol=1e-6)

    def test_downweights_large_residuals(self):
        # A sample with a large error should contribute less than its raw MSE would suggest.
        # Compare: adaptive(large_error) / adaptive(small_error) < raw_mse(large) / raw_mse(small).
        fn_adaptive = PairMSELoss(reduction="adaptive")
        fn_mean = PairMSELoss(reduction="mean")
        small_err = (torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4) * 0.1)
        large_err = (torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4) * 10.0)
        ratio_adaptive = fn_adaptive(large_err) / fn_adaptive(small_err)
        ratio_raw = fn_mean(large_err) / fn_mean(small_err)
        assert ratio_adaptive < ratio_raw

    def test_eps_prevents_division_by_zero(self):
        # Identical inputs → ||Δ||²=0; should not produce NaN or Inf with default eps.
        fn = PairMSELoss(reduction="adaptive")
        x = torch.zeros(2, 1, 4, 4)
        loss = fn((x, x))
        assert not torch.isnan(loss) and not torch.isinf(loss)

    def test_adaptive_less_than_mean_for_large_errors(self):
        # Adaptive loss is strictly less than plain mean MSE when errors are large
        # (the downweighting shrinks the effective contribution).
        fn_adaptive = PairMSELoss(reduction="adaptive")
        fn_mean = PairMSELoss(reduction="mean")
        x = torch.zeros(2, 1, 4, 4)
        y = torch.ones(2, 1, 4, 4) * 10.0
        assert fn_adaptive((x, y)) < fn_mean((x, y))
