import pytest
import torch

from losses.vae import VAELoss


@pytest.fixture
def loss_fn():
    return VAELoss()


def _outputs(recon_zero=True, kl_zero=True, b=2, d=4):
    output = torch.rand(b, d)
    inputs = output.clone() if recon_zero else torch.rand(b, d)
    z_mean = torch.zeros(b, d) if kl_zero else torch.ones(b, d) * 2.0
    z_logvar = torch.zeros(b, d)
    return (output, z_mean, z_logvar), inputs


class TestVAELoss:
    def test_loss_is_scalar(self, loss_fn):
        outputs, inputs = _outputs()
        assert loss_fn(outputs, inputs).ndim == 0

    def test_perfect_reconstruction_zero_kl_gives_zero(self, loss_fn):
        """recon=0 (output==input) and KL=0 (mean=0, logvar=0) → total loss ≈ 0."""
        outputs, inputs = _outputs(recon_zero=True, kl_zero=True)
        loss = loss_fn(outputs, inputs)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_loss_increases_with_reconstruction_error(self, loss_fn):
        output_good = torch.zeros(2, 4)
        output_bad = torch.ones(2, 4)
        inputs = torch.zeros(2, 4)
        z_mean = torch.zeros(2, 4)
        z_logvar = torch.zeros(2, 4)
        loss_good = loss_fn((output_good, z_mean, z_logvar), inputs)
        loss_bad = loss_fn((output_bad, z_mean, z_logvar), inputs)
        assert loss_bad > loss_good

    def test_kl_positive_for_nonzero_mean(self):
        """KL divergence is positive when mean != 0."""
        fn = VAELoss(recon_loss_weight=0.0, kl_div_weight=1.0)
        output = inputs = torch.zeros(2, 4)
        z_mean = torch.ones(2, 4) * 2.0
        z_logvar = torch.zeros(2, 4)
        assert fn((output, z_mean, z_logvar), inputs) > 0

    def test_kl_zero_at_standard_normal(self):
        """KL( N(0,1) || N(0,1) ) = 0."""
        fn = VAELoss(recon_loss_weight=0.0, kl_div_weight=1.0)
        output = inputs = torch.zeros(2, 4)
        z_mean = torch.zeros(2, 4)
        z_logvar = torch.zeros(2, 4)
        assert torch.isclose(fn((output, z_mean, z_logvar), inputs), torch.tensor(0.0), atol=1e-6)

    def test_recon_weight_scales_loss(self):
        fn_1x = VAELoss(recon_loss_weight=1.0, kl_div_weight=0.0)
        fn_2x = VAELoss(recon_loss_weight=2.0, kl_div_weight=0.0)
        output = torch.rand(2, 4)
        inputs = torch.rand(2, 4)
        z_mean = z_logvar = torch.zeros(2, 4)
        l1 = fn_1x((output, z_mean, z_logvar), inputs)
        l2 = fn_2x((output, z_mean, z_logvar), inputs)
        assert torch.isclose(l2, l1 * 2, atol=1e-5)

    def test_mean_vs_sum_reduction(self):
        fn_mean = VAELoss(reduction="mean", kl_div_weight=0.0)
        fn_sum = VAELoss(reduction="sum", kl_div_weight=0.0)
        output = torch.rand(2, 4)
        inputs = torch.rand(2, 4)
        z_mean = z_logvar = torch.zeros(2, 4)
        # sum = mean * num_elements (2 * 4 = 8)
        l_mean = fn_mean((output, z_mean, z_logvar), inputs)
        l_sum = fn_sum((output, z_mean, z_logvar), inputs)
        assert torch.isclose(l_sum, l_mean * 8, atol=1e-5)

    def test_invalid_reduction_raises(self):
        with pytest.raises(AssertionError):
            VAELoss(reduction="invalid")

    def test_wrong_tuple_length_raises(self, loss_fn):
        output = torch.rand(2, 4)
        with pytest.raises(AssertionError):
            loss_fn((output, output), output)  # missing z_logvar
