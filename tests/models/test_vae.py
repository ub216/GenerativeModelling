import pytest
import torch
from models.vae import VAE

_KWARGS = dict(
    in_channels=1,
    image_size=8,
    feature_dims=[4, 8],
    latent_dim=4,
    hidden_dim=16,
)


@pytest.fixture
def model():
    return VAE(**_KWARGS)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(2, 1, 8, 8)


# ---------------------------------------------------------------------------
# VAE.forward
# ---------------------------------------------------------------------------

class TestVAEForward:
    def test_returns_three_element_tuple(self, model, batch):
        result = model(batch)
        assert isinstance(result, tuple) and len(result) == 3

    def test_output_shape_matches_input(self, model, batch):
        out, _, _ = model(batch)
        assert out.shape == batch.shape

    def test_z_mean_shape(self, model, batch):
        _, z_mean, _ = model(batch)
        assert z_mean.shape == (2, _KWARGS["latent_dim"])

    def test_z_logvar_shape(self, model, batch):
        _, _, z_logvar = model(batch)
        assert z_logvar.shape == (2, _KWARGS["latent_dim"])

    def test_forward_is_differentiable(self, model, batch):
        batch.requires_grad_(False)
        out, z_mean, z_logvar = model(batch)
        loss = out.sum() + z_mean.sum() + z_logvar.sum()
        loss.backward()  # must not raise


# ---------------------------------------------------------------------------
# VAE.random_sample — reparameterisation
# ---------------------------------------------------------------------------

class TestRandomSample:
    def test_output_shape(self, model):
        z_mean = torch.zeros(2, _KWARGS["latent_dim"])
        z_logvar = torch.zeros(2, _KWARGS["latent_dim"])
        z = model.random_sample(z_mean, z_logvar)
        assert z.shape == z_mean.shape

    def test_is_stochastic(self, model):
        """Two calls with the same mean/logvar should produce different samples."""
        z_mean = torch.zeros(2, _KWARGS["latent_dim"])
        z_logvar = torch.zeros(2, _KWARGS["latent_dim"])
        z1 = model.random_sample(z_mean, z_logvar)
        z2 = model.random_sample(z_mean, z_logvar)
        assert not torch.equal(z1, z2)

    def test_deterministic_at_zero_variance(self, model):
        """With logvar → -∞, std → 0, so sample ≈ mean."""
        z_mean = torch.ones(2, _KWARGS["latent_dim"]) * 3.0
        z_logvar = torch.full_like(z_mean, -30.0)  # exp(-15) ≈ 0
        z = model.random_sample(z_mean, z_logvar)
        assert torch.allclose(z, z_mean, atol=1e-3)


# ---------------------------------------------------------------------------
# VAE.sample — prior sampling
# ---------------------------------------------------------------------------

class TestVAESample:
    def test_output_shape(self, model):
        samples = model.sample(3, device="cpu")
        assert samples.shape == (3, 1, 8, 8)

    def test_different_seeds_give_different_samples(self, model):
        torch.manual_seed(0)
        s1 = model.sample(2, device="cpu")
        torch.manual_seed(1)
        s2 = model.sample(2, device="cpu")
        assert not torch.equal(s1, s2)
