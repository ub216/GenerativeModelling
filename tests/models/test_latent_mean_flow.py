"""
Tests for LatentMeanFlowModel.

The pretrained VAE (stabilityai/sd-vae-ft-mse) is mocked so these tests run
without a network connection or GPU.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from models.latent_mean_flow import LatentMeanFlowModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TINY_KWARGS = dict(
    base_channels=8,
    channel_mults=[1, 2],
    num_blocks=[1, 1],
    time_emb_dim=16,
    test_timesteps=4,
    device="cpu",
    compile_vae=False,
)

# Latent spatial size for a 32×32 input image: 32 // 8 = 4
_LATENT_HW = 4
_LATENT_C = 4


def _make_mock_vae(latent_b=2, img_h=32, img_w=32):
    """Return a MagicMock that mimics AutoencoderKL well enough for our tests."""
    vae = MagicMock()
    vae.config.scaling_factor = 0.18215
    vae.to.return_value = vae
    vae.parameters.return_value = iter([])

    mock_dist = MagicMock()
    mock_dist.sample.return_value = torch.zeros(latent_b, _LATENT_C, _LATENT_HW, _LATENT_HW)
    mock_dist.mode.return_value = torch.zeros(latent_b, _LATENT_C, _LATENT_HW, _LATENT_HW)
    vae.encode.return_value = MagicMock(latent_dist=mock_dist)

    vae.decode.return_value = MagicMock(sample=torch.zeros(latent_b, 3, img_h, img_w))
    return vae


@pytest.fixture
def mock_vae():
    return _make_mock_vae()


@pytest.fixture
def model(mock_vae):
    with patch("models.latent_vae_base.AutoencoderKL.from_pretrained", return_value=mock_vae):
        m = LatentMeanFlowModel(**_TINY_KWARGS)
    return m


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_scaling_factor_set(self, model):
        assert model.scaling_factor == pytest.approx(0.18215)

    def test_vae_put_in_eval_mode(self, mock_vae):
        with patch("models.latent_vae_base.AutoencoderKL.from_pretrained", return_value=mock_vae):
            LatentMeanFlowModel(**_TINY_KWARGS)
        mock_vae.eval.assert_called()

    def test_internal_model_is_mean_flow(self, model):
        from models.mean_flow import MeanFlowModel

        assert isinstance(model.model, MeanFlowModel)

    def test_inner_model_has_four_in_channels(self, model):
        # The UNet inside MeanFlowModel must operate on 4-channel latents
        assert model.model.in_channels == 4

    def test_inner_model_renormalise_disabled(self, model):
        # The outer model owns pixel normalisation; the inner must not double-normalise
        assert model.model.renormalise is False

    def test_inner_model_has_dual_time_embedding(self, model):
        # MeanFlow UNet needs separate embeddings for t and Δt
        assert model.unet.time_emb_dim == (16, 16)


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------


class TestEncodeDecode:
    def test_encode_shape(self, model):
        x = torch.rand(2, 3, 32, 32)
        latents = model.encode(x)
        assert latents.shape == (2, _LATENT_C, _LATENT_HW, _LATENT_HW)

    def test_encode_applies_scaling_factor(self, model, mock_vae):
        """Latents must be multiplied by scaling_factor after sampling."""
        mock_vae.encode.return_value.latent_dist.sample.return_value = torch.ones(2, _LATENT_C, _LATENT_HW, _LATENT_HW)
        x = torch.rand(2, 3, 32, 32)
        latents = model.encode(x)
        expected = 1.0 * model.scaling_factor
        assert torch.allclose(latents, torch.full_like(latents, expected), atol=1e-5)

    def test_decode_shape(self, model):
        z = torch.rand(2, _LATENT_C, _LATENT_HW, _LATENT_HW)
        decoded = model.decode(z)
        assert decoded.shape == (2, 3, 32, 32)

    def test_decode_divides_by_scaling_factor(self, model, mock_vae):
        """decode() must pass z / scaling_factor to the VAE."""
        z = torch.ones(2, _LATENT_C, _LATENT_HW, _LATENT_HW) * model.scaling_factor
        model.decode(z)
        call_args = mock_vae.decode.call_args[0][0]
        assert torch.allclose(call_args, torch.ones_like(call_args), atol=1e-5)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shapes(self, model):
        x = torch.rand(2, 3, 32, 32)
        pred_u, target_u, _ = model(x)
        assert pred_u.shape == (2, _LATENT_C, _LATENT_HW, _LATENT_HW)
        assert target_u.shape == (2, _LATENT_C, _LATENT_HW, _LATENT_HW)

    def test_returns_three_tensors(self, model):
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        assert len(out) == 3

    def test_renormalise_maps_input(self, mock_vae):
        """With renormalise=True, [0,1] input is mapped to [-1,1] before encoding."""
        with patch("models.latent_vae_base.AutoencoderKL.from_pretrained", return_value=mock_vae):
            m = LatentMeanFlowModel(**{**_TINY_KWARGS, "renormalise": True})
        x = torch.rand(2, 3, 32, 32)
        pred_u, target_u, _ = m(x)
        assert pred_u.shape == (2, _LATENT_C, _LATENT_HW, _LATENT_HW)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


class TestSample:
    def test_output_shape(self, model):
        samples = model.sample(num_samples=2, device="cpu", image_size=32, batch_size=2)
        assert samples.shape == (2, 3, 32, 32)

    def test_output_in_unit_range(self, model):
        samples = model.sample(num_samples=2, device="cpu", image_size=32, batch_size=2)
        assert samples.min() >= 0.0 and samples.max() <= 1.0

    def test_image_size_as_tuple(self, model):
        samples = model.sample(num_samples=2, device="cpu", image_size=(32, 32), batch_size=2)
        assert samples.shape == (2, 3, 32, 32)

    def test_unet_restored_to_train_mode_after_sampling(self, model):
        model.sample(num_samples=2, device="cpu", image_size=32, batch_size=2)
        assert model.model.unet.training
