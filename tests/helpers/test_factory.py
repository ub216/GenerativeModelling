import pytest
import torch

from helpers.factory import get_loss_function, get_model, get_optimizer_manager
from helpers.optimizer_manager import OptimizerManager
from losses.gan_hinge_loss import GANHingeLoss
from losses.pair_mse import PairMSELoss
from losses.pair_smooth import PairSmoothLoss
from losses.vae import VAELoss
from models.diffusion import DiffusionModel
from models.ema import EMAModel
from models.flow import FlowModel
from models.vae import VAE

# Minimal params that keep models cheap on CPU
_DIFFUSION_PARAMS = dict(
    base_channels=8,
    channel_mults=[1, 2],
    num_blocks=[1, 1],
    time_emb_dim=16,
    timesteps=10,
    device="cpu",
)

_VAE_PARAMS = dict(
    feature_dims=[4, 8],
    latent_dim=4,
    hidden_dim=16,
)

# image_size tuple passed to get_model: (H, W, C)
_IMG_SIZE_1CH = (8, 8, 1)
_IMG_SIZE_3CH = (8, 8, 3)


# ---------------------------------------------------------------------------
# get_loss_function
# ---------------------------------------------------------------------------


class TestGetLossFunction:
    def test_vae_returns_vae_loss(self):
        loss = get_loss_function({"type": "vae", "params": {}})
        assert isinstance(loss, VAELoss)

    def test_pair_mse_returns_pair_mse_loss(self):
        loss = get_loss_function({"type": "pair_mse", "params": {}})
        assert isinstance(loss, PairMSELoss)

    def test_pair_smooth_returns_pair_smooth_loss(self):
        loss = get_loss_function({"type": "pair_smooth", "params": {}})
        assert isinstance(loss, PairSmoothLoss)

    def test_gan_hinge_returns_gan_hinge_loss(self):
        loss = get_loss_function({"type": "gan_hinge_loss", "params": {}})
        assert isinstance(loss, GANHingeLoss)

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError):
            get_loss_function({"type": "not_a_loss"})

    def test_unrecognised_param_raises_value_error(self):
        with pytest.raises(ValueError, match="Unrecognised config parameters for VAELoss"):
            get_loss_function({"type": "vae", "params": {"typo_param": 1.0}})


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------


class TestGetModel:
    def test_diffusion_returns_diffusion_model(self):
        cfg = {"type": "diffusion", "params": _DIFFUSION_PARAMS.copy()}
        model, ema = get_model(cfg, image_size=_IMG_SIZE_1CH)
        assert isinstance(model, DiffusionModel)
        assert isinstance(ema, EMAModel)

    def test_vae_returns_vae(self):
        cfg = {"type": "vae", "params": _VAE_PARAMS.copy()}
        model, ema = get_model(cfg, image_size=_IMG_SIZE_1CH)
        assert isinstance(model, VAE)
        assert isinstance(ema, EMAModel)

    def test_flow_returns_flow_model(self):
        cfg = {"type": "flow", "params": _DIFFUSION_PARAMS.copy()}
        model, ema = get_model(cfg, image_size=_IMG_SIZE_1CH)
        assert isinstance(model, FlowModel)

    def test_build_ema_false_returns_none(self):
        cfg = {"type": "diffusion", "params": _DIFFUSION_PARAMS.copy()}
        _, ema = get_model(cfg, image_size=_IMG_SIZE_1CH, build_ema=False)
        assert ema is None

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError):
            get_model({"type": "unknown", "params": {}}, image_size=_IMG_SIZE_1CH)

    def test_neither_image_size_nor_dataloader_raises(self):
        with pytest.raises(ValueError):
            get_model({"type": "diffusion", "params": _DIFFUSION_PARAMS.copy()})

    def test_image_size_inferred_from_dataloader(self):
        dummy = torch.rand(4, 1, 8, 8)
        dataset = torch.utils.data.TensorDataset(dummy, torch.zeros(4))
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        cfg = {"type": "diffusion", "params": _DIFFUSION_PARAMS.copy()}
        model, _ = get_model(cfg, dataloader=loader)
        assert isinstance(model, DiffusionModel)

    def test_in_channels_override_from_image_size(self):
        cfg = {"type": "diffusion", "params": {**_DIFFUSION_PARAMS, "in_channels": 99}}
        model, _ = get_model(cfg, image_size=_IMG_SIZE_1CH)
        # get_model overrides in_channels to match the dataloader/image_size
        assert model.in_channels == 1

    def test_unrecognised_param_raises_value_error(self):
        cfg = {"type": "diffusion", "params": {**_DIFFUSION_PARAMS, "typo_param": True}}
        with pytest.raises(ValueError, match="Unrecognised config parameters for DiffusionModel"):
            get_model(cfg, image_size=_IMG_SIZE_1CH)

    def test_factory_injected_image_size_not_flagged_for_models_without_it(self):
        # image_size is injected by the factory for all models; models that don't accept it
        # (e.g. DiffusionModel) should receive it silently filtered out, not as a ValueError.
        cfg = {"type": "diffusion", "params": _DIFFUSION_PARAMS.copy()}
        model, _ = get_model(cfg, image_size=_IMG_SIZE_1CH)
        assert isinstance(model, DiffusionModel)

    def test_unrecognised_param_vae_raises_value_error(self):
        cfg = {"type": "vae", "params": {**_VAE_PARAMS, "bad_key": 999}}
        with pytest.raises(ValueError, match="Unrecognised config parameters for VAE"):
            get_model(cfg, image_size=_IMG_SIZE_1CH)


# ---------------------------------------------------------------------------
# get_optimizer_manager
# ---------------------------------------------------------------------------


class TestGetOptimizerManager:
    def test_single_optimizer_all_key(self):
        model = VAE(**_VAE_PARAMS, in_channels=1, image_size=8)
        cfg = {"type": "adam", "params": {"lr": 1e-3}}
        manager = get_optimizer_manager(cfg, model)
        assert isinstance(manager, OptimizerManager)
        assert "all" in manager.optimizers

    def test_adamw_optimizer(self):
        model = VAE(**_VAE_PARAMS, in_channels=1, image_size=8)
        cfg = {"type": "adamw", "params": {"lr": 1e-3}}
        manager = get_optimizer_manager(cfg, model)
        assert isinstance(manager.optimizers["all"], torch.optim.AdamW)

    def test_unknown_optimizer_raises_value_error(self):
        model = VAE(**_VAE_PARAMS, in_channels=1, image_size=8)
        cfg = {"type": "sgd", "params": {"lr": 1e-3}}
        with pytest.raises(ValueError):
            get_optimizer_manager(cfg, model)
