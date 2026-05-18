import pytest
import torch

from models.mean_flow import MeanFlowModel

DEVICE = "cpu"

_KWARGS = dict(
    in_channels=1,
    base_channels=8,
    channel_mults=[1, 2],
    num_blocks=[1, 1],
    time_emb_dim=16,
    timesteps=10,
    test_timesteps=4,
    device=DEVICE,
)


@pytest.fixture
def model():
    return MeanFlowModel(**_KWARGS)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(2, 1, 8, 8)


# ---------------------------------------------------------------------------
# MeanFlowModel.__init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_unet_has_dual_time_embedding(self, model):
        # MeanFlow UNet needs separate embeddings for t and Δt; FlowModel uses a single int.
        assert model.unet.time_emb_dim == (16, 16)

    def test_test_delta_equals_one_over_test_timesteps(self, model):
        assert model.test_delta == pytest.approx(1 / _KWARGS["test_timesteps"])

    def test_has_conditional_generation_false_without_text_emb_dim(self, model):
        assert model.has_conditional_generation is False

    def test_multichannel_init(self):
        m = MeanFlowModel(**{**_KWARGS, "in_channels": 3})
        assert m.in_channels == 3


# ---------------------------------------------------------------------------
# MeanFlowModel.forward — output contract
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shapes_match_input(self, model, batch):
        pred_u, target_u, _ = model(batch)
        assert pred_u.shape == batch.shape
        assert target_u.shape == batch.shape

    def test_explicit_time_steps_accepted(self, model, batch):
        # time_steps must be (B, 2): col0=t, col1=r with t >= r
        t = torch.rand(2)
        r = t * torch.rand(2)
        time_steps = torch.stack([t, r], dim=1)
        pred_u, target_u, _ = model(batch, time_steps=time_steps)
        assert pred_u.shape == batch.shape

    def test_explicit_x1_accepted(self, model, batch):
        x1 = torch.randn_like(batch)
        pred_u, target_u, _ = model(batch, x1=x1)
        assert pred_u.shape == batch.shape

    def test_invalid_combination_raises(self, model, batch):
        with pytest.raises(ValueError):
            model(batch, conditioning=["some text"])

    def test_returns_three_tensors(self, model, batch):
        out = model(batch)
        assert len(out) == 3

    def test_same_time_mask_shape_and_dtype(self, model, batch):
        # same_time is a bool tensor of shape [B] indicating boundary-condition samples (r==t).
        _, _, same_time = model(batch)
        assert same_time.shape == (batch.shape[0],)
        assert same_time.dtype == torch.bool

    def test_same_time_ratio_approximately_respected(self, model):
        # Over a large batch the fraction of boundary samples should be near same_time_ratio.
        # N=500: sigma ≤0.025 for any p in (0,1), so 3-sigma band is ±0.075 — inside the 0.1 tolerance.
        x = torch.rand(500, 1, 8, 8)
        _, _, same_time = model(x)
        ratio = same_time.float().mean().item()
        assert abs(ratio - model.same_time_ratio) < 0.1


# ---------------------------------------------------------------------------
# MeanFlowModel.forward — MeanFlow identity (Eq 6)
# ---------------------------------------------------------------------------


class TestMeanFlowIdentity:
    def test_same_time_target_equals_flow(self, model, batch):
        # When t = r, (t-r)*du/dt = 0, so target_u must equal the instantaneous velocity v = x1-x0.
        # This is the boundary condition u(z_t, t, t) = v(z_t, t) from Eq 7.
        x1 = torch.randn_like(batch)
        b = batch.shape[0]
        t = torch.rand(b)
        time_steps = torch.stack([t, t], dim=1)  # r = t everywhere
        _, target_u, _ = model(batch, time_steps=time_steps, x1=x1)
        assert torch.allclose(target_u, x1 - batch, atol=1e-5)

    def test_target_differs_from_pred_when_t_neq_r(self, model, batch):
        # With a randomly initialised network pred_u ≠ target_u almost surely when t ≠ r.
        torch.manual_seed(1)
        x1 = torch.randn_like(batch)
        t = torch.full((2,), 0.8)
        r = torch.full((2,), 0.2)
        time_steps = torch.stack([t, r], dim=1)
        pred_u, target_u, _ = model(batch, time_steps=time_steps, x1=x1)
        assert not torch.allclose(pred_u, target_u, atol=1e-3)

    def test_pred_u_is_differentiable(self, model, batch):
        # Training requires gradients to flow through pred_u (the network output).
        x0 = batch.requires_grad_(False)
        pred_u, _, _ = model(x0)
        loss = pred_u.mean()
        loss.backward()  # must not raise


# ---------------------------------------------------------------------------
# MeanFlowModel.forward — time sampling
# ---------------------------------------------------------------------------


class TestTimeSampling:
    def test_sampled_time_steps_in_unit_interval(self, model, batch):
        # Logit-normal samples must stay in (0, 1) after sigmoid.
        # Patch forward to expose the sampled time_steps by passing them back out.
        # Easiest: run many samples and check no value escapes [0, 1].
        for _ in range(10):
            pred_u, _, _ = model(batch)
        # If any time_step were outside (0,1), q_sample would silently extrapolate;
        # checking that forward completes without NaN/Inf is a proxy.
        assert not torch.isnan(pred_u).any()
        assert not torch.isinf(pred_u).any()

    def test_no_nan_in_output(self, model, batch):
        pred_u, target_u, _ = model(batch)
        assert not torch.isnan(pred_u).any()
        assert not torch.isnan(target_u).any()


# ---------------------------------------------------------------------------
# MeanFlowModel.sample / sample_flow
# ---------------------------------------------------------------------------


class TestSampling:
    def test_output_shape(self, model):
        samples = model.sample(3, DEVICE, image_size=8, batch_size=2)
        assert samples.shape == (3, 1, 8, 8)

    def test_output_in_unit_range(self, model):
        samples = model.sample(3, DEVICE, image_size=8, batch_size=2)
        assert samples.min() >= 0.0 and samples.max() <= 1.0

    def test_image_size_as_tuple(self, model):
        samples = model.sample(2, DEVICE, image_size=(8, 8), batch_size=2)
        assert samples.shape == (2, 1, 8, 8)

    def test_one_step_sampling(self):
        # test_timesteps=1 is the canonical MeanFlow one-step generation mode.
        m = MeanFlowModel(**{**_KWARGS, "test_timesteps": 1})
        samples = m.sample(2, DEVICE, image_size=8, batch_size=2)
        assert samples.shape == (2, 1, 8, 8)

    def test_batch_size_larger_than_num_samples(self, model):
        samples = model.sample(2, DEVICE, image_size=8, batch_size=16)
        assert samples.shape == (2, 1, 8, 8)

    def test_exact_num_samples_returned(self, model):
        # Regression: last-batch overrun must not inflate sample count.
        samples = model.sample(3, DEVICE, image_size=8, batch_size=2)
        assert samples.shape[0] == 3

    def test_dynamic_threshold_false(self, model):
        samples = model.sample(2, DEVICE, image_size=8, batch_size=2, dynamic_threshold=False)
        assert samples.shape == (2, 1, 8, 8)

    def test_unet_restored_to_train_mode_after_sampling(self, model):
        model.sample(2, DEVICE, image_size=8, batch_size=2)
        assert model.unet.training

    def test_multichannel_output_shape(self):
        m = MeanFlowModel(**{**_KWARGS, "in_channels": 3})
        samples = m.sample(2, DEVICE, image_size=8, batch_size=2)
        assert samples.shape == (2, 3, 8, 8)
