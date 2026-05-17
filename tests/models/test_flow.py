import pytest
import torch

from models.flow import FlowModel

DEVICE = "cpu"

_KWARGS = dict(
    in_channels=1,
    base_channels=8,
    channel_mults=[1, 2],
    num_blocks=[1, 1],
    time_emb_dim=16,
    timesteps=10,
    device=DEVICE,
)


@pytest.fixture
def model():
    return FlowModel(**_KWARGS)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(2, 1, 8, 8)


# ---------------------------------------------------------------------------
# FlowModel.q_sample — linear interpolation
# ---------------------------------------------------------------------------


class TestQSample:
    def test_at_t0_equals_x0(self, model, batch):
        x1 = torch.randn_like(batch)
        t = torch.zeros(2)
        assert torch.allclose(model.q_sample(batch, t, x1), batch, atol=1e-6)

    def test_at_t1_equals_x1(self, model, batch):
        x1 = torch.randn_like(batch)
        t = torch.ones(2)
        assert torch.allclose(model.q_sample(batch, t, x1), x1, atol=1e-6)

    def test_midpoint_is_linear_blend(self, model, batch):
        x1 = torch.randn_like(batch)
        t = torch.full((2,), 0.5)
        result = model.q_sample(batch, t, x1)
        expected = 0.5 * batch + 0.5 * x1
        assert torch.allclose(result, expected, atol=1e-6)

    def test_output_shape_preserved(self, model, batch):
        x1 = torch.randn_like(batch)
        t = torch.rand(2)
        assert model.q_sample(batch, t, x1).shape == batch.shape


# ---------------------------------------------------------------------------
# FlowModel.forward
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shapes(self, model, batch):
        x1 = torch.randn_like(batch)
        t = torch.rand(2)
        pred_flow, flow = model(batch, time_steps=t, x1=x1)
        assert pred_flow.shape == batch.shape
        assert flow.shape == batch.shape

    def test_flow_target_is_x1_minus_x0(self, model, batch):
        """The flow target should always be x1 - x0."""
        x1 = torch.randn_like(batch)
        t = torch.rand(2)
        _, flow = model(batch, time_steps=t, x1=x1)
        assert torch.allclose(flow, x1 - batch, atol=1e-6)

    def test_random_noise_when_x1_not_provided(self, model, batch):
        """When x1=None, model should sample it internally."""
        pred_flow, flow = model(batch)
        assert pred_flow.shape == batch.shape
        assert flow.shape == batch.shape

    def test_invalid_combination_raises(self, model, batch):
        with pytest.raises(ValueError):
            model(batch, conditioning=["some text"])


# ---------------------------------------------------------------------------
# FlowModel.valid_input_combination
# ---------------------------------------------------------------------------


class TestInputCombination:
    def test_uncond_model_accepts_none(self, model):
        assert model.valid_input_combination(None) is True

    def test_uncond_model_rejects_conditioning(self, model):
        assert model.valid_input_combination(["text"]) is False


# ---------------------------------------------------------------------------
# FlowModel.sample
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

    def test_batch_size_larger_than_num_samples(self, model):
        samples = model.sample(2, DEVICE, image_size=8, batch_size=16)
        assert samples.shape == (2, 1, 8, 8)

    def test_clamp_output_false_returns_correct_shape(self, model):
        samples = model.sample(2, DEVICE, image_size=8, batch_size=2, clamp_output=False)
        assert samples.shape == (2, 1, 8, 8)

    def test_dynamic_threshold_false(self, model):
        samples = model.sample(2, DEVICE, image_size=8, batch_size=2, dynamic_threshold=False)
        assert samples.shape == (2, 1, 8, 8)

    def test_threshold_coeff_inf_disables_thresholding(self, model):
        samples = model.sample(2, DEVICE, image_size=8, batch_size=2, threshold_coeff=float("inf"))
        assert samples.shape == (2, 1, 8, 8)

    def test_multichannel_output_shape(self):
        """in_channels is used for sampling, not a hardcoded default."""
        m = FlowModel(**{**_KWARGS, "in_channels": 3})
        samples = m.sample(2, DEVICE, image_size=8, batch_size=2)
        assert samples.shape == (2, 3, 8, 8)


# ---------------------------------------------------------------------------
# FlowModel with use_attention
# ---------------------------------------------------------------------------


class TestAttentionBackbone:
    def test_forward_with_attention(self):
        m = FlowModel(**{**_KWARGS, "use_attention": True})
        batch = torch.rand(2, 1, 8, 8)
        pred_flow, flow = m(batch)
        assert pred_flow.shape == batch.shape

    def test_sample_with_attention(self):
        m = FlowModel(**{**_KWARGS, "use_attention": True})
        samples = m.sample(2, DEVICE, image_size=8, batch_size=2)
        assert samples.shape == (2, 1, 8, 8)


# ---------------------------------------------------------------------------
# FlowModel._dynamic_threshold
# ---------------------------------------------------------------------------


class TestDynamicThreshold:
    def test_inf_coeff_returns_input_unchanged(self, model):
        x = torch.randn(2, 1, 8, 8)
        result = model._dynamic_threshold(x, c=float("inf"))
        assert result is x
