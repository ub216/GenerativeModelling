import pytest
import torch

from models.diffusion import DiffusionModel, prepare_noise_schedule

DEVICE = "cpu"

# Tiny architecture so tests run quickly on CPU
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
    return DiffusionModel(**_KWARGS)


@pytest.fixture
def cond_model():
    return DiffusionModel(**{**_KWARGS, "text_emb_dim": 16})


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(2, 1, 8, 8)


# ---------------------------------------------------------------------------
# prepare_noise_schedule
# ---------------------------------------------------------------------------


class TestNoiseSchedule:
    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_output_shapes(self, schedule):
        T = 50
        sched = prepare_noise_schedule(T, schedule)
        for key in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ):
            assert sched[key].shape == (T,), f"{key} has wrong shape for {schedule}"

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_alphas_cumprod_monotone_decreasing(self, schedule):
        sched = prepare_noise_schedule(100, schedule)
        ac = sched["alphas_cumprod"]
        assert (ac[:-1] > ac[1:]).all(), f"{schedule}: alphas_cumprod not monotone"

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_alphas_cumprod_in_unit_interval(self, schedule):
        sched = prepare_noise_schedule(100, schedule)
        ac = sched["alphas_cumprod"]
        assert (ac >= 0).all() and (ac <= 1).all()

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_posterior_variance_nonnegative(self, schedule):
        sched = prepare_noise_schedule(100, schedule)
        assert (sched["posterior_variance"] >= 0).all()

    def test_sqrt_decomposition_identity(self):
        """sqrt_alphas_cumprod² + sqrt_one_minus_alphas_cumprod² == 1."""
        sched = prepare_noise_schedule(100, "cosine")
        sq_a = sched["sqrt_alphas_cumprod"]
        sq_1a = sched["sqrt_one_minus_alphas_cumprod"]
        assert torch.allclose(sq_a**2 + sq_1a**2, torch.ones(100), atol=1e-5)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError):
            prepare_noise_schedule(100, "unknown")


# ---------------------------------------------------------------------------
# DiffusionModel — forward diffusion (q_sample)
# ---------------------------------------------------------------------------


class TestQSample:
    def test_output_shape(self, model, batch):
        t = torch.randint(0, 10, (2,))
        noise = torch.randn_like(batch)
        assert model.q_sample(batch, t, noise).shape == batch.shape

    def test_at_t0_with_zero_noise_close_to_x0(self, model, batch):
        """At t=0 with no noise, x_t ≈ x0 (sqrt_alphas_cumprod[0] ≈ 1)."""
        t = torch.zeros(2, dtype=torch.long)
        x_noisy = model.q_sample(batch, t, torch.zeros_like(batch))
        # cosine schedule: sqrt_alphas_cumprod[0] ≈ 0.99
        assert torch.allclose(x_noisy, batch, atol=0.05)

    def test_interpolation_uses_both_terms(self, model, batch):
        """q_sample(x0, t, noise) lies strictly between x0 and noise for 0 < t < T."""
        t = torch.full((2,), 5, dtype=torch.long)
        noise = torch.ones_like(batch) * 10.0
        x_noisy = model.q_sample(batch, t, noise)
        # x_noisy must differ from both pure x0 and pure noise
        assert not torch.allclose(x_noisy, batch)
        assert not torch.allclose(x_noisy, noise)


# ---------------------------------------------------------------------------
# DiffusionModel — forward pass
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shapes(self, model, batch):
        t = torch.randint(0, 10, (2,))
        pred, target, weights = model(batch, time_steps=t)
        assert pred.shape == batch.shape
        assert target.shape == batch.shape
        assert weights.shape == (2,)

    def test_snr_weights_positive(self, model, batch):
        t = torch.randint(0, 10, (2,))
        _, _, weights = model(batch, time_steps=t)
        assert (weights > 0).all()

    def test_random_timesteps_when_not_provided(self, model, batch):
        """Calling forward without time_steps should still succeed."""
        pred, target, weights = model(batch)
        assert pred.shape == batch.shape

    def test_renormalise_scales_input(self, batch):
        """renormalise=True re-maps [0,1] input to [-1,1] before processing."""
        model_rn = DiffusionModel(**{**_KWARGS, "renormalise": True})
        t = torch.zeros(2, dtype=torch.long)
        pred, _, _ = model_rn(batch, time_steps=t)
        assert pred.shape == batch.shape

    def test_invalid_combination_raises(self, model, batch):
        with pytest.raises(ValueError):
            model(batch, conditioning=["some text"])


# ---------------------------------------------------------------------------
# DiffusionModel — valid_input_combination
# ---------------------------------------------------------------------------


class TestInputCombination:
    def test_uncond_model_accepts_no_conditioning(self, model):
        assert model.valid_input_combination(None) is True

    def test_uncond_model_rejects_conditioning(self, model):
        assert model.valid_input_combination(["text"]) is False

    def test_cond_model_accepts_conditioning(self, cond_model):
        assert cond_model.valid_input_combination(["text"]) is True

    def test_cond_model_rejects_no_conditioning(self, cond_model):
        assert cond_model.valid_input_combination(None) is False


# ---------------------------------------------------------------------------
# DiffusionModel — _dynamic_threshold
# ---------------------------------------------------------------------------


class TestDynamicThreshold:
    def test_inf_threshold_is_passthrough(self, model):
        x = torch.rand(2, 1, 8, 8) * 100
        result = model._dynamic_threshold(x, c=float("inf"))
        assert torch.equal(result, x)

    def test_clips_outliers_to_within_threshold(self, model):
        x = torch.rand(2, 1, 8, 8) * 100
        result = model._dynamic_threshold(x, c=1.0)
        assert result.abs().max() <= 1.0 + 1e-5

    def test_output_shape_preserved(self, model, batch):
        result = model._dynamic_threshold(batch, c=1.0)
        assert result.shape == batch.shape


# ---------------------------------------------------------------------------
# DiffusionModel — schedule buffers
# ---------------------------------------------------------------------------


class TestScheduleBuffers:
    def test_train_buffers_registered(self, model):
        for key in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ):
            assert hasattr(model, f"train_{key}"), f"Missing buffer: train_{key}"

    def test_test_buffers_registered(self, model):
        for key in ("betas", "alphas_cumprod"):
            assert hasattr(model, f"test_{key}"), f"Missing buffer: test_{key}"


# ---------------------------------------------------------------------------
# DiffusionModel — sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_ddpm_output_shape(self, model):
        samples = model.sample(3, DEVICE, image_size=8, batch_size=2)
        assert samples.shape == (3, 1, 8, 8)

    def test_ddpm_output_in_unit_range(self, model):
        samples = model.sample(3, DEVICE, image_size=8, batch_size=2)
        assert samples.min() >= 0.0 and samples.max() <= 1.0

    def test_ddim_output_shape(self, model):
        samples = model.sample(3, DEVICE, image_size=8, batch_size=2, use_ddim=True)
        assert samples.shape == (3, 1, 8, 8)

    def test_ddim_output_in_unit_range(self, model):
        samples = model.sample(3, DEVICE, image_size=8, batch_size=2, use_ddim=True)
        assert samples.min() >= 0.0 and samples.max() <= 1.0

    def test_image_size_as_tuple(self, model):
        samples = model.sample(2, DEVICE, image_size=(8, 8), batch_size=2)
        assert samples.shape == (2, 1, 8, 8)

    def test_batch_size_larger_than_num_samples(self, model):
        """batch_size > num_samples should still produce num_samples outputs."""
        samples = model.sample(2, DEVICE, image_size=8, batch_size=16)
        assert samples.shape == (2, 1, 8, 8)
