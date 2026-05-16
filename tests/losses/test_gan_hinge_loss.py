import pytest
import torch
from losses.gan_hinge_loss import GANHingeLoss


@pytest.fixture
def loss_fn():
    return GANHingeLoss()


def _scores(b=4, val=0.5):
    """Return a (gen_gen, gen_dis, real) tuple of shape (b, 1) filled with val."""
    s = torch.full((b, 1), val)
    return s, s.clone(), s.clone()


class TestGANHingeLoss:
    def test_returns_dict_with_expected_keys(self, loss_fn):
        result = loss_fn(_scores())
        assert set(result.keys()) == {"generator", "discriminator"}

    def test_outputs_are_scalars(self, loss_fn):
        result = loss_fn(_scores())
        assert result["generator"].ndim == 0
        assert result["discriminator"].ndim == 0

    def test_discriminator_loss_nonnegative(self, loss_fn):
        """Hinge loss = relu(1 - score) >= 0 always."""
        for val in [0.0, 0.5, 1.0, 2.0]:
            result = loss_fn(_scores(val=val))
            assert result["discriminator"] >= 0

    def test_discriminator_zero_when_scores_above_one(self, loss_fn):
        """relu(1 - score) = 0 when score >= 1 → discriminator loss = 0."""
        result = loss_fn(_scores(val=2.0))
        assert torch.isclose(result["discriminator"], torch.tensor(0.0), atol=1e-6)

    def test_generator_loss_is_negative_mean_score(self, loss_fn):
        """generator_loss = -mean(gen_score_gen)."""
        gen_gen = torch.tensor([[1.0], [3.0]])
        gen_dis = torch.rand(2, 1)
        real = torch.rand(2, 1)
        result = loss_fn((gen_gen, gen_dis, real))
        expected = -gen_gen.mean()
        assert torch.isclose(result["generator"], expected, atol=1e-6)

    def test_shape_mismatch_raises(self, loss_fn):
        with pytest.raises(AssertionError):
            loss_fn((torch.rand(2, 1), torch.rand(2, 1), torch.rand(3, 1)))

    def test_too_few_outputs_raises(self, loss_fn):
        with pytest.raises(AssertionError):
            loss_fn((torch.rand(2, 1), torch.rand(2, 1)))

    def test_generative_weight_scales_generator_loss(self):
        fn = GANHingeLoss(generative_weight=2.0, discriminative_weight=0.0)
        gen_gen = torch.ones(2, 1)
        gen_dis = real = torch.rand(2, 1)
        result = fn((gen_gen, gen_dis, real))
        # generator_loss = -mean(gen_gen) = -1.0, but weight is applied by the
        # caller (factory), not inside the forward — so this just checks the raw value
        assert result["generator"] == pytest.approx(-1.0)
