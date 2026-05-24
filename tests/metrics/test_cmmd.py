from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from metrics.cmmd import CMMDClip


def _loader(n: int = 32, batch_size: int = 8) -> DataLoader:
    return DataLoader(TensorDataset(torch.rand(n, 3, 8, 8)), batch_size=batch_size)


def _mock_clip_output(batch_size: int, feat_dim: int = 1024) -> MagicMock:
    out = MagicMock()
    out.pooler_output = torch.randn(batch_size, feat_dim)
    return out


@pytest.fixture
def cmmd():
    """CMMDClip with CLIPVisionModel mocked out to avoid model download."""
    with patch("metrics.cmmd.CLIPVisionModel") as mock_clip:
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_clip.from_pretrained.return_value = mock_model
        metric = CMMDClip(samples=32, device="cpu")
    metric.clip.side_effect = lambda **kw: _mock_clip_output(kw["pixel_values"].size(0))
    return metric


class TestEncodeBatch:
    def test_output_shape_rgb(self, cmmd):
        feats = cmmd._encode_batch(torch.rand(4, 3, 64, 64))
        assert feats.shape == (4, 1024)

    def test_grayscale_expanded_to_rgb(self, cmmd):
        feats = cmmd._encode_batch(torch.rand(4, 1, 64, 64))
        assert feats.shape == (4, 1024)

    def test_output_on_device(self, cmmd):
        feats = cmmd._encode_batch(torch.rand(4, 3, 64, 64))
        assert feats.device.type == cmmd.device


class TestPairwiseDistance:
    def test_self_comparison_returns_float(self, cmmd):
        assert isinstance(cmmd.pairwise_distance(torch.randn(8, 32)), float)

    def test_cross_comparison_returns_float(self, cmmd):
        assert isinstance(cmmd.pairwise_distance(torch.randn(8, 32), torch.randn(8, 32)), float)

    def test_rbf_output_in_unit_interval(self, cmmd):
        # RBF kernel is bounded in [0, 1] so any average is too
        assert 0.0 <= cmmd.pairwise_distance(torch.randn(8, 32)) <= 1.0

    def test_cross_term_in_unit_interval(self, cmmd):
        assert 0.0 <= cmmd.pairwise_distance(torch.randn(8, 32), torch.randn(8, 32)) <= 1.0

    def test_identical_features_gives_one(self, cmmd):
        # k(x, x) = exp(0/σ²) = 1 for all off-diagonal pairs → average = 1
        assert abs(cmmd.pairwise_distance(torch.zeros(8, 32)) - 1.0) < 1e-5

    def test_cross_term_is_symmetric(self, cmmd):
        torch.manual_seed(42)
        feats1, feats2 = torch.randn(8, 32), torch.randn(8, 32)
        d_xy = cmmd.pairwise_distance(feats1, feats2)
        d_yx = cmmd.pairwise_distance(feats2, feats1)
        assert abs(d_xy - d_yx) < 1e-5


class TestCompute:
    def test_returns_float(self, cmmd):
        feats = torch.randn(32, 64)
        with patch.object(cmmd, "_accumulate_features", return_value=feats):
            assert isinstance(cmmd._compute(_loader(), _loader()), float)

    def test_larger_for_different_distributions(self, cmmd):
        # Distributions far apart in feature space should yield a higher score
        feats_a = torch.randn(32, 64)
        feats_b = torch.zeros(32, 64) + 100.0  # k(feats_a, feats_b) ≈ 0

        with patch.object(cmmd, "_accumulate_features") as mock_acc:
            mock_acc.side_effect = [feats_a, feats_a.clone()]
            score_similar = cmmd._compute(_loader(), _loader())

        with patch.object(cmmd, "_accumulate_features") as mock_acc:
            mock_acc.side_effect = [feats_a, feats_b]
            score_diff = cmmd._compute(_loader(), _loader())

        assert score_diff > score_similar

    def test_factor_scales_output(self):
        with patch("metrics.cmmd.CLIPVisionModel") as mock_clip:
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_clip.from_pretrained.return_value = mock_model
            cmmd_1x = CMMDClip(samples=32, device="cpu", factor=1.0)
            cmmd_1000x = CMMDClip(samples=32, device="cpu", factor=1000.0)

        feats_real = torch.zeros(32, 32)
        feats_gen = torch.ones(32, 32)

        with patch.object(cmmd_1x, "_accumulate_features") as m:
            m.side_effect = [feats_real.clone(), feats_gen.clone()]
            score_1x = cmmd_1x._compute(_loader(), _loader())

        with patch.object(cmmd_1000x, "_accumulate_features") as m:
            m.side_effect = [feats_real.clone(), feats_gen.clone()]
            score_1000x = cmmd_1000x._compute(_loader(), _loader())

        assert abs(score_1000x - 1000.0 * score_1x) < 1e-3
