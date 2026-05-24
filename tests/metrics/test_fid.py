from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from metrics.fid import FIDInception


def _loader(n: int = 32, batch_size: int = 8) -> DataLoader:
    return DataLoader(TensorDataset(torch.rand(n, 3, 8, 8)), batch_size=batch_size)


@pytest.fixture
def fid():
    """FIDInception with inception_v3 mocked out to avoid model download."""
    with patch("metrics.fid.inception_v3") as mock_iv3:
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_iv3.return_value = mock_model
        metric = FIDInception(samples=32, device="cpu")
    metric.inception.side_effect = lambda x: torch.randn(x.size(0), 2048)
    return metric


class TestCalculateStatistics:
    def test_mu_shape(self, fid):
        mu, _ = fid.calculate_statistics(np.random.randn(64, 32))
        assert mu.shape == (32,)

    def test_sigma_shape(self, fid):
        _, sigma = fid.calculate_statistics(np.random.randn(64, 32))
        assert sigma.shape == (32, 32)

    def test_mu_matches_numpy_mean(self, fid):
        feats = np.random.randn(64, 32)
        mu, _ = fid.calculate_statistics(feats)
        np.testing.assert_allclose(mu, np.mean(feats, axis=0))

    def test_sigma_is_symmetric(self, fid):
        _, sigma = fid.calculate_statistics(np.random.randn(64, 32))
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)


class TestCalculateFID:
    def test_zero_for_identical_distributions(self, fid):
        feats = np.random.randn(128, 32)
        mu, sigma = fid.calculate_statistics(feats)
        assert abs(fid.calculate_fid(mu, sigma, mu, sigma)) < 1e-4

    def test_positive_for_shifted_distributions(self, fid):
        rng = np.random.default_rng(0)
        mu1, s1 = fid.calculate_statistics(rng.standard_normal((128, 32)))
        mu2, s2 = fid.calculate_statistics(rng.standard_normal((128, 32)) + 5.0)
        assert fid.calculate_fid(mu1, s1, mu2, s2) > 0

    def test_symmetric(self, fid):
        rng = np.random.default_rng(1)
        mu1, s1 = fid.calculate_statistics(rng.standard_normal((128, 32)))
        mu2, s2 = fid.calculate_statistics(rng.standard_normal((128, 32)) + 2.0)
        assert abs(fid.calculate_fid(mu1, s1, mu2, s2) - fid.calculate_fid(mu2, s2, mu1, s1)) < 1e-4


class TestEncodeBatch:
    def test_output_shape_rgb(self, fid):
        assert fid._encode_batch(torch.rand(4, 3, 64, 64)).shape == (4, 2048)

    def test_grayscale_expanded_to_rgb(self, fid):
        # grayscale (1 channel) must be broadcast to 3 channels before inception
        assert fid._encode_batch(torch.rand(4, 1, 64, 64)).shape == (4, 2048)


class TestCompute:
    def test_returns_float(self, fid):
        feats = torch.from_numpy(np.random.randn(32, 512).astype(np.float32))
        with patch.object(fid, "_accumulate_features", return_value=feats):
            assert isinstance(fid._compute(_loader(), _loader()), float)

    def test_near_zero_for_identical_features(self, fid):
        # Same features for both distributions → FID ≈ 0
        feats = torch.from_numpy(np.random.randn(128, 64).astype(np.float32))
        with patch.object(fid, "_accumulate_features", return_value=feats):
            assert abs(fid._compute(_loader(), _loader())) < 1.0

    def test_larger_for_different_distributions(self, fid):
        feats_a = torch.from_numpy(np.random.randn(128, 64).astype(np.float32))
        feats_b = torch.from_numpy((np.random.randn(128, 64) + 10.0).astype(np.float32))

        with patch.object(fid, "_accumulate_features", return_value=feats_a):
            score_same = fid._compute(_loader(), _loader())

        with patch.object(fid, "_accumulate_features") as mock_acc:
            mock_acc.side_effect = [feats_a, feats_b]
            score_diff = fid._compute(_loader(), _loader())

        assert score_diff > score_same
