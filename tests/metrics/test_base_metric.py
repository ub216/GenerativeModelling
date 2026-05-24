import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from metrics.base import ImageDistributionMetric


class _DummyMetric(ImageDistributionMetric):
    """Minimal concrete subclass for testing shared base-class logic."""

    name = "dummy"
    feat_dim = 16

    def _encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.size(0), self.feat_dim)

    def _compute(self, real_loader, gen_loader) -> float:
        return 0.0


def _loader(n: int = 32, batch_size: int = 8, as_tuple: bool = False) -> DataLoader:
    imgs = torch.rand(n, 3, 8, 8)
    if as_tuple:
        ds = TensorDataset(imgs, torch.zeros(n, dtype=torch.long))
    else:
        ds = TensorDataset(imgs)
    return DataLoader(ds, batch_size=batch_size)


class TestAccumulateFeatures:
    @pytest.fixture
    def metric(self):
        return _DummyMetric(samples=20, device="cpu")

    def test_returns_tensor(self, metric):
        feats = metric._accumulate_features(_loader())
        assert isinstance(feats, torch.Tensor)

    def test_feature_dim_matches_encoder(self, metric):
        feats = metric._accumulate_features(_loader())
        assert feats.size(1) == _DummyMetric.feat_dim

    def test_output_on_cpu(self, metric):
        feats = metric._accumulate_features(_loader())
        assert feats.device.type == "cpu"

    def test_stops_at_sample_boundary(self, metric):
        # samples=20, batch_size=8 → breaks after batch 3 (24 images accumulated)
        feats = metric._accumulate_features(_loader(n=64, batch_size=8))
        assert metric.samples <= feats.size(0) < metric.samples + 8

    def test_handles_tuple_batch(self, metric):
        # DataLoader returning (imgs, labels): only imgs should reach _encode_batch
        feats = metric._accumulate_features(_loader(as_tuple=True))
        assert feats.size(1) == _DummyMetric.feat_dim

    def test_handles_bare_tensor_batch(self, metric):
        # DataLoader wrapping a plain tensor (no TensorDataset)
        loader = DataLoader(torch.rand(32, 3, 8, 8), batch_size=8)
        feats = metric._accumulate_features(loader)
        assert feats.size(1) == _DummyMetric.feat_dim


class TestForwardValidation:
    @pytest.fixture
    def metric(self):
        return _DummyMetric(samples=10, device="cpu")

    def test_rejects_tensor_as_real_loader(self, metric):
        with pytest.raises(AssertionError):
            metric(torch.rand(10, 3, 8, 8), _loader())

    def test_rejects_tensor_as_gen_loader(self, metric):
        with pytest.raises(AssertionError):
            metric(_loader(), torch.rand(10, 3, 8, 8))

    def test_accepts_two_dataloaders_and_returns_float(self, metric):
        result = metric(_loader(), _loader())
        assert isinstance(result, float)
