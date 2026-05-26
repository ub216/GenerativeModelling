from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from tqdm import tqdm

import helpers.custom_types as custom_types


class ImageDistributionMetric(nn.Module, ABC):
    """Base class for metrics that compare a real and a generated image distribution.

    Subclasses must define:
      - ``name``  — class-level string used for logging (e.g. "FID", "CMMD")
      - ``_encode_batch`` — extract features from a batch of [0, 1] images
      - ``_compute`` — compute the scalar metric from two DataLoaders
    """

    name: str

    def __init__(
        self,
        samples: int,
        device: custom_types.DeviceType = "cuda",
        primary_metric: bool = False,
    ):
        super().__init__()
        self.samples = samples
        self.device = device
        self.primary_metric = primary_metric

    @abstractmethod
    def _encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Feature-extract a batch of [0, 1] images. Must return features on self.device."""
        pass

    @torch.no_grad()
    def _accumulate_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        desc: str = "Collecting features",
    ) -> torch.Tensor:
        """Collect encoded features from the dataloader for up to self.samples images."""
        feats = []
        total_batches = (self.samples + dataloader.batch_size - 1) // dataloader.batch_size
        pbar = tqdm(dataloader, desc=desc, total=total_batches, leave=False)
        num_feats = 0
        for batch in pbar:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_feats = self._encode_batch(imgs).cpu()
            feats.append(batch_feats)
            num_feats += batch_feats.size(0)
            pbar.set_postfix({"features": num_feats})
            if num_feats >= self.samples:
                break
        return torch.cat(feats, dim=0)

    @abstractmethod
    def _compute(
        self,
        real_loader: torch.utils.data.DataLoader,
        gen_loader: torch.utils.data.DataLoader,
    ) -> float:
        """Compute the metric given dataloaders for real and generated images."""
        pass

    def forward(
        self,
        real_loader: torch.utils.data.DataLoader,
        gen_loader: torch.utils.data.DataLoader,
    ) -> float:
        if not isinstance(real_loader, torch.utils.data.DataLoader):
            raise TypeError(f"real_loader must be a DataLoader, got {type(real_loader).__name__}")
        if not isinstance(gen_loader, torch.utils.data.DataLoader):
            raise TypeError(f"gen_loader must be a DataLoader, got {type(gen_loader).__name__}")
        return self._compute(real_loader, gen_loader)
