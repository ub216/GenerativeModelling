from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3

import helpers.custom_types as custom_types
from metrics.base import ImageDistributionMetric


class FIDInception(ImageDistributionMetric):
    name = "FID"

    _INCEPTION_MEAN = np.array([0.485, 0.456, 0.406])
    _INCEPTION_STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, samples: int, device: custom_types.DeviceType = "cuda", primary_metric: bool = False):
        super().__init__(samples=samples, device=device, primary_metric=primary_metric)
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()
        self.inception.eval().to(device)

    def _encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 3:
            x = x.repeat(1, 3, 1, 1)
        x = x.to(self.device)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor(self._INCEPTION_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self._INCEPTION_STD, device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.inception(x)

    def calculate_statistics(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.mean(feats, axis=0), np.cov(feats, rowvar=False)

    def calculate_fid(self, mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

    def _compute(
        self,
        real_loader: torch.utils.data.DataLoader,
        gen_loader: torch.utils.data.DataLoader,
    ) -> float:
        feats_real = self._accumulate_features(real_loader, desc="Calculating real data statistics").numpy()
        feats_gen = self._accumulate_features(gen_loader, desc="Calculating generated data statistics").numpy()
        mu_real, sigma_real = self.calculate_statistics(feats_real)
        mu_gen, sigma_gen = self.calculate_statistics(feats_gen)
        return self.calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
