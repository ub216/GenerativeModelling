from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3
from tqdm import tqdm

import helpers.custom_types as custom_types


class FIDInception(nn.Module):
    def __init__(self, samples: int, device: custom_types.DeviceType = "cuda"):
        super().__init__()
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()
        self.inception.eval().to(device)
        self.samples = samples

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 3:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            feats = self.inception(x)
        return feats

    def get_dataloader_statistics(
        self,
        dataloader: torch.utils.data.DataLoader,
        desc: str = "Calculating statistics",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Accumulate mean and covariance batch by batch from dataloader."""
        feats = []
        with torch.no_grad():
            total_batches = (
                self.samples + dataloader.batch_size - 1
            ) // dataloader.batch_size
            pbar = tqdm(dataloader, desc=desc, total=total_batches, leave=False)
            num_feats = 0
            for batch in pbar:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                imgs = imgs.to(self.device)
                batch_feats = self.get_features(imgs).cpu()
                feats.append(batch_feats)

                num_feats += batch_feats.size(0)
                pbar.set_postfix({"features": num_feats})
                if num_feats >= self.samples:
                    break
        feats = torch.cat(feats, dim=0).numpy()
        mu, sigma = self.calculate_statistics(feats)
        return mu, sigma

    def calculate_fid(
        self, mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
    ) -> float:
        if isinstance(mu1, np.ndarray):
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        else:
            raise TypeError(f"Incorrect format of mean and covriances to compute fid")
        return float(fid)

    def forward(
        self,
        real_loader: torch.utils.data.DataLoader,
        gen_loader: torch.utils.data.DataLoader,
    ) -> float:
        """Accumulate mean and covariance batch by batch from dataloader."""
        assert isinstance(
            real_loader, torch.utils.data.DataLoader
        ), "real_loader must be a DataLoader"
        assert isinstance(
            gen_loader, torch.utils.data.DataLoader
        ), "gen_loader must be a DataLoader"

        mu_real, sigma_real = self.get_dataloader_statistics(
            real_loader, desc="Calculating real data statistics"
        )
        mu_gen, sigma_gen = self.get_dataloader_statistics(
            gen_loader, desc="Calculating generated data statistics"
        )
        fid = self.calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
        return fid

    def calculate_statistics(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(feats, np.ndarray):
            mu = np.mean(feats, axis=0)
            sigma = np.cov(feats, rowvar=False)
        else:
            raise TypeError(f"Incorrect format of mean and covriances to compute fid")
        return mu, sigma
