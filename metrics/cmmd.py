import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel

import helpers.custom_types as custom_types
from metrics.base import ImageDistributionMetric


class CMMDClip(ImageDistributionMetric):
    name = "CMMD"

    _CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    def __init__(
        self,
        samples: int,
        sigma: float = 10.0,
        device: custom_types.DeviceType = "cuda",
        primary_metric: bool = False,
        factor: float = 1000.0,
    ):
        super().__init__(samples=samples, device=device, primary_metric=primary_metric)
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").eval().to(self.device)
        self.sigma = sigma
        self.factor = factor

    def _encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLIP vision encoder pooler_output for a batch of images in [0, 1]."""
        if x.size(1) != 3:
            x = x.repeat(1, 3, 1, 1)
        x = x.to(self.device)
        x = F.interpolate(x, size=(336, 336), mode="bilinear", align_corners=False)
        mean = torch.tensor(self._CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self._CLIP_STD, device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.clip(pixel_values=x).pooler_output  # [B, 1024]

    def pairwise_distance(self, feat1: torch.Tensor, feat2: torch.Tensor | None = None) -> float:
        # Capture identity before any tensor reassignment so device moves don't break it.
        same_set = feat2 is None
        if same_set:
            feat2 = feat1

        feat1 = feat1.to(self.device)
        feat2 = feat2.to(self.device)

        feat1_sqrd_norm = torch.sum(feat1**2, dim=1, keepdim=True)  # [m, 1]
        feat2_sqrd_norm = torch.sum(feat2**2, dim=1).unsqueeze(0)  # [1, n]
        sqrd_dist = (feat1_sqrd_norm + feat2_sqrd_norm - 2 * feat1 @ feat2.T).clamp(min=0)  # [m, n]
        rbf = torch.exp(-sqrd_dist / (self.sigma**2))

        if same_set:
            # Upper-triangle average (i < j) is equivalent to the full off-diagonal
            # average for a symmetric kernel, but uses half the memory for the mask.
            m = feat1.size(0)
            idx = torch.arange(m, device=self.device)
            mask = idx.unsqueeze(1) < idx.unsqueeze(0)
        else:
            mask = torch.ones(feat1.size(0), feat2.size(0), dtype=torch.bool, device=self.device)

        return (rbf[mask].sum() / mask.sum()).item()

    def _compute(
        self,
        real_loader: torch.utils.data.DataLoader,
        gen_loader: torch.utils.data.DataLoader,
    ) -> float:
        feats_real = self._accumulate_features(real_loader, desc="Collecting feature vectors for real data")
        feats_gen = self._accumulate_features(gen_loader, desc="Collecting feature vectors for generated data")

        dist_real = self.pairwise_distance(feats_real)
        dist_gen = self.pairwise_distance(feats_gen)
        dist_cross = self.pairwise_distance(feats_real, feats_gen)
        return (dist_real + dist_gen - 2 * dist_cross) * self.factor
