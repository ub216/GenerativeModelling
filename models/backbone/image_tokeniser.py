from typing import Tuple

import torch
import torch.nn as nn


class ImageTokeniser(nn.Module):
    """Converts images into sequences of patch tokens, similar to ViT.
    Each patch is flattened and projected to an embedding dimension.
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.unproj = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Image dimensions must be divisible by patch size ({self.patch_size}), but got {H}x{W}")
        x = self.proj(x)  # (B, embed_dim, H//P, W//P)
        return x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

    def unpatchify(self, x: torch.Tensor, image_size: Tuple[int, int] | int) -> torch.Tensor:
        """x: (B, N, embed_dim)
        returns: (B, C, H, W)
        """
        B, N, D = x.shape
        P = self.patch_size

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        H, W = image_size
        if N != (H // P) * (W // P):
            raise ValueError(
                f"Number of tokens {N} does not match expected {(H//P)*(W//P)} for image size {H} and patch size {P}"
            )
        x = x.transpose(1, 2).view(B, D, H // P, W // P)  # (B, embed_dim, H//P, W//P)
        return self.unproj(x)  # (B, in_channels, H, W)
