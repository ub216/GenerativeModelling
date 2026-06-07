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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Image dimensions must be divisible by patch size ({self.patch_size}), but got {H}x{W}")
        x = self.proj(x)  # (B, embed_dim, H//P, W//P)
        return x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
