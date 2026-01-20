from typing import Optional

import torch
from loguru import logger
from torch import nn


class ResidualConv(nn.Module):
    """
    Residual block for encoder: Conv -> Norm -> Act (+time_embedding) -> Conv + Norm + skip
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
        time_emb_dim: Optional[int] = None,
        text_emb_dim: Optional[int] = None,
        bias: bool = True,
        activation: nn.Module = nn.SiLU(),
        spectral_norm: bool = False,
        affine: bool = False,
    ):
        super().__init__()
        # standard Convolutions
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=bias
        )

        # for GAN stability
        if spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)

        # normalisation layers
        self.norm = nn.GroupNorm(1, out_channels, affine=affine)

        self.act = activation
        self.dropout = nn.Dropout2d(dropout)

        # conditioning MLP
        # It takes (time + text) and predicts (Scale + Shift) for the norm layer
        self.cond_dim = (time_emb_dim or 0) + (text_emb_dim or 0)
        if self.cond_dim > 0:
            self.condition_projection = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.cond_dim, out_channels * 2),  # *2 for Scale and Shift
            )
        else:
            self.condition_projection = None

        # 4. Skip connection
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride, bias=bias
            )

        self.intialise_zero()

    def intialise_zero(self):
        nn.init.zeros_(self.condition_projection[-1].weight)
        nn.init.zeros_(self.condition_projection[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        identity = self.skip(x)

        # First conv
        out = self.conv1(x)

        # Apply Normalization
        out = self.norm(out)

        # Apply Adaptive Conditioning
        if self.condition_projection is not None:
            # Combine embeddings
            cond = []
            if time_emb is not None:
                cond.append(time_emb)
            if text_emb is not None:
                cond.append(text_emb)

            if cond:
                # Predict Scale (gamma) and Shift (beta)
                combined_cond = torch.cat(cond, dim=1)
                ada_params = self.condition_projection(combined_cond)
                gamma, beta = ada_params.chunk(2, dim=1)

                # Apply: out = out * (1 + gamma) + beta
                # We unsqueeze gamma/beta to (B, C, 1, 1) to match (B, C, H, W)
                out = out * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += identity
        return self.act(out)


class ResidualDeconv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        dropout: float = 0.0,
        norm: bool = True,
        bias: bool = True,
        final_layer: bool = False,
        activation: nn.modules = nn.SiLU(),
    ):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=1,
            bias=bias,
        )
        self.act = activation
        self.dropout = nn.Dropout2d(dropout)
        self.deconv2 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.norm1 = self.norm2 = None
        if norm:
            self.norm1 = (
                nn.GroupNorm(8, out_channels)
                if out_channels >= 8
                else nn.BatchNorm2d(out_channels)
            )
            self.norm2 = (
                nn.GroupNorm(8, out_channels)
                if out_channels >= 8
                else nn.BatchNorm2d(out_channels)
            )

        # skip connection
        self.skip = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            output_padding=1,
            bias=bias,
        )

        self.final_layer = final_layer

    def forward(self, x):
        identity = self.skip(x)
        out = self.deconv1(x)
        out = self.norm1(out) if self.norm1 is not None else out
        out = self.act(out)
        out = self.dropout(out)
        out = self.deconv2(out)
        out = self.norm2(out) if self.norm2 is not None else out
        out += identity
        out = self.act(out) if not self.final_layer else torch.sigmoid(out)
        return out
