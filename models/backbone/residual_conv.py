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
        norm: bool = True,
        time_emb_dim: Optional[int] = None,
        text_emb_dim: Optional[int] = None,
        bias: bool = True,
        activation: nn.Module = nn.SiLU(),
        spectral_norm=False,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )

        self.act = activation
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        if spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)

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

        self.time_mlp = None
        self.text_mlp = None
        if time_emb_dim is not None and text_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, out_channels // 2)
            )
            self.text_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(text_emb_dim, out_channels // 2)
            )
        elif time_emb_dim is not None and text_emb_dim is None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, out_channels)
            )
        else:
            logger.warning("Invalid embedding configuration requested")

        # match dimensions for residual
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=bias
            )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out) if self.norm1 is not None else out
        out = self.act(out)
        out = self.dropout(out)
        # add time embedding
        if (
            (self.time_mlp is not None)
            and (time_emb is not None)
            and (self.text_mlp is not None)
            and (text_emb is not None)
        ):
            text_proj = self.text_mlp(text_emb)
            time_proj = self.time_mlp(time_emb)
            t_proj = torch.cat([time_proj, text_proj], dim=1)
            out = out + t_proj[:, :, None, None]
        elif (self.time_mlp is not None) and (time_emb is not None):
            time_proj = self.time_mlp(time_emb)  # (B, out_ch)
            out = out + time_proj[:, :, None, None]

        out = self.conv2(out)
        out = self.norm2(out) if self.norm2 is not None else out
        out += identity
        out = self.act(out)
        return out


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
