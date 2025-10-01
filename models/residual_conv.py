import torch
from torch import nn

class ResidualConv(nn.Module):
    """
    Residual block for encoder: Conv -> Norm -> Act (+time_embedding) -> Conv + Norm + skip
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout=0.0,
        norm=True,
        time_emb_dim=None,
        bias=True,
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
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(
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

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

        # match dimensions for residual
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=bias
            )

    def forward(self, x, t_emb=None):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out) if self.norm1 is not None else out
        out = self.act(out)
        out = self.dropout(out)
        # add time embedding
        if (self.time_mlp is not None) and (t_emb is not None):
            # t_emb is (B, time_emb_dim) -> project & add across spatial dims
            t_proj = self.time_mlp(t_emb)  # (B, out_ch)
            out = out + t_proj[:, :, None, None]

        out = self.conv2(out)
        out = self.norm2(out) if self.norm2 is not None else out
        out += identity
        out = self.act(out)
        return out


class ResidualDeconv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        dropout=0.0,
        norm=True,
        bias=True,
        final_layer=False,
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
        self.act = nn.SiLU()
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
