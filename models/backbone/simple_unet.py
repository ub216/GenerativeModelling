from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

import helpers.custom_types as custom_types
from models.backbone.residual_conv import ResidualConv
from models.text_model import TextModel
from models.utils import sinusoidal_embedding


# -------------------------
# Small Residual UNet
# -------------------------
class SimpleUNet(nn.Module):
    """
    Small UNet variant with residual blocks, timestep embeddings, and up/down sampling.
    Predicts the noise epsilon.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4],
        time_emb_dim: int = 128,
        text_emb_dim: Optional[int] = None,
        device: custom_types.DeviceType = "cuda",
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        if text_emb_dim is not None:
            tm = TextModel(device)
            self.text_model = nn.Sequential(tm, nn.Linear(tm.dim, time_emb_dim))

        # Encoder
        self.encs = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.encs.append(
                ResidualConv(
                    ch, out_ch, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim
                )
            )
            self.downs.append(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )  # downsample
            ch = out_ch

        # Bottleneck
        self.mid1 = ResidualConv(
            ch, ch, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim
        )
        self.mid2 = ResidualConv(
            ch, ch, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim
        )

        # Decoder
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        rev_mults = list(channel_mults)[::-1]
        for mult in rev_mults:
            out_ch = base_channels * mult
            self.ups.append(
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1)
            )  # upsample x2
            self.decs.append(
                ResidualConv(
                    out_ch * 2,
                    out_ch,
                    time_emb_dim=time_emb_dim,
                    text_emb_dim=text_emb_dim,
                )
            )
            ch = out_ch

        self.out_norm = nn.GroupNorm(8, ch) if ch >= 8 else nn.BatchNorm2d(ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(
            ch, in_channels, kernel_size=3, padding=1
        )  # predict noise

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: Optional[List[str]] = None,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, C, H, W) noisy image
        timesteps: (B,) long
        conditioning: (B,)
        text_emb: (B, D) pre-computed embeddings for the conditioning.
        return: (B, C, H, W)
        """
        time_emb = sinusoidal_embedding(timesteps, self.time_mlp[0].in_features)
        time_emb = self.time_mlp(time_emb)
        # Compute text_emb on the conditioning only if they haven't
        # been computed before. Else use the ones that are provided
        # as input. Helps reduce computation while sampling
        text_emb = (
            self.text_model(conditioning)
            if text_emb is None
            and conditioning is not None
            and self.text_model is not None
            else text_emb
        )

        hs = []
        h = self.in_conv(x)
        # encode
        for enc, down in zip(self.encs, self.downs):
            h = enc(h, time_emb, text_emb)
            hs.append(h)
            h = down(h)

        # bottleneck
        h = self.mid1(h, time_emb, text_emb)
        h = self.mid2(h, time_emb, text_emb)

        # decode
        for up, dec, skip in zip(self.ups, self.decs, reversed(hs)):
            h = up(h)
            # if shapes mismatch due to odd sizes, center-crop or pad (here we assume divisible by 2**len(channel_mults))
            if h.shape[-2:] != skip.shape[-2:]:
                # naive center crop/pad to match
                diff_y = skip.shape[-2] - h.shape[-2]
                diff_x = skip.shape[-1] - h.shape[-1]
                h = F.pad(h, (0, max(0, diff_x), 0, max(0, diff_y)))
                h = h[:, :, : skip.shape[-2], : skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = dec(h, time_emb, text_emb)

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h), text_emb
