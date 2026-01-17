from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

import helpers.custom_types as custom_types
from models.residual_conv import ResidualConv
from models.text_model import TextModel
from models.utils import sinusoidal_embedding


# -------------------------
# Simple Residual UNet
# -------------------------
class SimpleUNet(nn.Module):
    """
    Refactored UNet with support for variable blocks per resolution stage.
    Optimized for complex datasets like CelebA.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4],
        num_blocks: Union[int, List[int]] = [1, 2, 2],
        time_emb_dim: int = 128,
        text_emb_dim: Optional[int] = None,
        device: custom_types.DeviceType = "cuda",
    ):
        super().__init__()
        
        # Normalize num_blocks to a list matching the number of stages
        if isinstance(num_blocks, int):
            self.num_blocks_list = [num_blocks] * len(channel_mults)
        else:
            if len(num_blocks) != len(channel_mults):
                raise ValueError("num_blocks list must match length of channel_mults")
            self.num_blocks_list = num_blocks

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        if text_emb_dim is not None:
            tm = TextModel(device)
            self.text_model = nn.Sequential(tm, nn.Linear(tm.dim, time_emb_dim))
        else:
            self.text_model = None

        # Encoder
        self.encs = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            # Create a stage consisting of multiple residual blocks
            stage_blocks = nn.ModuleList()
            for b in range(self.num_blocks_list[i]):
                # First block handles channel transition, subsequent blocks maintain out_ch
                block_in = ch if b == 0 else out_ch
                stage_blocks.append(
                    ResidualConv(block_in, out_ch, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim)
                )
            
            self.encs.append(stage_blocks)
            self.downs.append(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            ch = out_ch

        # Bottleneck
        self.mid1 = ResidualConv(ch, ch, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim)
        self.mid2 = ResidualConv(ch, ch, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim)

        # Decoder
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        rev_mults = list(channel_mults)[::-1]
        rev_blocks = list(self.num_blocks_list)[::-1]
        
        for i, mult in enumerate(rev_mults):
            out_ch = base_channels * mult
            self.ups.append(
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            
            stage_blocks = nn.ModuleList()
            for b in range(rev_blocks[i]):
                # First block handles concatenated skip connection (out_ch * 2)
                block_in = out_ch * 2 if b == 0 else out_ch
                stage_blocks.append(
                    ResidualConv(block_in, out_ch, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim)
                )
            self.decs.append(stage_blocks)
            ch = out_ch

        self.out_norm = nn.GroupNorm(8, ch) if ch >= 8 else nn.Identity()
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: Optional[List[str]] = None,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Time Embedding
        t_emb = sinusoidal_embedding(timesteps, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        
        # Text Embedding
        # Compute text_emb on the conditioning only if they haven't
        # been computed before. Else use the ones that are provided
        # as input. Helps reduce computation while sampling
        if text_emb is None and conditioning is not None and self.text_model is not None:
            text_emb = self.text_model(conditioning)

        hs = []
        h = self.in_conv(x)
        
        # Encode Stage
        for stage_blocks, down in zip(self.encs, self.downs):
            for block in stage_blocks:
                h = block(h, t_emb, text_emb)
            hs.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid1(h, t_emb, text_emb)
        h = self.mid2(h, t_emb, text_emb)

        # Decode Stage
        for up, stage_blocks, skip in zip(self.ups, self.decs, reversed(hs)):
            h = up(h)
            
            # Align shapes if input was not a power of 2
            if h.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.shape[-2] - h.shape[-2]
                diff_x = skip.shape[-1] - h.shape[-1]
                h = F.pad(h, (0, diff_x, 0, diff_y))
            
            h = torch.cat([h, skip], dim=1)
            for block in stage_blocks:
                h = block(h, t_emb, text_emb)

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h), text_emb