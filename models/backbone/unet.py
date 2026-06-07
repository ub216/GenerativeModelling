from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

import helpers.custom_types as custom_types
from helpers.utils import log_once_info
from models.backbone.dit_block_simple import DiTBlockSimple
from models.backbone.positional_embeddings import get_2d_sincos_pos_embed_and_freqs
from models.backbone.residual_conv import ResidualConv
from models.text_model import ClipTextModel
from models.utils import sinusoidal_embedding


# -------------------------
# UNet
# -------------------------
class UNet(nn.Module):
    """
    UNet architecture for image generation with optional text conditioning and attention mechanism.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4],
        num_blocks: Union[int, List[int]] = [1, 2, 2],
        time_emb_dim: Union[
            int, Tuple[int, int]
        ] = 128,  # for mean flow, this can be a tuple specifying separate dims for start time and delta time
        text_emb_dim: Optional[int] = None,
        use_attention: bool = False,
        device: custom_types.DeviceType = "cuda",
        max_image_size: int = 256,
    ):
        super().__init__()

        self.use_attention = use_attention
        # Normalize num_blocks to a list matching the number of stages
        if isinstance(num_blocks, int):
            self.num_blocks_list = [num_blocks] * len(channel_mults)
        else:
            if len(num_blocks) != len(channel_mults):
                raise ValueError("num_blocks list must match length of channel_mults")
            self.num_blocks_list = num_blocks

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.time_emb_dim = time_emb_dim
        if isinstance(self.time_emb_dim, int):
            self.time_emb_dim = (self.time_emb_dim,)  # make it a tuple for uniform processing
        t_time_emb_dim = sum(self.time_emb_dim)

        self.time_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(t, t),
                    nn.SiLU(),
                    nn.Linear(t, t),
                )
                for t in self.time_emb_dim
            ]
        )

        if text_emb_dim is not None:
            tm = ClipTextModel(device)
            self.text_model = nn.Sequential(tm, nn.Linear(tm.dim, text_emb_dim))
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
                    ResidualConv(
                        block_in,
                        out_ch,
                        time_emb_dim=t_time_emb_dim,
                        text_emb_dim=text_emb_dim,
                    )
                )

            self.encs.append(stage_blocks)
            self.downs.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            ch = out_ch

        # Bottleneck
        if not use_attention:
            self.mid1 = ResidualConv(ch, ch, time_emb_dim=t_time_emb_dim, text_emb_dim=text_emb_dim)
            self.mid2 = ResidualConv(ch, ch, time_emb_dim=t_time_emb_dim, text_emb_dim=text_emb_dim)
        else:
            self.mid1 = DiTBlockSimple(
                in_channels=ch,
                time_emb_dim=t_time_emb_dim,
                text_emb_dim=text_emb_dim,
            )
            self.mid2 = DiTBlockSimple(
                in_channels=ch,
                time_emb_dim=t_time_emb_dim,
                text_emb_dim=text_emb_dim,
            )
            bottleneck_grid = max_image_size // (2 ** len(channel_mults))
            pos_embed, _ = get_2d_sincos_pos_embed_and_freqs(ch, max_image_size // (2 ** len(channel_mults)))
            pos_embed = pos_embed.view(1, bottleneck_grid, bottleneck_grid, ch)
            log_once_info(f"pos_embed shape: {pos_embed.shape} for bottleneck size {bottleneck_grid}")
            self.register_buffer("pos_embed", pos_embed)  # (1, h, w, c)

        # Decoder
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        rev_mults = list(channel_mults)[::-1]
        rev_blocks = list(self.num_blocks_list)[::-1]

        for i, mult in enumerate(rev_mults):
            out_ch = base_channels * mult
            self.ups.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1))

            stage_blocks = nn.ModuleList()
            for b in range(rev_blocks[i]):
                # First block handles concatenated skip connection (out_ch * 2)
                block_in = out_ch * 2 if b == 0 else out_ch
                stage_blocks.append(
                    ResidualConv(
                        block_in,
                        out_ch,
                        time_emb_dim=t_time_emb_dim,
                        text_emb_dim=text_emb_dim,
                    )
                )
            self.decs.append(stage_blocks)
            ch = out_ch

        self.out_norm = nn.GroupNorm(8, ch) if ch >= 8 else nn.Identity()
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,  # can be (B,) or (B, 2) for mean flow
        conditioning: Optional[List[str]] = None,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Time Embedding
        if timesteps.ndim != len(self.time_emb_dim):
            raise ValueError(
                f"Expected timesteps shape to match time_emb_dim, got {timesteps.shape} and {self.time_emb_dim}"
            )
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)  # (B,) -> (B, 1)
        t_emb = [sinusoidal_embedding(timesteps[:, i], self.time_emb_dim[i]) for i in range(len(self.time_emb_dim))]
        t_emb = torch.cat(
            [self.time_mlp[i](t_emb[i]) for i in range(len(self.time_emb_dim))], dim=1
        )  # Final time embedding

        # Text Embedding
        # Compute text_emb on the conditioning only if they haven't
        # been computed before. Else use the ones that are provided
        # as input. Helps reduce computation while sampling
        if text_emb is None and conditioning is not None and self.text_model is not None:
            text_emb = self.text_model(conditioning)

        ys = []
        y = self.in_conv(x)

        # Encode Stage
        for stage_blocks, down in zip(self.encs, self.downs):
            for block in stage_blocks:
                y = block(y, t_emb, text_emb)
            ys.append(y)
            y = down(y)

        if self.use_attention:
            if not torch.compiler.is_compiling():
                log_once_info(f"Using attention in the bottleneck of UNet with bottleneck shape {y.shape}")
            b, c, h, w = y.shape
            y = y.view(b, c, h * w).permute(0, 2, 1)  # (b, seq_len, c)
            if not torch.compiler.is_compiling():
                log_once_info(
                    f"Reshaped bottleneck to {y.shape} for attention, pos_embed shape: {self.pos_embed.shape}"
                )
            curr_pos = self.pos_embed[:, :h, :w, :].reshape(1, h * w, c)  # (1, seq_len, c)
            y = y + curr_pos

        # Bottleneck
        y = self.mid1(y, t_emb, text_emb)
        y = self.mid2(y, t_emb, text_emb)

        if self.use_attention:
            y = y.permute(0, 2, 1).contiguous().view(b, c, h, w)
        # Decode Stage
        for up, stage_blocks, skip in zip(self.ups, self.decs, reversed(ys)):
            y = up(y)

            # Align shapes if input was not a power of 2
            if y.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.shape[-2] - y.shape[-2]
                diff_x = skip.shape[-1] - y.shape[-1]
                y = F.pad(y, (0, diff_x, 0, diff_y))

            y = torch.cat([y, skip], dim=1)
            for block in stage_blocks:
                y = block(y, t_emb, text_emb)

        y = self.out_norm(y)
        y = self.out_act(y)
        return self.out_conv(y), text_emb
