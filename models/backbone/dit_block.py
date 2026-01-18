from typing import Optional

import torch
import torch.nn as nn


class DiTBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        text_emb_dim: Optional[int] = None,
        num_heads: int = 8,
        intialise_zero: bool = True,
    ):
        super().__init__()
        # attention block
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=num_heads, batch_first=True
        )
        self.norm = torch.nn.LayerNorm(in_channels, elementwise_affine=False)

        # always define time_mlp
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))

        cond_in_dim = time_emb_dim
        if text_emb_dim is not None:
            self.text_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(text_emb_dim, text_emb_dim)
            )
            cond_in_dim += text_emb_dim

        # use 6 * in_channels to scale every channel individually
        self.condition_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_in_dim, 6 * in_channels)
        )

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, out_channels),
        )
        if intialise_zero:
            self.intialise_zero()

    def intialise_zero(self):
        nn.init.constant_(self.condition_projection[-1].weight, 0)
        nn.init.constant_(self.condition_projection[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Process Conditionings
        t_cond = self.time_mlp(time_emb)
        cond = t_cond
        if text_emb is not None:
            txt_cond = self.text_mlp(text_emb)
            cond = torch.cat([t_cond, txt_cond], dim=-1)

        # project and unsqueeze for broadcasting: (b, 1, 6*c)
        cond_params = self.condition_projection(cond).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = cond_params.chunk(6, dim=2)

        # attention block with adaLN-Zero
        normed_x = self.norm(x)
        # apply scale and shift (gamma and beta)
        normed_x = normed_x * (1 + gamma1) + beta1
        attn_output, _ = self.attention(normed_x, normed_x, normed_x)
        # apply gate (alpha)
        x_attn = x + (alpha1 * attn_output)

        # FFN block with adaLN-Zero
        normed_x = self.norm(x_attn)
        normed_x = normed_x * (1 + gamma2) + beta2
        ffn_output = self.ffn(normed_x)
        # apply gate (alpha)
        x_dit = x_attn + (alpha2 * ffn_output)

        return x_dit
