import torch
import torch.nn as nn

from models.backbone.positional_embeddings import apply_rope


class DiTBlockCrossAttn(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_emb_dim: int,
        num_heads: int = 8,
        intialise_zero: bool = True,
        bias: bool = False,
        rope_freqs: torch.Tensor | None = None,
    ):
        super().__init__()

        # define time_mlp
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim, bias=bias))

        # use 9 * in_channels to scale every channel individually
        # TODO: Naive AdaLN-Zero style conditioning adds a lot of parametersand compute.
        # Future work: implement AdaLN-LORA and compare.
        self.condition_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 9 * in_channels, bias=bias))

        # attention block
        self.s_norm = torch.nn.LayerNorm(in_channels, elementwise_affine=False)
        self.s_attention = RMSNormAttention(
            embed_dim=in_channels, num_heads=num_heads, bias=bias, rope_freqs=rope_freqs
        )

        # cross attention block does not use RoPE, as text embeddings are not spatial
        # and thus do not benefit from relative positional encoding
        self.c_norm = torch.nn.LayerNorm(in_channels, elementwise_affine=False)
        self.c_attention = RMSNormAttention(embed_dim=in_channels, num_heads=num_heads, bias=bias)

        # ffn block
        self.ffn_norm = torch.nn.LayerNorm(in_channels, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4, bias=bias),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels),
        )
        if intialise_zero:
            self.intialise_zero()

    def intialise_zero(self):
        nn.init.constant_(self.condition_projection[-1].weight, 0)
        if self.condition_projection[-1].bias is not None:
            nn.init.constant_(self.condition_projection[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:

        # Process Conditionings
        t_cond = self.time_mlp(time_emb)

        # project and unsqueeze for broadcasting: (b, 1, 9*c)
        cond_params = self.condition_projection(t_cond).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2, gamma3, beta3, alpha3 = cond_params.chunk(9, dim=2)

        # Self attention block with adaLN-Zero
        normed_x = self.s_norm(x)
        # apply scale and shift (gamma and beta)
        normed_x = normed_x * (1 + gamma1) + beta1
        attn_output = self.s_attention(normed_x, normed_x)
        # apply gate (alpha)
        x_attn = x + (alpha1 * attn_output)

        # Cross attention block with adaLN-Zero
        normed_x = self.c_norm(x_attn)
        normed_x = normed_x * (1 + gamma2) + beta2
        attn_output = self.c_attention(normed_x, text_emb)
        # apply gate (alpha)
        x_attn = x_attn + (alpha2 * attn_output)

        # FFN block with adaLN-Zero
        normed_x = self.ffn_norm(x_attn)
        normed_x = normed_x * (1 + gamma3) + beta3
        ffn_output = self.ffn(normed_x)
        # apply gate (alpha)
        x_dit = x_attn + (alpha3 * ffn_output)

        return x_dit


class RMSNormAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        rope_freqs: torch.Tensor | None = None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )  # always use bias in output projection to allow for zero initialization of attention output if needed
        self.num_heads = num_heads
        self.query_norm = nn.RMSNorm(embed_dim // num_heads)
        self.key_norm = nn.RMSNorm(embed_dim // num_heads)
        self.dropout = dropout
        if rope_freqs is not None and rope_freqs.shape[1] != embed_dim // num_heads:
            raise ValueError(
                f"RoPE frequencies shape {rope_freqs.shape} incompatible with "
                f"embed_dim {embed_dim} and num_heads {num_heads}"
            )
        self.register_buffer("rope_freqs", rope_freqs)  # (L, Ch)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(x)  # (B, N, C)
        key = self.key_proj(y)  # (B, N, C)
        value = self.value_proj(y)  # (B, N, C)

        query = query.view(query.size(0), query.size(1), self.num_heads, -1).transpose(1, 2)  # (B, h, N, Ch)
        key = key.view(key.size(0), key.size(1), self.num_heads, -1).transpose(1, 2)  # (B, h, N, Ch)
        value = value.view(value.size(0), value.size(1), self.num_heads, -1).transpose(1, 2)  # (B, h, N, Ch)

        query = self.query_norm(query)
        key = self.key_norm(key)

        if self.rope_freqs is not None:
            query = apply_rope(query, self.rope_freqs)
            key = apply_rope(key, self.rope_freqs)

        # Use Flash Attention if available, otherwise fall back to manual scaled dot-product attention
        out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=self.dropout, training=self.training
        )
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)  # (B, N, C)
        out = self.out_proj(out)

        return out
