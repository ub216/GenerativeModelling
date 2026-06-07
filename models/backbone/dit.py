from typing import List, Optional

import torch
import torch.nn as nn

import helpers.custom_types as custom_types
from models.backbone.dit_block_ca import DiTBlockCrossAttn
from models.backbone.image_tokeniser import ImageTokeniser
from models.backbone.positional_embeddings import get_2d_sincos_pos_embed_and_freqs
from models.text_model import T5TextModel
from models.utils import sinusoidal_embedding


# -------------------------
# DiT
# -------------------------
class DiT(nn.Module):
    """
    DiT architecture for image generation with text conditioning and attention mechanism.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_blocks: int = 10,
        emb_dim: int = 512,
        time_emb_dim: int = 128,
        device: custom_types.DeviceType = "cuda",
        max_image_size: int = 256,
        patch_size: int = 16,
        num_attention_heads: int = 8,
    ):
        super().__init__()

        self.max_image_size = max_image_size
        self.patch_size = patch_size

        # Time embedding MLP to process the sinusoidal time embeddings
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Text model usually T5 or CLIP text encoder; projects text conditioning to text_emb_dim
        tm = T5TextModel(device)
        self.text_model = nn.Sequential(tm, nn.Linear(tm.dim, emb_dim))

        # Image tokenizer to convert input images into patch tokens for the DiT blocks
        self.image_tokenizer = ImageTokeniser(patch_size=patch_size, in_channels=in_channels, embed_dim=emb_dim)

        # Get rope frequencies for the attention blocks; these are fixed and can be precomputed
        max_patches = max_image_size // patch_size
        _, rope_freqs = get_2d_sincos_pos_embed_and_freqs(emb_dim // num_attention_heads, max_patches)
        rope_freqs = rope_freqs.view(max_patches, max_patches, -1)  # (H//P, W//P, D)
        self.register_buffer("rope_freqs", rope_freqs)

        # DiT blocks with cross attention for text conditioning
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                DiTBlockCrossAttn(
                    in_channels=emb_dim,
                    time_emb_dim=time_emb_dim,
                    num_heads=num_attention_heads,
                    intialise_zero=True,
                    bias=False,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,  # (B,)
        conditioning: Optional[List[str]] = None,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Time Embedding
        if timesteps.ndim != 1:
            raise ValueError(f"Expected timesteps shape to be (B,), got {timesteps.shape}")
        # check image size is less than max_image_size
        original_shape = x.shape  # (B, C, H, W)
        if original_shape[2] > self.max_image_size or original_shape[3] > self.max_image_size:
            raise ValueError(
                f"Image {original_shape[2]}x{original_shape[3]} exceeds "
                f"max supported size {self.max_image_size}x{self.max_image_size}"
            )

        # Time embedding: (B, time_emb_dim) after MLP
        t_emb = sinusoidal_embedding(timesteps, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # (B, time_emb_dim)

        # Text Embedding
        # Use provided text_emb if available (e.g. from a previous stage in a
        # cascaded setup); otherwise compute from conditioning strings
        # If no conditioning is provided, use dummy empty strings to get a
        # zero text embedding
        if text_emb is None:
            if conditioning is not None:
                text_emb = self.text_model(conditioning)
            else:
                text_emb = self.text_model([""] * x.shape[0])  # Dummy conditioning if none provided

        # Image Tokenization
        x = self.image_tokenizer(x)  # (B, N, emb_dim)

        # DiT Blocks
        height_patches = original_shape[2] // self.image_tokenizer.patch_size
        width_patches = original_shape[3] // self.image_tokenizer.patch_size
        rope_freqs = self.rope_freqs[:height_patches, :width_patches].flatten(0, 1)  # (N, D)
        for block in self.blocks:
            x = block(x, t_emb, text_emb, rope_freqs)

        # Unpatchify the output tokens back to image space
        x = self.image_tokenizer.unpatchify(x, image_size=(original_shape[2], original_shape[3]))  # (B, C, H, W)

        return x, text_emb
