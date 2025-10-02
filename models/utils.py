import math
import random

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModelWithProjection,
)


# -------------------------
# Utilities: timestep embedding
# -------------------------
def sinusoidal_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings (like Transformer / diffusion papers).
    timesteps: (B,) long tensor
    returns: (B, dim) float tensor
    """
    assert len(timesteps.shape) == 1
    device = timesteps.device
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)
