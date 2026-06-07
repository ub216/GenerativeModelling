from typing import Tuple

import numpy as np
import torch


def get_2d_sincos_pos_embed_and_freqs(
    embed_dim: int, grid_size: int | Tuple[int, int], cls_token: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    h, w = grid_size
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)  # 2, h, w

    grid = grid.reshape([2, 1, h, w])
    pos_embed, freq = get_2d_sincos_pos_embed_and_freqs_from_grid(embed_dim, grid)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        freq = np.concatenate([np.zeros([1, embed_dim]), freq], axis=0)
    return torch.from_numpy(pos_embed).float(), torch.from_numpy(freq).float()


def get_2d_sincos_pos_embed_and_freqs_from_grid(embed_dim: int, grid: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    # use half of dimensions to encode grid_h
    emb_h, freq_h = get_1d_sincos_pos_embed_and_freqs(embed_dim // 2, grid[0].flatten())  # (H*W, D/2), (H*W, D/2)
    emb_w, freq_w = get_1d_sincos_pos_embed_and_freqs(embed_dim // 2, grid[1].flatten())  # (H*W, D/2), (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    # For frequencies, we need to interleave the h and w frequencies to match
    # the way apply_rope expects them (first half of dimensions for h, second half for w)
    freq = np.concatenate([freq_h[:, : embed_dim // 4], freq_w[:, : embed_dim // 4]], axis=1)  # (H*W, D/2)
    freq = np.concatenate([freq, freq], axis=1)  # (M, D)

    return emb, freq


def get_1d_sincos_pos_embed_and_freqs(embed_dim: int, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    embed_dim: output dimension for each position
    grid: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    freq = np.outer(grid, omega)  # (M, D/2), outer product

    emb_sin = np.sin(freq)  # (M, D/2)
    emb_cos = np.cos(freq)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    freq = np.concatenate([freq, freq], axis=1)  # (M, D)
    return emb, freq


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (B, h, L, C)
    # freqs: (L, C)
    if x.ndim != 4 or freqs.shape[0] != x.shape[2] or freqs.shape[1] != x.shape[3]:
        raise ValueError(f"Frequency shape {freqs.shape} is not compatible with input shape {x.shape}")
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, L, C)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, L, C)
    x1, x2 = x.chunk(2, dim=-1)
    x_rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (x_rotated * sin)
