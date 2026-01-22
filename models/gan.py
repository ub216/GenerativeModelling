from typing import List, Optional, Tuple

import torch
import torch.nn as nn

import helpers.custom_types as custom_types
from models.backbone.residual_conv import ResidualConv, ResidualDeconv
from models.base_model import BaseModel


# -----------------------------
# GAN Model
# -----------------------------
class GAN(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 28,
        generator_feature_dims: List[int] = [64, 32],
        discriminator_feature_dims: List[int] = [32, 64],
        latent_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        device: custom_types.DeviceType = "cuda",
        *args,
        **kwargs,
    ):
        """
        GAN can only handle square images for now
        TODO: Relax constraint that images need to be square
        """
        super().__init__()

        self.image_size = image_size
        self.generator_feature_dims = generator_feature_dims
        self.discriminator_feature_dims = discriminator_feature_dims
        self.latent_dim = latent_dim
        self.device = device

        # ---------- Generator ----------
        # # compute flattened size for convs
        generator_reduced_size = image_size // (2 ** len(generator_feature_dims))
        conv_out_dim = (
            generator_feature_dims[0] * generator_reduced_size * generator_reduced_size
        )

        gen_blocks = [
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, conv_out_dim),
            nn.ReLU(),
            nn.Unflatten(
                1,
                (
                    generator_feature_dims[0],
                    generator_reduced_size,
                    generator_reduced_size,
                ),
            ),
        ]

        for i in range(len(generator_feature_dims) - 1):
            gen_blocks.append(
                ResidualDeconv(
                    generator_feature_dims[i],
                    generator_feature_dims[i + 1],
                    stride=2,
                    norm=False,
                    bias=False,
                )
            )
        gen_blocks.append(
            ResidualDeconv(
                generator_feature_dims[-1],
                in_channels,
                stride=2,
                final_layer=True,
                norm=False,
                bias=False,
            )
        )
        self.generator = nn.Sequential(*gen_blocks)

        # ---------- Discriminator ----------
        dis_blocks = []
        prev_ch = in_channels
        for ch in discriminator_feature_dims:
            dis_blocks.append(
                ResidualConv(
                    prev_ch,
                    ch,
                    stride=2,
                    bias=False,
                    spectral_norm=True,
                    dropout=dropout,
                    activation=nn.LeakyReLU(),
                )
            )
            prev_ch = ch
        dis_blocks.append(nn.Flatten())

        discriminator_reduced_size = image_size // (
            2 ** len(discriminator_feature_dims)
        )
        discriminator_reduced_size = (
            discriminator_reduced_size * discriminator_reduced_size * prev_ch
        )

        # Change the last layer based on required GAN type/loss.
        # For hinge loss Linear O/P is recommended
        dis_blocks.append(nn.Linear(discriminator_reduced_size, 1))
        self.discriminator = nn.Sequential(*dis_blocks)

    def forward(
        self, x: torch.Tensor, latent: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent is None:
            b = x.shape[0]
            latent = torch.randn(b, self.latent_dim).to(self.device)
        gen_sample = self.generator(latent)
        gen_score_gen = self.discriminator(gen_sample)
        gen_score_dis = self.discriminator(gen_sample.detach())
        real_score = self.discriminator(x)

        return gen_score_gen, gen_score_dis, real_score

    def sample(
        self, num_samples: int, device: custom_types.DeviceType, *args, **kwargs
    ):
        latent = torch.randn(num_samples, self.latent_dim).to(device)
        gen_sample = self.generator(latent)
        return gen_sample
