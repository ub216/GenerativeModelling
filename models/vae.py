from typing import List

import torch
import torch.nn as nn

import helpers.custom_types as custom_types
from models.base_model import BaseModel
from models.backbone.residual_conv import ResidualConv, ResidualDeconv


# -----------------------------
# VAE Model
# -----------------------------
class VAE(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 28,
        feature_dims: List[int] = [32, 64],
        latent_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        *args,
        **kwargs,
    ):
        """
        VAE can only handle square images for now
        TODO: Relax constraint that images need to be square
        """
        super().__init__()

        self.image_size = image_size
        self.feature_dims = feature_dims
        self.latent_dim = latent_dim

        # ---------- Encoder ----------
        enc_blocks = []
        prev_c = in_channels
        for feat in feature_dims:
            enc_blocks.append(
                ResidualConv(
                    prev_c, feat, stride=2, dropout=dropout, bias=False, norm=False
                )
            )
            prev_c = feat
        self.encoder_conv = nn.Sequential(*enc_blocks)

        # compute flattened size after convs
        reduced_size = image_size // (2 ** len(feature_dims))
        conv_out_dim = feature_dims[-1] * reduced_size * reduced_size

        self.encoder_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(conv_out_dim, hidden_dim), nn.ReLU()
        )

        # latent variables
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)

        # ---------- Decoder ----------
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, conv_out_dim),
            nn.ReLU(),
            nn.Unflatten(1, (feature_dims[-1], reduced_size, reduced_size)),
        )

        dec_blocks = []
        rev_feats = feature_dims[::-1]
        for i in range(len(rev_feats) - 1):
            dec_blocks.append(
                ResidualDeconv(
                    rev_feats[i],
                    rev_feats[i + 1],
                    stride=2,
                    dropout=dropout,
                    norm=False,
                    bias=False,
                )
            )
        dec_blocks.append(
            ResidualDeconv(
                rev_feats[-1],
                in_channels,
                stride=2,
                final_layer=True,
                norm=False,
                bias=False,
            )
        )

        self.decoder_conv = nn.Sequential(*dec_blocks)

    # ---------- Latent sampling ----------
    def random_sample(
        self, z_mean: torch.Tensor, z_logvar: torch.Tensor
    ) -> torch.Tensor:
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_logvar) * eps

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor, *args, **kwargs):
        h = self.encoder_conv(x)
        h = self.encoder_fc(h)

        z_mean = self.z_mean(h)
        z_logvar = self.z_logvar(h)

        z = self.random_sample(z_mean, z_logvar)

        h_dec = self.decoder_fc(z)
        out = self.decoder_conv(h_dec)

        return out, z_mean, z_logvar

    def sample(
        self, num_samples: int, device: custom_types.DeviceType, *args, **kwargs
    ):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        h_dec = self.decoder_fc(z)
        out = self.decoder_conv(h_dec)
        return out
