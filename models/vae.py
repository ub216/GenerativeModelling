import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.residual_conv import ResidualConv, ResidualDeconv


# -----------------------------
# VAE Model
# -----------------------------
class VAE(BaseModel):
    def __init__(
        self,
        in_channels=1,
        img_size=28,
        feature_dims=[32, 64],
        latent_dim=32,
        hidden_dim=128,
        dropout=0.5,
    ):
        super().__init__()

        self.img_size = img_size
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
        reduced_size = img_size // (2 ** len(feature_dims))
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
    def random_sample(self, z_mean, z_logvar):
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_logvar) * eps

    # ---------- Forward ----------
    def forward(self, x):
        h = self.encoder_conv(x)
        h = self.encoder_fc(h)

        z_mean = self.z_mean(h)
        z_logvar = self.z_logvar(h)

        z = self.random_sample(z_mean, z_logvar)

        h_dec = self.decoder_fc(z)
        out = self.decoder_conv(h_dec)

        return out, z_mean, z_logvar

    def sample(self, num_samples, device, *args, **kwargs):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        h_dec = self.decoder_fc(z)
        out = self.decoder_conv(h_dec)
        return out
