# train_diffusion.py
import math

import torch

from models.base_model import BaseModel
from models.simple_unet import SimpleUNet


# -----------------------------
# DDIM Model
# -----------------------------
class DiffusionModel(BaseModel):
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        timesteps=1000,
        device="cuda",
        test_timesteps=None,
    ):
        super().__init__()
        self.unet = SimpleUNet(in_channels, base_channels, channel_mults, time_emb_dim)
        self.timesteps = timesteps
        self.device = device
        self.test_timesteps = (
            test_timesteps if test_timesteps is not None else timesteps
        )
        self.train_schedule = prepare_noise_schedule(self.timesteps, device)
        self.test_schedule = prepare_noise_schedule(self.test_timesteps, device)

    def forward(self, x0, t=None, noise=None):
        """
        Predict the noise epsilon that was added to x0 to make x_t.
        """
        if t is None:
            b = x0.shape[0]
            t = torch.randint(
                0, self.timesteps, (b,), device=self.device, dtype=torch.long
            )
        if noise is None:
            noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise, self.train_schedule)
        predicted_noise = self.unet(x_noisy, t)
        return predicted_noise, noise

    def sample(self, num_samples, device, img_size, batch_size=16):
        return self.sample_ddpm(num_samples, device, img_size, batch_size)

    # -------------------------
    # Sampling (ancestral reverse diffusion)
    # -------------------------
    @torch.no_grad()
    def sample_ddpm(self, num_samples, device, img_size, batch_size=16, channels=1):
        """
        Generate samples by iteratively denoising from pure noise.
        num_samples: int
        device: torch device
        img_size: int (assumes square images)
        batch_size: int
        returns: (num_samples, C, H, W) tensor of generated images
        """
        self.unet.eval()
        schedule = self.test_schedule
        T = schedule["betas"].shape[0]
        samples = []
        for _ in range(math.ceil(num_samples / batch_size)):
            cur_bs = min(batch_size, num_samples - len(samples))
            x_t = torch.randn(cur_bs, channels, img_size, img_size, device=device)
            for t in reversed(range(T)):
                t_batch = torch.full((cur_bs,), t, device=device, dtype=torch.long)
                # predict noise
                eps_theta = self.unet(x_t, t_batch)
                beta_t = schedule["betas"][t]
                alpha_t = schedule["alphas"][t]
                alpha_cum_t = schedule["alphas_cumprod"][t]
                sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
                # alternative simpler mean formula used commonly:
                mean = sqrt_recip_alpha * (
                    x_t - (beta_t / torch.sqrt(1 - alpha_cum_t)) * eps_theta
                )
                if t > 0:
                    noise = torch.randn_like(x_t)
                    sigma_t = torch.sqrt(beta_t)
                    x_t = mean + sigma_t * noise
                else:
                    x_t = mean
            samples.append(x_t.cpu())
        samples = torch.cat(samples, dim=0)[:num_samples]
        self.unet.train()
        return samples.clamp(0.0, 1.0)

    # -------------------------
    # Forward diffusion q(x_t | x_0)
    # -------------------------
    def q_sample(self, x0, t, noise, schedule):
        """
        x0: (B,C,H,W)
        t: (B,) long with values in [0..T-1]
        noise: same shape as x0
        schedule: dict from prepare_noise_schedule
        returns: x_t = sqrt_alphas_cumprod[t] * x0 + sqrt(1 - alphas_cumprod[t]) * noise
        """
        # index into precomputed tensors
        sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
        sqrt_1_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_1_a * noise


# -------------------------
# Noise schedule helpers (linear Beta schedule)
# -------------------------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    """
    Linear schedule from beta_start to beta_end
    timesteps: int
    returns: (timesteps,) tensor of betas
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def prepare_noise_schedule(num_timesteps, device):
    """
    Prepare noise schedule tensors for diffusion process.
    num_timesteps: int
    device: torch device
    returns: dict of tensors (betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    """
    betas = linear_beta_schedule(num_timesteps).to(device)  # (T,)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }
