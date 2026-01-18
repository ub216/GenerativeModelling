import math
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

import helpers.custom_types as custom_types
from helpers.utils import drop_condition, log_once
from models.base_model import BaseModel
from models.backbone.unet import UNet


# -----------------------------
# DDPM Model
# -----------------------------
class DiffusionModel(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4],
        num_blocks: int | List[int] = [1, 2, 2],
        time_emb_dim: int = 128,
        timesteps: int = 1000,
        device: custom_types.DeviceType = "cuda",
        test_timesteps: Optional[int] = None,
        text_emb_dim: Optional[int] = None,
        drop_condition_ratio: float = 0.25,
        sample_condition_weight: float = 7.5,
        renormalise: bool = False,
        schedule_type: str = "cosine",  # linear or cosine
        use_attention: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.unet = UNet(
            in_channels,
            base_channels,
            channel_mults,
            num_blocks=num_blocks,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            device=device,
            use_attention=use_attention,
        )
        self.in_channels = in_channels
        self.timesteps = timesteps
        self.test_timesteps = (
            test_timesteps if test_timesteps is not None else timesteps
        )
        self.text_emb_dim = text_emb_dim
        self.drop_condition_ratio = drop_condition_ratio
        self.sample_condition_weight = sample_condition_weight
        self.has_conditional_generation = text_emb_dim is not None
        self.renormalise = renormalise

        # Register Training Schedule as Buffers (Multi-GPU compatibility)
        train_sched = prepare_noise_schedule(
            self.timesteps, schedule_type=schedule_type
        )
        for k, v in train_sched.items():
            self.register_buffer(f"train_{k}", v)

        if self.has_conditional_generation:
            logger.info("Created a conditioned diffusion model")
        else:
            logger.info("Created an unconditioned diffusion model")

    def forward(
        self,
        x0: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        conditioning: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the noise epsilon that was added to x0 to make x_t.
        """
        if not self.valid_input_combination(conditioning):
            raise ValueError("Invalid input combination")

        if self.renormalise:
            x0 = x0 * 2.0 - 1.0  # Scale [0, 1] -> [-1, 1]

        if time_steps is None:
            time_steps = torch.randint(
                0, self.timesteps, (x0.shape[0],), device=x0.device, dtype=torch.long
            )

        if noise is None:
            noise = torch.randn_like(x0)

        if conditioning is not None:
            # Randomly drop condition to train model for unconditioned input
            conditioning = drop_condition(conditioning, self.drop_condition_ratio)

        x_noisy = self.q_sample(x0, time_steps, noise, mode="train")
        predicted_noise, _ = self.unet(x_noisy, time_steps, conditioning=conditioning)
        return predicted_noise, noise

    # -------------------------
    # Sampling (ancestral reverse diffusion)
    # -------------------------
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: custom_types.DeviceType,
        image_size: int | Tuple[int, int],
        batch_size: int = 16,
        conditioning: Optional[List[str]] = None,
        use_ddim: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        High-level sample function.
        If use_ddim=True, it allows any test_timesteps <= self.timesteps.
        """
        self.unet.eval()
        T_test = (
            self.test_timesteps if self.test_timesteps is not None else self.timesteps
        )

        # Ensure conditioning is set up
        if conditioning is None and self.has_conditional_generation:
            conditioning = [""] * num_samples

        # Ensure image_size is a tuple
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        # Choose sampling logic
        if use_ddim:
            samples = self._sample_ddim(
                num_samples, device, image_size, batch_size, conditioning, T_test
            )
        else:
            if T_test != self.timesteps:
                log_once(
                    "DDPM sampling usually requires test_timesteps == train_timesteps. Switching to DDIM."
                )
                samples = self._sample_ddim(
                    num_samples, device, image_size, batch_size, conditioning, T_test
                )
            else:
                samples = self._sample_ddpm(
                    num_samples, device, image_size, batch_size, conditioning
                )

        self.unet.train()
        if self.renormalise:
            samples = (samples + 1.0) / 2.0
        return samples.clamp(0.0, 1.0)

    @torch.no_grad()
    def _sample_ddim(
        self, num_samples, device, image_size, batch_size, conditioning, T_test
    ):
        # Create the sparse indices for the skip-steps
        times = torch.linspace(
            -1, self.timesteps - 1, T_test + 1, dtype=torch.long, device=device
        )
        times = list(reversed(times.tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # e.g., [(999, 949), (949, 899)...]

        samples = []
        for idx in range(math.ceil(num_samples / batch_size)):
            cur_bs = min(batch_size, num_samples - len(samples))
            x_t = torch.randn(
                cur_bs, self.in_channels, image_size[0], image_size[1], device=device
            )

            c_batch = (
                conditioning[idx * batch_size : idx * batch_size + cur_bs]
                if conditioning
                else None
            )
            u_batch = [""] * cur_bs if self.has_conditional_generation else None

            for t_curr, t_next in time_pairs:
                t_batch = torch.full((cur_bs,), t_curr, device=device, dtype=torch.long)

                # Predict epsilon with CFG
                if self.has_conditional_generation:
                    eps_all, _ = self.unet(
                        torch.cat([x_t, x_t]),
                        torch.cat([t_batch, t_batch]),
                        conditioning=c_batch + u_batch,
                    )
                    e_cond, e_uncond = eps_all.chunk(2)
                    eps_theta = e_uncond + self.sample_condition_weight * (
                        e_cond - e_uncond
                    )
                else:
                    eps_theta, _ = self.unet(x_t, t_batch)

                # DDIM Jump Math
                alpha_bar_curr = self.train_alphas_cumprod[t_curr]
                alpha_bar_next = (
                    self.train_alphas_cumprod[t_next]
                    if t_next >= 0
                    else torch.tensor(1.0).to(device)
                )

                # 1. Estimate x0
                pred_x0 = (
                    x_t - torch.sqrt(1 - alpha_bar_curr) * eps_theta
                ) / torch.sqrt(alpha_bar_curr)
                # 2. Compute direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_bar_next) * eps_theta
                x_t = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt

            samples.append(x_t.cpu())
        return torch.cat(samples, dim=0)[:num_samples]

    @torch.no_grad()
    def _sample_ddpm(self, num_samples, device, image_size, batch_size, conditioning):
        # Standard DDPM requires stepping through every single train timestep
        samples = []
        for idx in range(math.ceil(num_samples / batch_size)):
            cur_bs = min(batch_size, num_samples - len(samples))
            x_t = torch.randn(
                cur_bs, self.in_channels, image_size[0], image_size[1], device=device
            )
            c_batch = (
                conditioning[idx * batch_size : idx * batch_size + cur_bs]
                if conditioning
                else None
            )
            u_batch = [""] * cur_bs if self.has_conditional_generation else None

            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((cur_bs,), t, device=device, dtype=torch.long)

                if self.has_conditional_generation:
                    eps_all, _ = self.unet(
                        torch.cat([x_t, x_t]),
                        torch.cat([t_batch, t_batch]),
                        conditioning=c_batch + u_batch,
                    )
                    e_cond, e_uncond = eps_all.chunk(2)
                    eps_theta = e_uncond + self.sample_condition_weight * (
                        e_cond - e_uncond
                    )
                else:
                    eps_theta, _ = self.unet(x_t, t_batch)

                alpha_t = self.train_alphas[t]
                alpha_cum_t = self.train_alphas_cumprod[t]
                beta_t = self.train_betas[t]
                sigma_t = torch.sqrt(self.train_posterior_variance[t])

                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x_t - (beta_t / torch.sqrt(1 - alpha_cum_t)) * eps_theta
                )
                x_t = mean + sigma_t * torch.randn_like(x_t) if t > 0 else mean

            samples.append(x_t.cpu())
        return torch.cat(samples, dim=0)[:num_samples]

    # -------------------------
    # Forward diffusion q(x_t | x_0)
    # -------------------------
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        x0: (B,C,H,W)
        t: (B,) long with values in [0..T-1]
        noise: same shape as x0
        schedule: dict from prepare_noise_schedule
        returns: x_t = sqrt_alphas_cumprod[t] * x0 + sqrt(1 - alphas_cumprod[t]) * noise
        """
        sqrt_a = getattr(self, f"{mode}_sqrt_alphas_cumprod")[t].view(-1, 1, 1, 1)
        sqrt_1_a = getattr(self, f"{mode}_sqrt_one_minus_alphas_cumprod")[t].view(
            -1, 1, 1, 1
        )
        return sqrt_a * x0 + sqrt_1_a * noise

    def valid_input_combination(self, conditioning: Optional[List[str]]) -> bool:
        """
        Check if conditioning exits iff model performs conditional generation
        """
        return self.has_conditional_generation == (conditioning is not None)


# -------------------------
# Noise schedule helpers
# -------------------------
def prepare_noise_schedule(
    num_timesteps: int, schedule_type: str = "cosine"
) -> Dict[str, torch.Tensor]:
    if schedule_type == "linear":
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    elif schedule_type == "cosine":
        # Cosine schedule: alpha_bar(t) = cos^2((t/T + s) / (1 + s) * pi / 2)
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)

        alphas_cumprod = (
            torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize

        # Calculate betas from alphas_cumprod: beta(t) = 1 - alpha_bar(t) / alpha_bar(t-1)
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0, 0.999)  # Prevent singularities

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

    # Posterior Variance (Small Variance)
    # This calculation is the same regardless of the beta schedule type
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1 - alphas_cumprod),
        "posterior_variance": posterior_variance,
    }
