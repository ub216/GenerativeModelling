import math
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

import helpers.custom_types as custom_types
from helpers.diffusion_utils import drop_condition
from helpers.utils import log_once_warning
from models.backbone.unet import UNet
from models.base_model import BaseModel


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

        # Test schedule (can be different number of timesteps)
        if self.test_timesteps != self.timesteps:
            test_sched = prepare_noise_schedule(
                self.test_timesteps, schedule_type=schedule_type
            )
            for k, v in test_sched.items():
                self.register_buffer(f"test_{k}", v, persistent=False)
        else:
            for k in train_sched.keys():
                self.register_buffer(
                    f"test_{k}", getattr(self, f"train_{k}"), persistent=False
                )

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

        # TODO: SNR-weighted loss add as a config option
        # Compute SNR-weighted MSE loss weights
        if 0:
            alphas_cumprod = self.train_alphas_cumprod[time_steps]
            snr = alphas_cumprod / (1 - alphas_cumprod).clamp(min=1e-7)
            mse_loss_weights = torch.stack(
                [snr, torch.ones_like(snr) * 5.0], dim=1
            ).min(dim=1)[0]
            return predicted_noise, noise, mse_loss_weights
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
        dynamic_threshold: bool = True,
        threshold_coeff: float = 1.0,
        clamp_output=True,
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
                num_samples,
                device,
                image_size,
                batch_size,
                conditioning,
                T_test,
                dynamic_threshold=dynamic_threshold,
                threshold_coeff=threshold_coeff,
            )
        else:
            samples = self._sample_ddpm(
                num_samples,
                device,
                image_size,
                batch_size,
                conditioning,
                dynamic_threshold=dynamic_threshold,
                threshold_coeff=threshold_coeff,
            )

        self.unet.train()
        if self.renormalise:
            samples = (samples + 1.0) / 2.0
        if clamp_output:
            samples = samples.clamp(0.0, 1.0)
        return samples

    @torch.no_grad()
    def _sample_ddim(
        self,
        num_samples: int,
        device: torch.device,
        image_size: Tuple[int, int],
        batch_size: int,
        conditioning: Optional[List[str]],
        T_test: int,
        dynamic_threshold: bool,
        threshold_coeff: float,
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

                # estimate x0
                pred_x0 = (
                    x_t - torch.sqrt(1 - alpha_bar_curr) * eps_theta
                ) / torch.sqrt(alpha_bar_curr)

                # dynamic thresholding to keep pixel values in check
                if dynamic_threshold:
                    pred_x0 = self._dynamic_threshold(pred_x0, c=threshold_coeff)
                else:
                    pred_x0 = pred_x0.clamp(-threshold_coeff, threshold_coeff)

                # compute direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_bar_next) * eps_theta
                x_t = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt

            samples.append(x_t)
        return torch.cat(samples, dim=0)[:num_samples]

    @torch.no_grad()
    def _sample_ddpm(
        self,
        num_samples: int,
        device: torch.device,
        image_size: Tuple[int, int],
        batch_size: int,
        conditioning: Optional[List[str]],
        dynamic_threshold: bool,
        threshold_coeff: float,
    ) -> torch.Tensor:
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

            for t in reversed(range(self.test_timesteps)):
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

                alpha_cum_t = self.test_alphas_cumprod[t]

                if alpha_cum_t < 1e-7:
                    # At extreme noise, the model's prediction is unreliable for x0 reconstruction
                    # We can approximate pred_x0 as 0 or a very small value to prevent explosion
                    pred_x0 = torch.zeros_like(x_t)
                    log_once_warning(f"Warning: Extremely low alpha_cum_t at t = {t}")
                else:
                    sqrt_alpha_cum_t = torch.sqrt(alpha_cum_t).clamp(min=1e-12)
                    sqrt_one_minus_alpha_cum_t = torch.sqrt(1 - alpha_cum_t)

                    # estimate x0 from the noise prediction
                    pred_x0 = (
                        x_t - sqrt_one_minus_alpha_cum_t * eps_theta
                    ) / sqrt_alpha_cum_t

                # apply thresholding to keep values in check
                if dynamic_threshold:
                    pred_x0 = self._dynamic_threshold(pred_x0, c=threshold_coeff)
                else:
                    pred_x0 = pred_x0.clamp(-threshold_coeff, threshold_coeff)

                # use the thresholded x0 to find the mean for the next step (x_{t-1})
                alpha_t = self.test_alphas[t]
                beta_t = self.test_betas[t]
                alpha_cum_prev = (
                    self.test_alphas_cumprod[t - 1]
                    if t > 0
                    else torch.tensor(1.0).to(device)
                )
                # At t=0, (1 - alpha_cum_t) becomes very small (~4e-5).
                # We clamp the denominator to prevent the final latent from blowing up.
                denom = (1 - alpha_cum_t).clamp(min=1e-7)

                # posterior mean coefficients
                coeff_x0 = (torch.sqrt(alpha_cum_prev) * beta_t) / denom
                coeff_xt = (torch.sqrt(alpha_t) * (1 - alpha_cum_prev)) / denom
                mean = coeff_x0 * pred_x0 + coeff_xt * x_t

                sigma_t = torch.sqrt(self.test_posterior_variance[t])
                x_t = mean + sigma_t * torch.randn_like(x_t) if t > 0 else mean
            samples.append(x_t)
        return torch.cat(samples, dim=0)[:num_samples]

    def _dynamic_threshold(
        self, x0: torch.Tensor, p: float = 0.995, c: float = 1.0
    ) -> torch.Tensor:
        """
        x0: (B, C, H, W) - The predicted clean image
        p: Percentile (usually 0.995)
        c: Target threshold (usually 1.0)
        """
        if c == float("inf"):
            return x0
        batch_size, channels, height, width = x0.shape
        # Flatten to (batch, pixels) to find quantile per image
        x_flat = x0.reshape(batch_size, -1)

        # Calculate the s-th percentile of the absolute values
        s = torch.quantile(torch.abs(x_flat), p, dim=1)

        # Only scale if the s-th percentile is greater than the target threshold 'c'
        s = torch.clamp(s, min=c).view(batch_size, 1, 1, 1)

        # Clamp to [-s, s] and then divide by s to bring back to [-1, 1]
        return torch.clamp(x0, -s, s) / s

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
    # Use float64 for schedule generation to avoid precision drift
    if schedule_type == "linear":
        betas = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float64)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    elif schedule_type == "cosine":
        s = 0.02
        x = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)
        # Standard Improved DDPM cosine curve
        alphas_cumprod = (
            torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # Calculate betas from the curve
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.999)  # Clip min to avoid no-op steps
        # Use the original curve values for better precision
        alphas_cumprod = alphas_cumprod[1:]
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    alphas = 1.0 - betas
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]]
    )

    # --- STABILITY FIX FOR LATENT DIFFUSION ---
    # The denominator (1 - alphas_cumprod) approaches 0 as t approaches 0 (alphas_cumprod -> 1).
    # We clamp the denominator to prevent the posterior variance from becoming infinite/NaN.
    # We also clamp the numerator to ensure non-negative variance.
    denom = (1.0 - alphas_cumprod).clamp(min=1e-12)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev).clamp(min=0) / denom

    logger.info(
        f"SNR range: {((alphas_cumprod / (1 - alphas_cumprod)).sqrt()).min()} to {((alphas_cumprod / (1 - alphas_cumprod)).sqrt()).max()}"
    )

    # Cast back to float32 only at the end
    return {
        "betas": betas.float(),
        "alphas": alphas.float(),
        "alphas_cumprod": alphas_cumprod.float(),
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod).float(),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1 - alphas_cumprod).float(),
        "posterior_variance": posterior_variance.float(),
    }
