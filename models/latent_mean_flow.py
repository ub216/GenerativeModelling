from typing import List, Optional, Tuple

import torch
from diffusers import AutoencoderKL
from loguru import logger

import helpers.custom_types as custom_types
from models.base_model import BaseModel
from models.mean_flow import MeanFlowModel


class LatentMeanFlowModel(BaseModel):
    def __init__(
        self,
        # UNet / flow params
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4],
        num_blocks: List[int] = [1, 2, 2],
        time_emb_dim: int = 128,
        text_emb_dim: Optional[int] = None,
        timesteps: int = 1000,
        test_timesteps: int = 1,  # one-step generation (arXiv:2505.13447)
        drop_condition_ratio: float = 0.1,  # paper Sec 4.2
        sample_condition_weight: float = 1.0,  # ω=1.0 (ImageNet B/2 Table 4)
        use_attention: bool = False,
        # MeanFlow-specific params (ImageNet B/2 Table 4 defaults)
        same_time_ratio: float = 0.75,
        kappa: float = 0.5,
        logit_sigma: float = 1.0,
        logit_mu: float = -0.4,
        # VAE params
        renormalise: bool = True,
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        device: custom_types.DeviceType = "cuda",
        compile_vae: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.device = device

        # Load pre-trained VAE (frozen); in_channels is always 4 for this VAE
        self.vae = AutoencoderKL.from_pretrained(vae_model_name, local_files_only=True).to(device)
        self.renormalise = renormalise

        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        if compile_vae:
            try:
                self.vae.encoder = torch.compile(self.vae.encoder, mode="reduce-overhead")
                logger.info("VAE Encoder compiled successfully using torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile VAE: {e}. Falling back to eager mode.")

        # Scaling factor maps latents to unit-ish variance so N(0,I) is a valid prior
        self.scaling_factor = self.vae.config.scaling_factor

        # Drop dataloader-injected fields we hardcode below
        kwargs.pop("in_channels", None)
        kwargs.pop("image_size", None)

        # renormalize=False: VAE scaling factor already handles the range; the MeanFlow
        # model must not re-normalize latents a second time
        self.model = MeanFlowModel(
            in_channels=4,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_blocks=num_blocks,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            timesteps=timesteps,
            test_timesteps=test_timesteps,
            drop_condition_ratio=drop_condition_ratio,
            sample_condition_weight=sample_condition_weight,
            use_attention=use_attention,
            same_time_ratio=same_time_ratio,
            kappa=kappa,
            logit_sigma=logit_sigma,
            logit_mu=logit_mu,
            device=device,
            renormalize=False,
            *args,
            **kwargs,
        )
        self.sample_condition_weight = self.model.sample_condition_weight
        self.has_conditional_generation = self.model.has_conditional_generation

    def encode(self, x: torch.Tensor, use_sample: bool = True) -> torch.Tensor:
        """Pixels (B, 3, H, W) in [-1, 1] -> scaled latents (B, 4, H/8, W/8)"""
        with torch.no_grad():
            posterior = self.vae.encode(x).latent_dist
            if use_sample:
                latents = posterior.sample() * self.scaling_factor
            else:
                latents = posterior.mode() * self.scaling_factor
        return latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latents (B, 4, h, w) -> pixels (B, 3, H, W) in [-1, 1]"""
        z = z / self.scaling_factor
        images = self.vae.decode(z).sample
        return images

    def forward(
        self,
        x: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None,
        x1: Optional[torch.Tensor] = None,
        conditioning: Optional[List[str]] = None,
        use_sample: bool = True,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass:
        1. Encode image to latent space.
        2. Run MeanFlow on the latents (returns (pred_u, target_u, same_time) via JVP, Eq 6).
        Returns (pred_u, target_u) both shaped (B, 4, h, w) and same_time bool mask (B,).
        """
        x0 = x.to(self.device)
        if self.renormalise:
            x0 = x0 * 2.0 - 1.0  # [0, 1] -> [-1, 1] expected by VAE

        with torch.no_grad():
            latents = self.encode(x0, use_sample=use_sample)

        return self.model(latents, time_steps=time_steps, x1=x1, conditioning=conditioning)

    @property
    def unet(self):
        return self.model.unet

    @property
    def timesteps(self):
        return self.model.timesteps

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: custom_types.DeviceType,
        image_size: int | Tuple[int, int],
        batch_size: int = 16,
        conditioning: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        1. Compute latent spatial size (image_size // 8).
        2. Run MeanFlow ODE in latent space to produce clean latents.
        3. Decode latents back to pixels via the VAE.
        Returns tensor of shape (num_samples, 3, H, W) in [0, 1].
        """
        self.model.eval()

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        latent_size = (image_size[0] // 8, image_size[1] // 8)

        latents = self.model.sample(
            num_samples=num_samples,
            device=device,
            image_size=latent_size,
            batch_size=batch_size,
            conditioning=conditioning,
            dynamic_threshold=False,
            threshold_coeff=15.0,
            clamp_output=False,
        )
        if not torch.compiler.is_compiling():
            logger.info(
                f"Generated latents absolute distribution: "
                f"min {latents.abs().min():.4f}, max {latents.abs().max():.4f}"
            )

        all_images = []
        for i in range(0, latents.shape[0], batch_size):
            batch_latents = latents[i : i + batch_size].to(device)
            decoded = self.decode(batch_latents)
            all_images.append(decoded)

        samples = torch.cat(all_images, dim=0)

        # VAE outputs [-1, 1]; map to [0, 1] for visualization / metrics
        if self.renormalise:
            samples = (samples + 1.0) / 2.0

        self.model.train()
        return samples.clamp(0.0, 1.0)
