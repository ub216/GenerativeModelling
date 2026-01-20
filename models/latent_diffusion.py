import math
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import AutoencoderKL
from loguru import logger

from models.base_model import BaseModel
from models.diffusion import DiffusionModel


class LatentDiffusionModel(BaseModel):
    def __init__(
        self,
        # Diffusion specific params
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4, 8],
        num_blocks: List[int] = [1, 2, 2, 2],
        time_emb_dim: int = 128,
        text_emb_dim: Optional[int] = 128,
        timesteps: int = 1000,
        schedule_type: str = "cosine",
        # VAE params
        renormalise: bool = True,
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        device: str = "cuda",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.device = device

        # 1. Load the Pre-trained VAE
        # This VAE expects 256x256 and produces 32x32x4 latents
        self.vae = AutoencoderKL.from_pretrained(vae_model_name).to(device)
        self.renormalise = renormalise

        # Freeze VAE - we only train the Diffusion backbone
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # The VAE scaling factor is crucial for training stability
        self.scaling_factor = 0.18215

        # initialize diffusionModel backbone
        # Note: in_channels is ALWAYS 4 for this VAE (latent channels)
        del kwargs["in_channels"]  # Remove if passed in kwargs
        self.model = DiffusionModel(
            in_channels=4,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_blocks=num_blocks,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            timesteps=timesteps,
            schedule_type=schedule_type,
            device=device,
            renormalise=False,  # We handle scaling via the VAE
            *args,
            **kwargs,
        )

        self.has_conditional_generation = self.model.has_conditional_generation

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pixels (B, 3, H, W) -> Latents (B, 4, H/8, W/8)"""
        # x should be in range [-1, 1]
        posterior = self.vae.encode(x).latent_dist
        latents = posterior.sample() * self.scaling_factor
        return latents

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latents (B, 4, h, w) -> Pixels (B, 3, H, W)"""
        z = z / self.scaling_factor
        images = self.vae.decode(z).sample
        return images

    def forward(self, x: torch.Tensor, conditioning: Optional[List[str]] = None):
        """
        Training forward pass:
        1. Encode image to latents
        2. Add noise to latents
        3. Predict noise using the internal DiffusionModel
        """
        x0 = x.to(self.device)
        if self.renormalise:
            x0 = x0 * 2.0 - 1.0  # Map [0, 1] to [-1, 1]

        # Encode real images to latent space
        with torch.no_grad():
            latents = self.encode(x0)

        # Run the standard diffusion training logic on the latents
        return self.model(latents, conditioning=conditioning)

    @property
    def train_alphas_cumprod(self):
        return self.model.train_alphas_cumprod

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
        device: str,
        image_size: int | Tuple[int, int],  # The HIGH-RES size (e.g., 256)
        batch_size: int = 16,
        conditioning: Optional[List[str]] = None,
        use_ddim: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        1. Figure out latent dimensions (image_size // 8).
        2. Generate clean latents using the backbone DiffusionModel.
        3. Decode latents back to pixels via VAE.
        """
        self.model.eval()

        # determine Latent Dimensions
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        latent_size = (image_size[0] // 8, image_size[1] // 8)

        # This returns latents in the range the model was trained on (roughly -1 to 1)
        latents = self.model.sample(
            num_samples=num_samples,
            device=device,
            image_size=latent_size,
            batch_size=batch_size,
            conditioning=conditioning,
            use_ddim=use_ddim,
            c=float("inf"),  # Guidance scale for sampling
            **kwargs,
        )

        # decode Latents to Pixels
        # We must process this in batches to avoid OOM on the VAE decoder
        all_images = []
        for i in range(0, latents.shape[0], batch_size):
            batch_latents = latents[i : i + batch_size].to(device)

            # Un-scale the latents before decoding (VAE Scaling Factor)
            batch_latents = batch_latents / self.scaling_factor

            # Decode to pixels
            # decoded.sample is in range [-1, 1]
            decoded = self.decode(batch_latents)
            all_images.append(decoded.cpu())

        samples = torch.cat(all_images, dim=0)

        # final Post-processing
        # Map from [-1, 1] (VAE output) to [0, 1] for visualization
        if self.renormalise:
            samples = (samples + 1.0) / 2.0
        return samples.clamp(0.0, 1.0)
