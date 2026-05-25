import torch
from diffusers import AutoencoderKL
from loguru import logger

import helpers.custom_types as custom_types
from models.base_model import BaseModel


class LatentVAEBase(BaseModel):
    """
    Shared base for latent generative models that use a frozen VAE encoder/decoder.
    Handles VAE loading, freezing, optional compilation, encode/decode, and the
    batched latent-to-pixel decode loop used at sampling time.
    Subclasses provide the generative backbone (__init__), forward(), and sample().
    """

    def __init__(
        self,
        renormalise: bool = True,
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        device: custom_types.DeviceType = "cuda",
        compile_vae: bool = False,
    ):
        super().__init__()
        self.device = device

        # Load pre-trained VAE (frozen); latent channels are always 4 for this VAE
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

    def _decode_latents_to_pixels(
        self,
        latents: torch.Tensor,
        batch_size: int,
        device: custom_types.DeviceType,
    ) -> torch.Tensor:
        """
        Batched VAE decode of a full latent tensor.
        Returns pixels in [0, 1] (after renormalise if enabled).
        Processes in chunks of batch_size to avoid OOM on the VAE decoder.
        """
        all_images = []
        for i in range(0, latents.shape[0], batch_size):
            batch_latents = latents[i : i + batch_size].to(device)
            all_images.append(self.decode(batch_latents))

        samples = torch.cat(all_images, dim=0)

        # VAE outputs [-1, 1]; map to [0, 1] for visualization / metrics
        if self.renormalise:
            samples = (samples + 1.0) / 2.0

        return samples.clamp(0.0, 1.0)
