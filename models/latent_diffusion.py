import math
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL
from loguru import logger
from matplotlib import pyplot as plt
from PIL import Image

from models.base_model import BaseModel
from models.diffusion import DiffusionModel


class LatentDiffusionModel(BaseModel):
    def __init__(
        self,
        # Diffusion specific params
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4],
        num_blocks: List[int] = [1, 2, 2],
        time_emb_dim: int = 128,
        text_emb_dim: Optional[int] = None,
        timesteps: int = 1000,
        schedule_type: str = "cosine",
        # VAE params
        renormalise: bool = True,
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        device: str = "cuda",
        compile_vae: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.device = device

        # load the Pre-trained VAE
        self.vae = AutoencoderKL.from_pretrained(
            vae_model_name, local_files_only=True
        ).to(device)
        self.renormalise = renormalise

        # Freeze VAE - we only train the Diffusion backbone
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        if compile_vae:
            try:
                self.vae.encoder = torch.compile(
                    self.vae.encoder, mode="reduce-overhead"
                )
                logger.info("VAE Encoder compiled successfully using torch.compile")
            except Exception as e:
                logger.warning(
                    f"Failed to compile VAE: {e}. Falling back to eager mode."
                )

        # The VAE scaling factor is crucial for training stability
        self.scaling_factor = self.vae.config.scaling_factor

        # initialize diffusionModel backbone
        # Note: in_channels is ALWAYS 4 for this VAE (latent channels)
        kwargs.pop("in_channels", None)  # Remove if passed in kwargs
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
        self.sample_condition_weight = self.model.sample_condition_weight
        self.has_conditional_generation = self.model.has_conditional_generation

    def encode(self, x: torch.Tensor, use_sample=True) -> torch.Tensor:
        """Pixels (B, 3, H, W) -> Latents (B, 4, H/8, W/8)"""
        # x should be in range [-1, 1]
        with torch.no_grad():
            posterior = self.vae.encode(x).latent_dist
            if use_sample:
                latents = posterior.sample() * self.scaling_factor
            else:
                latents = posterior.mode() * self.scaling_factor
        return latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latents (B, 4, h, w) -> Pixels (B, 3, H, W)"""
        with torch.no_grad():
            z = z / self.scaling_factor
            images = self.vae.decode(z).sample
        return images

    def forward(
        self,
        x: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        conditioning: Optional[List[str]] = None,
        use_sample: bool = True,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            latents = self.encode(x0, use_sample=use_sample)

        # Run the standard diffusion training logic on the latents
        return self.model(
            latents, time_steps=time_steps, noise=noise, conditioning=conditioning
        )

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
            dynamic_threshold=False,  # No dynamic thresholding during sampling
            threshold_coeff=15.0,  # Clamping value
            clamp_output=False,
            **kwargs,
        )
        logger.info(
            f"Generated latents absolute distribution: min {latents.abs().min()}, max {latents.abs().max()}"
        )

        # decode Latents to Pixels
        # We must process this in batches to avoid OOM on the VAE decoder
        all_images = []
        for i in range(0, latents.shape[0], batch_size):
            batch_latents = latents[i : i + batch_size].to(device)

            # Decode to pixels
            # decoded.sample is in range [-1, 1]
            decoded = self.decode(batch_latents)
            all_images.append(decoded)

        samples = torch.cat(all_images, dim=0)

        # final Post-processing
        # Map from [-1, 1] (VAE output) to [0, 1] for visualization
        if self.renormalise:
            samples = (samples + 1.0) / 2.0

        self.model.train()
        return samples.clamp(0.0, 1.0)


def test_vae_reconstruction(
    model_path: str = "stabilityai/sd-vae-ft-mse",
    img_path: str = "assets/test_image.jpg",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 512

    # Initialize the Model
    print("Initializing LatentDiffusionModel...")
    model = LatentDiffusionModel()
    model.eval()

    # Preprocess Test Image
    transform = T.Compose(
        [
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),  # Maps to [0, 1]
        ]
    )

    # Load a real image or create a dummy one if not provided
    try:
        raw_img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"Image {img_path} not found, using random noise for test.")
        raw_img = Image.fromarray(
            (torch.rand(img_size, img_size, 3).numpy() * 255).astype("uint8")
        )

    img_tensor = transform(raw_img).unsqueeze(0).to(device)
    img_tensor = img_tensor * 2.0 - 1.0  # Map to [-1, 1] as expected by VAE

    # Generate Latents via Encoder
    print("Encoding...")
    with torch.no_grad():
        # use_sample=False for deterministic mode comparison
        latents = model.encode(img_tensor, use_sample=False)
    print(f"latents distribution: min {latents.abs().min()}, max {latents.abs().max()}")

    # Decode Latents via PyTorch Decoder
    print("Decoding...")
    with torch.no_grad():
        reconstructed_tensor = model.decode(latents)

    # Post-process and Plot
    def to_numpy(t):
        t = (t.clamp(-1, 1) + 1) / 2  # Denormalize to [0, 1]
        return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    orig_np = to_numpy(img_tensor)
    recon_np = to_numpy(reconstructed_tensor)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(orig_np)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("TRT Encoded -> PT Decoded")
    plt.imshow(recon_np)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    print("Test complete.")


if __name__ == "__main__":
    test_vae_reconstruction()
