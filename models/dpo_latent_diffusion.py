import copy
from typing import List, Optional

import torch
from loguru import logger
from peft import LoraConfig, get_peft_model

from models.latent_diffusion import LatentDiffusionModel


class DPOLatentDiffusionModel(LatentDiffusionModel):
    def __init__(
        self,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout=0.05,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if (
            not hasattr(self.model.unet, "use_attention")
            or not self.model.unet.use_attention
        ):
            logger.warning("DPO training works best with attention blocks enabled.")

        # setup Reference Model
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval().requires_grad_(False)

        # inject LoRA into the Policy Model
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["in_proj_weight", "out_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )

        # Wrap the internal DiffusionModel with LoRA
        # This automatically freezes self.model's original parameters
        self.model = get_peft_model(self.model, lora_config)

        # Log trainable parameters to verify LoRA is working
        self.model.print_trainable_parameters()

    def merge_lora_weights(self):
        """
        Permanently merges the LoRA weights into the base model weights.
        Useful for deployment or final evaluation.
        """
        logger.info("Merging LoRA weights into the base UNet...")
        self.model = self.model.merge_and_unload()
        # After merging, self.model becomes a standard DiffusionModel again

    def forward(self, x: torch.Tensor, conditioning: Optional[List[str]] = None):
        batch_size = x.shape[0]
        device = x.device

        # 1encode once to save compute (Using Mode for DPO stability)
        with torch.no_grad():
            x0 = x * 2.0 - 1.0 if self.renormalise else x
            latents = self.encode(x0, use_sample=False)

        # setup Interleaved pairs
        half_bs = batch_size // 2
        t_half = torch.randint(0, self.timesteps, (half_bs,), device=device)
        noise_half = torch.randn_like(latents[:half_bs])

        # Interleave so [W1, L1, W2, L2...]
        t = torch.repeat_interleave(t_half, 2)
        noise = torch.repeat_interleave(noise_half, 2, dim=0)

        # policy Pass (Trainable LoRA weights)
        pol_pred, target_noise = self.model(
            latents, time_steps=t, noise=noise, conditioning=conditioning
        )

        # reference Pass (Original frozen weights)
        with torch.no_grad():
            ref_pred, _ = self.ref_model(
                latents, time_steps=t, noise=noise, conditioning=conditioning
            )

        return (pol_pred, ref_pred, target_noise)
