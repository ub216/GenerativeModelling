import math
from typing import List, Optional, Tuple

import torch
from loguru import logger

import helpers.custom_types as custom_types
from helpers.utils import drop_condition
from models.base_model import BaseModel
from models.simple_unet import SimpleUNet


# -----------------------------
# Flow matching Model
# -----------------------------
class FlowModel(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = (1, 2, 4),
        time_emb_dim: int = 128,
        timesteps: int = 1000,
        device: custom_types.DeviceType = "cuda",
        test_timesteps: Optional[int] = None,
        text_emb_dim: Optional[int] = None,
        drop_condition_ratio: float = 0.25,
        sample_condition_weight: int = 10,
    ):
        super().__init__()
        self.unet = SimpleUNet(
            in_channels,
            base_channels,
            channel_mults,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            device=device,
        )
        self.device = device
        self.timesteps = timesteps
        self.test_timesteps = (
            test_timesteps if test_timesteps is not None else timesteps
        )
        self.test_delta = 1 / self.test_timesteps
        self.text_emb_dim = text_emb_dim
        self.drop_condition_ratio = drop_condition_ratio
        self.sample_condition_weight = sample_condition_weight
        self.has_conditional_generation = True if text_emb_dim is not None else False
        if self.has_conditional_generation:
            logger.info("Created a conditioned flow matching model")
        else:
            logger.info("Created an unconditioned flow matching model")

    def forward(
        self,
        x0: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None,
        x1: Optional[torch.Tensor] = None,
        conditioning: Optional[List[str]] = None,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.valid_input_combination(conditioning):
            raise ValueError("Invalid input combination")
        if time_steps is None:
            b = x0.shape[0]
            time_steps = torch.rand(b, device=self.device)
        if x1 is None:
            x1 = torch.randn_like(x0)
        if conditioning is not None:
            # Randomly drop condition to train model for unconditioned input
            conditioning = drop_condition(conditioning, self.drop_condition_ratio)

        intermidiate = self.q_sample(x0, time_steps, x1)
        flow = x1 - x0
        pred_flow = self.unet(intermidiate, time_steps, conditioning=conditioning)
        return pred_flow, flow

    def sample(
        self,
        num_samples: int,
        device: custom_types.DeviceType,
        img_size: int,
        batch_size: int = 16,
        conditioning: Optional[List[str]] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        assert conditioning is None or len(conditioning) == num_samples
        if conditioning is None and self.has_conditional_generation:
            conditioning = [""] * num_samples
        return self.sample_flow(
            num_samples, device, img_size, batch_size, conditioning=conditioning
        )

    @torch.no_grad()
    def sample_flow(
        self,
        num_samples: int,
        device: custom_types.DeviceType,
        img_size: int,
        batch_size: int = 16,
        channels: int = 1,
        conditioning: Optional[List[str]] = None,
    ):
        """
        Generate samples by iteratively predicting the flow.
        num_samples: int
        device: torch device
        img_size: int (assumes square images)
        batch_size: int
        channels: output channels (C)
        conditioning: input conditioning
        returns: (num_samples, C, H, W) tensor of generated images
        """
        if not self.valid_input_combination(conditioning):
            raise ValueError("Invalid input combination")

        self.unet.eval()
        samples = []
        for idx in range(math.ceil(num_samples / batch_size)):
            cur_bs = min(batch_size, num_samples - len(samples))
            x_t = torch.randn(cur_bs, channels, img_size, img_size, device=device)
            conditioning_batch = (
                conditioning[idx * batch_size : idx * batch_size + cur_bs]
                if conditioning is not None
                else None
            )
            for t in reversed(range(self.test_timesteps)):
                timestep_batch = torch.full(
                    (cur_bs,), t * self.test_delta, device=device, dtype=torch.float
                )
                # predict conditional and unconditional and combine (if conditioning)
                if self.has_conditional_generation:
                    flow_conditioning = self.unet(
                        x_t, timestep_batch, conditioning=conditioning_batch
                    )
                    unconditioning_batch = [""] * cur_bs
                    flow_unconditioning = self.unet(
                        x_t, timestep_batch, conditioning=unconditioning_batch
                    )
                    # guided flow: flow_uncond + scale * (flow_cond - flow_uncond)
                    flow = flow_unconditioning + self.sample_condition_weight * (
                        flow_conditioning - flow_unconditioning
                    )
                else:
                    flow = self.unet(x_t, timestep_batch)

                x_t -= flow * self.test_delta
            samples.append(x_t.cpu())
        samples = torch.cat(samples, dim=0)[:num_samples]
        self.unet.train()
        return samples.clamp(0.0, 1.0)  # adjust if dataset is [-1,1]

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        """
        x0: (B,C,H,W)
        t: (B,) float in [0,1]
        x1: same shape as x0
        returns: interpolated sample x_t
        """
        return (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1

    def valid_input_combination(self, conditioning: Optional[List[str]]) -> bool:
        """
        Check if conditioning exits iff model performs conditional generation
        """
        return (self.has_conditional_generation and conditioning is not None) or (
            conditioning is None and not self.has_conditional_generation
        )
