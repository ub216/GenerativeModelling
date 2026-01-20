import math
from typing import List, Optional, Tuple

import torch
from loguru import logger

import helpers.custom_types as custom_types
from helpers.utils import drop_condition
from models.backbone.simple_unet import SimpleUNet
from models.base_model import BaseModel


# -----------------------------
# Flow matching Model
# -----------------------------
class FlowModel(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = (1, 2, 4),
        num_blocks: int | List[int] = [1, 2, 2],
        time_emb_dim: int = 128,
        timesteps: int = 1000,
        device: custom_types.DeviceType = "cuda",
        test_timesteps: Optional[int] = None,
        text_emb_dim: Optional[int] = None,
        drop_condition_ratio: float = 0.25,
        sample_condition_weight: int = 10,
        renormalize: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.unet = SimpleUNet(
            in_channels,
            base_channels,
            channel_mults,
            num_blocks=num_blocks,
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
        self.renormalize = renormalize
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
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.valid_input_combination(conditioning):
            raise ValueError("Invalid input combination")

        if self.renormalize:
            x0 = x0 * 2.0 - 1.0  # to [-1, 1]

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
        pred_flow, _ = self.unet(intermidiate, time_steps, conditioning=conditioning)
        return pred_flow, flow

    def sample(
        self,
        num_samples: int,
        device: custom_types.DeviceType,
        image_size: int | Tuple[int, int],
        batch_size: int = 16,
        conditioning: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        assert conditioning is None or len(conditioning) == num_samples
        # Ensure image_size is a tuple
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        # Ensure conditioning is set up
        if conditioning is None and self.has_conditional_generation:
            conditioning = [""] * num_samples
        return self.sample_flow(
            num_samples, device, image_size, batch_size, conditioning=conditioning
        )

    @torch.no_grad()
    def sample_flow(
        self,
        num_samples: int,
        device: custom_types.DeviceType,
        image_size: Tuple[int, int],
        batch_size: int = 16,
        channels: int = 1,
        conditioning: Optional[List[str]] = None,
    ):
        """
        Generate samples by iteratively predicting the flow.
        num_samples: int
        device: torch device
        image_size: int (assumes square images)
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
            x_t = torch.randn(
                cur_bs, channels, image_size[0], image_size[1], device=device
            )

            cond_batch = (
                conditioning[idx * batch_size : idx * batch_size + cur_bs]
                if conditioning
                else None
            )
            uncond_batch = [""] * cur_bs if self.has_conditional_generation else None

            for t in reversed(range(self.test_timesteps)):
                vec_t = torch.full((cur_bs,), t * self.test_delta, device=device)

                if self.has_conditional_generation:
                    # We double the batch size to do cond and uncond in ONE pass
                    batched_x = torch.cat([x_t, x_t], dim=0)
                    batched_t = torch.cat([vec_t, vec_t], dim=0)
                    batched_cond = cond_batch + uncond_batch

                    # Single forward pass
                    batched_flow, _ = self.unet(
                        batched_x, batched_t, conditioning=batched_cond
                    )

                    # Split the results back
                    flow_cond, flow_uncond = batched_flow.chunk(2)

                    # Apply Guidance
                    flow = flow_uncond + self.sample_condition_weight * (
                        flow_cond - flow_uncond
                    )
                else:
                    flow, _ = self.unet(x_t, vec_t)

                # dynamic thresholding to avoid exploding values
                current_t = t * self.test_delta
                pred_x0 = x_t - current_t * flow
                pred_x0_thresholded = self._dynamic_threshold(pred_x0)
                if current_t > 0:
                    flow = (x_t - pred_x0_thresholded) / current_t

                # Euler step
                x_t -= flow * self.test_delta

            samples.append(x_t.cpu())
        samples = torch.cat(samples, dim=0)[:num_samples]
        self.unet.train()
        if self.renormalize:
            return (samples + 1.0) / 2.0  # to [0, 1]
        return samples.clamp(0.0, 1.0)  # adjust if dataset is [-1,1]

    def _dynamic_threshold(
        self, x0: torch.Tensor, p: float = 0.995, c: float = 1.0
    ) -> torch.Tensor:
        """
        x0: (B, C, H, W) - The predicted clean image
        p: Percentile (usually 0.995)
        c: Target threshold (usually 1.0)
        """
        batch_size, channels, height, width = x0.shape
        # Flatten to (batch, pixels) to find quantile per image
        x_flat = x0.reshape(batch_size, -1)

        # Calculate the s-th percentile of the absolute values
        s = torch.quantile(torch.abs(x_flat), p, dim=1)

        # Only scale if the s-th percentile is greater than the target threshold 'c'
        s = torch.clamp(s, min=c).view(batch_size, 1, 1, 1)

        # Clamp to [-s, s] and then divide by s to bring back to [-1, 1]
        return torch.clamp(x0, -s, s) / s

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
