import math
from typing import List, Optional, Tuple

import torch
from loguru import logger

import helpers.custom_types as custom_types
from helpers.diffusion_utils import drop_condition
from models.backbone.unet import UNet
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
        self.test_timesteps = test_timesteps if test_timesteps is not None else timesteps
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
            time_steps = torch.rand(b, device=x0.device)
        if x1 is None:
            x1 = torch.randn_like(x0)
        if conditioning is not None:
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
        dynamic_threshold: bool = True,
        threshold_coeff: float = 1.0,
        clamp_output: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        assert conditioning is None or len(conditioning) == num_samples
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if conditioning is None and self.has_conditional_generation:
            conditioning = [""] * num_samples
        samples = self.sample_flow(
            num_samples,
            device,
            image_size,
            batch_size,
            conditioning=conditioning,
            dynamic_threshold=dynamic_threshold,
            threshold_coeff=threshold_coeff,
        )
        if self.renormalize:
            samples = (samples + 1.0) / 2.0
        if clamp_output:
            samples = samples.clamp(0.0, 1.0)
        return samples

    @torch.no_grad()
    def sample_flow(
        self,
        num_samples: int,
        device: custom_types.DeviceType,
        image_size: Tuple[int, int],
        batch_size: int = 16,
        conditioning: Optional[List[str]] = None,
        dynamic_threshold: bool = True,
        threshold_coeff: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples by iteratively predicting the flow.
        num_samples: int
        device: torch device
        image_size: (H, W) tuple
        batch_size: int
        conditioning: input conditioning
        returns: (num_samples, C, H, W) raw tensor (not clamped or renormalized)
        """
        if not self.valid_input_combination(conditioning):
            raise ValueError("Invalid input combination")

        self.unet.eval()
        samples = []
        for idx in range(math.ceil(num_samples / batch_size)):
            cur_bs = min(batch_size, num_samples - len(samples))
            x_t = torch.randn(cur_bs, self.in_channels, image_size[0], image_size[1], device=device)

            cond_batch = conditioning[idx * batch_size : idx * batch_size + cur_bs] if conditioning else None
            uncond_batch = [""] * cur_bs if self.has_conditional_generation else None

            for t in reversed(range(self.test_timesteps)):
                vec_t = torch.full((cur_bs,), t * self.test_delta, device=device)

                if self.has_conditional_generation:
                    batched_x = torch.cat([x_t, x_t], dim=0)
                    batched_t = torch.cat([vec_t, vec_t], dim=0)
                    batched_cond = cond_batch + uncond_batch

                    batched_flow, _ = self.unet(batched_x, batched_t, conditioning=batched_cond)

                    flow_cond, flow_uncond = batched_flow.chunk(2)
                    flow = flow_uncond + self.sample_condition_weight * (flow_cond - flow_uncond)
                else:
                    flow, _ = self.unet(x_t, vec_t)

                current_t = t * self.test_delta
                pred_x0 = x_t - current_t * flow
                if dynamic_threshold:
                    pred_x0 = self._dynamic_threshold(pred_x0, c=threshold_coeff)
                else:
                    pred_x0 = pred_x0.clamp(-threshold_coeff, threshold_coeff)
                if current_t > 0:
                    flow = (x_t - pred_x0) / current_t

                x_t -= flow * self.test_delta

            samples.append(x_t)

        self.unet.train()
        return torch.cat(samples, dim=0)[:num_samples]

    def _dynamic_threshold(self, x0: torch.Tensor, p: float = 0.995, c: float = 1.0) -> torch.Tensor:
        """
        x0: (B, C, H, W) - The predicted clean image
        p: Percentile (usually 0.995)
        c: Target threshold (usually 1.0)
        """
        if c == float("inf"):
            return x0
        batch_size = x0.shape[0]
        x_flat = x0.reshape(batch_size, -1)
        s = torch.quantile(torch.abs(x_flat), p, dim=1)
        s = torch.clamp(s, min=c).view(batch_size, 1, 1, 1)
        return torch.clamp(x0, -s, s) / s

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
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
