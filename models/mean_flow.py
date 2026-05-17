import math
from typing import List, Optional, Tuple

import torch
from loguru import logger
from torch.func import jvp

import helpers.custom_types as custom_types
from helpers.diffusion_utils import drop_condition
from models.backbone.unet import UNet
from models.flow import FlowModel


# -----------------------------
# MeanFlow model implementation: https://arxiv.org/pdf/2505.13447
# -----------------------------
class MeanFlowModel(FlowModel):
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
        sample_condition_weight: int = 3,
        renormalize: bool = False,
        use_attention: bool = False,
        same_time_ratio: float = 0.25,
        kappa: float = 0.5,
        logit_sigma=1.0,
        logit_mu=-0.4,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_blocks=num_blocks,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            timesteps=timesteps,
            device=device,
            test_timesteps=test_timesteps,
            drop_condition_ratio=drop_condition_ratio,
            sample_condition_weight=sample_condition_weight,
            renormalize=renormalize,
            use_attention=use_attention,
            *args,
            **kwargs,
        )
        # Replace the single-time UNet created by FlowModel with a dual-time one for MeanFlow
        self.unet = UNet(
            in_channels,
            base_channels,
            channel_mults,
            num_blocks=num_blocks,
            time_emb_dim=(time_emb_dim, time_emb_dim),
            text_emb_dim=text_emb_dim,
            device=device,
            use_attention=use_attention,
        )

        self.same_time_ratio = same_time_ratio
        self.kappa = kappa
        self.logit_sigma = logit_sigma
        self.logit_mu = logit_mu
        self.has_conditional_generation = True if text_emb_dim is not None else False
        if self.has_conditional_generation:
            logger.info("Created a conditioned mean flow matching model")
        else:
            logger.info("Created an unconditioned mean flow matching model")

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
            time_steps = torch.sigmoid(torch.randn(b, 2, device=x0.device) * self.logit_sigma + self.logit_mu)
            time_steps = torch.sort(time_steps, dim=1, descending=True).values  # ensure t > r (t0 > t1)
            same_time = torch.rand(b, device=x0.device) < self.same_time_ratio
            time_steps[same_time] = time_steps[same_time][:, 0:1]  # set t=r for some samples

        if x1 is None:
            x1 = torch.randn_like(x0)
        if conditioning is not None:
            conditioning = drop_condition(conditioning, self.drop_condition_ratio)

        intermidiate = self.q_sample(x0, time_steps[:, 0], x1)  # zt = (1-t) * z0 + t * noise
        flow = x1 - x0
        text_emb = (
            self.unet.text_model(conditioning)
            if conditioning is not None and self.unet.text_model is not None
            else None
        )  # vt = noise - z0
        if self.has_conditional_generation:
            with torch.no_grad():
                same_timesteps = torch.stack([time_steps[:, 0], torch.zeros_like(time_steps[:, 0])], dim=1)
                u_cond, _ = self.unet(intermidiate, same_timesteps, text_emb=text_emb)
                u_uncond, _ = self.unet(intermidiate, same_timesteps, text_emb=None)
                flow = (
                    self.sample_condition_weight * flow
                    + self.kappa * u_cond
                    + (1 - self.sample_condition_weight - self.kappa) * u_uncond
                )  # vt^cfg = w*vt + kappa*u_cond + (1-w-kappa)*u_uncond
        time_conditioning = time_steps.clone()
        time_conditioning[:, 1] = (
            time_steps[:, 0] - time_steps[:, 1]
        )  # second time input is the duration (t-r) instead of absolute time r

        # Compute the predicted mean flow and the time derivative using the Jacobian-vector product (JVP)
        def fn(z, t_cond):
            return self.unet(z, t_cond, text_emb=text_emb)[0]

        time_tangent = torch.ones_like(time_conditioning)
        pred_u, dudt = jvp(fn, (intermidiate, time_conditioning), (flow, time_tangent))

        target_u = (
            flow - (time_steps[:, 0] - time_steps[:, 1]).view(-1, 1, 1, 1) * dudt.detach()
        )  # v = z1-z0, du/dt = (u_pred - u) / (t-r) => u_pred = v - (t-r)*du/dt
        return pred_u, target_u

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
            cur_bs = min(batch_size, num_samples - idx * batch_size)
            x_t = torch.randn(cur_bs, self.in_channels, image_size[0], image_size[1], device=device)

            cond_batch = conditioning[idx * batch_size : idx * batch_size + cur_bs] if conditioning else None

            for t in range(self.test_timesteps, 0, -1):  # t is T, T-1, ..., 1
                vec_t = torch.stack(
                    [
                        torch.full((cur_bs,), t / self.test_timesteps, device=device),
                        torch.full((cur_bs,), self.test_delta, device=device),
                    ],
                    dim=1,
                )  # (B, 2) with start time step and delta time

                if self.has_conditional_generation:

                    u_pred, _ = self.unet(x_t, vec_t, conditioning=cond_batch)
                else:
                    u_pred, _ = self.unet(x_t, vec_t)

                pred_x0 = x_t - self.test_delta * u_pred
                if dynamic_threshold:
                    pred_x0 = self._dynamic_threshold(pred_x0, c=threshold_coeff)
                else:
                    pred_x0 = pred_x0.clamp(-threshold_coeff, threshold_coeff)

                x_t = pred_x0

            samples.append(x_t)

        self.unet.train()
        return torch.cat(samples, dim=0)[:num_samples]
