import math

import torch
from loguru import logger

from models.base_model import BaseModel
from models.simple_unet import SimpleUNet
from utils import drop_condition


# -----------------------------
# Flow matching Model
# -----------------------------
class FlowModel(BaseModel):
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        timesteps=1000,
        device="cuda",
        test_timesteps=None,
        text_emb_dim=None,
        drop_condition_ratio=0.25,
        sample_condition_weight=10,
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

    def forward(self, x0, time_steps=None, x1=None, conditioning=None, *args, **kwargs):
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
        num_samples,
        device,
        img_size,
        batch_size=16,
        conditioning=None,
        *args,
        **kwargs
    ):
        assert conditioning is None or len(conditioning) == num_samples
        if conditioning is None and self.has_conditional_generation:
            conditioning = [""] * num_samples
        return self.sample_flow(
            num_samples, device, img_size, batch_size, conditioning=conditioning
        )

    @torch.no_grad()
    def sample_flow(
        self,
        num_samples,
        device,
        img_size,
        batch_size=16,
        channels=1,
        conditioning=None,
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

    def q_sample(self, x0, t, x1):
        """
        x0: (B,C,H,W)
        t: (B,) float in [0,1]
        x1: same shape as x0
        returns: interpolated sample x_t
        """
        return (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1

    def valid_input_combination(self, conditioning):
        """
        Check if conditioning exits iff model performs conditional generation
        """
        return (self.has_conditional_generation and conditioning is not None) or (
            conditioning is None and not self.has_conditional_generation
        )
