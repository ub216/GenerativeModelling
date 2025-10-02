import math

import torch

from models.base_model import BaseModel
from models.simple_unet import SimpleUNet


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
    ):
        super().__init__()
        self.unet = SimpleUNet(in_channels, base_channels, channel_mults, time_emb_dim)
        self.device = device
        self.timesteps = timesteps
        self.test_timesteps = (
            test_timesteps if test_timesteps is not None else timesteps
        )
        self.test_delta = 1 / self.test_timesteps

    def forward(self, x0, t=None, x1=None, *args, **kwargs):
        if t is None:
            b = x0.shape[0]
            t = torch.rand(b, device=self.device)
        if x1 is None:
            x1 = torch.randn_like(x0)
        intermidiate = self.q_sample(x0, t, x1)
        flow = x1 - x0
        pred_flow = self.unet(intermidiate, t)
        return pred_flow, flow

    @torch.no_grad()
    def sample(
        self, num_samples, device, img_size, batch_size=16, channels=1, *args, **kwargs
    ):
        self.unet.eval()
        samples = []
        for _ in range(math.ceil(num_samples / batch_size)):
            cur_bs = min(batch_size, num_samples - len(samples))
            x_t = torch.randn(cur_bs, channels, img_size, img_size, device=device)
            for t in reversed(range(self.test_timesteps)):
                t_batch = torch.full(
                    (cur_bs,), t * self.test_delta, device=device, dtype=torch.float
                )
                flow = self.unet(x_t, t_batch)
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
