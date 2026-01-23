import copy

import torch

from models.base_model import BaseModel


class EMAModel(BaseModel):
    def __init__(self, model_ema, decay=0.999):
        super().__init__()
        self.model = model_ema
        self.model.eval()
        self.decay = decay
        self.device = next(model_ema.parameters()).device
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, online_model):
        """Update the shadow weights: shadow = shadow * decay + online * (1 - decay)"""
        online_params = dict(online_model.named_parameters())
        shadow_params = dict(self.model.named_parameters())

        for name, param in online_params.items():
            # Update the shadow parameter in-place
            shadow_params[name].sub_((1.0 - self.decay) * (shadow_params[name] - param))

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        """Use the shadow model for generating images"""
        return self.model.sample(*args, **kwargs)
