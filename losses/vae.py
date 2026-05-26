from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import helpers.custom_types as custom_types


class VAELoss(nn.Module):
    def __init__(
        self,
        recon_loss_weight: float = 1.0,
        kl_div_weight: float = 1.0,
        reduction: custom_types.ReductionType = "sum",
    ):
        super().__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")
        self.recon_loss_weight = recon_loss_weight
        self.kl_div_weight = kl_div_weight
        self.reduction = reduction

    def forward(
        self, outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], inputs: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Computes the VAE loss = KLdivergence + L2 reconstructing error
        """
        if not isinstance(outputs, tuple) or len(outputs) != 3:
            raise TypeError(
                f"outputs must be a tuple of length 3 (output, z_mean, z_logvar), got {type(outputs).__name__}"
            )

        output, z_mean, z_logvar = outputs

        recon_loss = F.mse_loss(output, inputs, reduction=self.reduction)
        kl_div = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        loss = self.recon_loss_weight * recon_loss + self.kl_div_weight * kl_div

        return loss
