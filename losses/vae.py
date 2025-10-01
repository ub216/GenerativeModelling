import torch
import torch.nn.functional as F


class VAELoss:
    def __init__(self, recon_loss_weight=1.0, kl_div_weight=1.0, reduction="sum"):
        assert reduction in [
            "sum",
            "mean",
            "none",
        ], "Reduction must be 'sum', 'mean', or 'none'"
        self.recon_loss_weight = recon_loss_weight
        self.kl_div_weight = kl_div_weight
        self.reduction = reduction

    def __call__(self, outputs, inputs):
        assert (
            isinstance(outputs, tuple) and len(outputs) == 3
        ), "Outputs must be a tuple of (output, z_logvar, z_mean)"

        output, z_mean, z_logvar = outputs

        recon_loss = F.mse_loss(output, inputs, reduction=self.reduction)
        kl_div = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        loss = self.recon_loss_weight * recon_loss + self.kl_div_weight * kl_div

        return loss
