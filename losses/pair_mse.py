from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import helpers.custom_types as custom_types


class PairMSELoss(nn.Module):
    def __init__(
        self, reduction: custom_types.ReductionType = "mean", adaptive_power: float = 0.5, adaptive_eps: float = 1e-3
    ):
        super().__init__()
        assert reduction in [
            "sum",
            "mean",
            "adaptive",
        ], "Reduction must be 'sum', 'mean' or 'adaptive'"
        self.reduction = reduction
        self.adaptive_power = adaptive_power
        self.adaptive_eps = adaptive_eps

    def forward(
        self, outputs: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], *args, **kwargs
    ) -> torch.Tensor:
        """
        Computes the Pair Mean Squared Error Loss with optional weighting.
        Args:
            outputs (Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]): A tuple containing:
                - predicted (torch.Tensor): The predicted outputs from the model.
                - target (torch.Tensor): The ground truth targets.
                - weights (Optional[torch.Tensor]): Optional weights for each sample in the batch.
        Returns:
            torch.Tensor: The computed loss value.
        """
        assert (
            len(outputs) >= 2 and outputs[0].shape == outputs[1].shape
        ), "Outputs and targets must have the same shape"
        predicted, target = outputs[0], outputs[1]

        # Calculate raw MSE per pixel
        loss = F.mse_loss(predicted, target, reduction="none")

        # Average over all dimensions except Batch: Result is [B]
        # This ensures the weight applies to the "image-level" error
        loss = loss.mean(dim=list(range(1, loss.ndim)))

        if len(outputs) > 2:
            weights = outputs[2]
            # Apply weights: [B] * [B]
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "adaptive":
            # Adaptive weight w = 1/(loss + eps)^p (stop-gradient) — downweights large residuals
            # so high-error samples don't dominate gradients (ECT / MeanFlow Appendix B.2).
            return (loss * (1 / (loss.detach() + self.adaptive_eps) ** self.adaptive_power)).mean()
        else:
            return loss.sum()
