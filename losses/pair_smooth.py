from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import helpers.custom_types as custom_types


class PairSmoothLoss(nn.Module):
    def __init__(
        self, reduction: custom_types.ReductionType = "mean", beta: float = 1.0
    ):
        super().__init__()
        assert reduction in [
            "sum",
            "mean",
        ], "Reduction must be 'sum' or 'mean'"
        self.reduction = reduction
        self.beta = beta

    def forward(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes the Pair Smooth L1 Loss with optional weighting.
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
        ), "Outputs and inputs must have the same shape"
        predicted, target = outputs[0], outputs[1]

        # Calculate raw MSE per pixel
        loss = F.smooth_l1_loss(predicted, target, reduction="none", beta=self.beta)

        # Average over all dimensions except Batch: Result is [B]
        # This ensures the weight applies to the "image-level" error
        loss = loss.mean(dim=list(range(1, loss.ndim)))

        if len(outputs) > 2:
            weights = outputs[2]
            # Apply weights: [B] * [B]
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
