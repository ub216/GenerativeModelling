from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import helpers.custom_types as custom_types


class PairMADLoss(nn.Module):
    def __init__(self, reduction: custom_types.ReductionType = "mean"):
        super().__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")
        self.reduction = reduction

    def forward(
        self, outputs: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], *args, **kwargs
    ) -> torch.Tensor:
        """
        Computes the Pair Mean Absolute Deviation Loss with optional weighting.
        Args:
            outputs (Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]): A tuple containing:
                - predicted (torch.Tensor): The predicted outputs from the model.
                - target (torch.Tensor): The ground truth targets.
                - weights (Optional[torch.Tensor]): Optional weights for each sample in the batch.
        Returns:
            torch.Tensor: The computed loss value.
        """

        if len(outputs) < 2 or outputs[0].shape != outputs[1].shape:
            raise ValueError(
                f"outputs[0] and outputs[1] must have the same shape, " f"got {outputs[0].shape} vs {outputs[1].shape}"
            )
        predicted, target = outputs[0], outputs[1]

        loss = F.l1_loss(predicted, target, reduction="none")

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
