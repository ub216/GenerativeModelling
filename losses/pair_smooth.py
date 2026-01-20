from typing import Dict, Tuple

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
        self, outputs: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        assert (
            len(outputs) == 2 and outputs[0].shape == outputs[1].shape
        ), "Outputs and inputs must have the same shape"
        predicted, input = outputs
        loss = F.smooth_l1_loss(
            predicted, input, reduction=self.reduction, beta=self.beta
        )
        return loss
