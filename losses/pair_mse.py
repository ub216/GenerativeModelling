from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import helpers.custom_types as custom_types


class PairMSELoss(nn.Module):
    def __init__(self, reduction: custom_types.ReductionType = "mean"):
        assert reduction in [
            "sum",
            "mean",
        ], "Reduction must be 'sum' or 'mean'"
        self.reduction = reduction

    def forward(
        self, outputs: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        assert (
            len(outputs) == 2 and outputs[0].shape == outputs[1].shape
        ), "Outputs and inputs must have the same shape"
        predicted, input = outputs
        loss = F.mse_loss(predicted, input, reduction=self.reduction)
        return loss
