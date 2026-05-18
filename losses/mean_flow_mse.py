from typing import Dict

import torch
import torch.nn as nn

import helpers.custom_types as custom_types
from losses.pair_mse import PairMSELoss


class MeanFlowMSELoss(nn.Module):
    """
    Wraps PairMSELoss for MeanFlow, splitting metrics by the boundary condition mask.

    Expects model outputs as (pred_u, target_u, same_time) where same_time is a
    bool tensor [B] indicating samples where r==t (boundary condition, Eq 7).

    Returns a dict with:
      "all"      — full-batch loss (has grad; used for backprop)
      "boundary" — loss on r==t samples (no_grad; wandb monitoring only)
      "meanflow" — loss on r<t samples  (no_grad; wandb monitoring only)

    The "boundary" and "meanflow" keys are omitted when their group is empty.
    """

    def __init__(
        self,
        reduction: custom_types.ReductionType = "adaptive",
        adaptive_power: float = 0.5,
        adaptive_eps: float = 1e-3,
    ):
        super().__init__()
        self._loss = PairMSELoss(reduction=reduction, adaptive_power=adaptive_power, adaptive_eps=adaptive_eps)

    def forward(self, outputs, *args, **kwargs) -> Dict[str, torch.Tensor]:
        assert len(outputs) >= 3, "MeanFlowMSELoss requires (pred, target, same_time_mask) from model"
        pred, target, same_time = outputs[0], outputs[1], outputs[2]

        result = {"all": self._loss((pred, target))}

        with torch.no_grad():
            if same_time.any():
                result["boundary"] = self._loss((pred[same_time], target[same_time])).detach()
            if (~same_time).any():
                result["meanflow"] = self._loss((pred[~same_time], target[~same_time])).detach()

        return result
