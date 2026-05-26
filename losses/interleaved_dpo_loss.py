from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterleavedDPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args, **kwargs):
        """
        Computes the DPO Loss assuming data is interleaved as [W1, L1, W2, L2, ...].
        Args:
            outputs (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing:
                - pol_pred (torch.Tensor): The policy model predictions.
                - ref_pred (torch.Tensor): The reference model predictions.
                - target_noise (torch.Tensor): The ground truth noisy targets.
        Returns:
            torch.Tensor: The computed loss value.
        """
        if len(outputs) != 3:
            raise ValueError(f"outputs must contain (pol_pred, ref_pred, target_noise), got {len(outputs)} elements")
        pol_pred, ref_pred, target_noise = outputs
        if not (pol_pred.shape == ref_pred.shape == target_noise.shape):
            raise ValueError(
                f"output tensors must have the same shape, "
                f"got pol_pred={pol_pred.shape}, ref_pred={ref_pred.shape}, target_noise={target_noise.shape}"
            )

        # calculate Per-Sample MSE [B_total]
        # B_total is 2 * batch_size_pairs
        pol_err = F.mse_loss(pol_pred, target_noise, reduction="none").mean(dim=[1, 2, 3])
        ref_err = F.mse_loss(ref_pred, target_noise, reduction="none").mean(dim=[1, 2, 3])

        reward = ref_err - pol_err
        # Assumes Interleaving: [W1, L1, W2, L2, ...]
        logits = reward[0::2] - reward[1::2]
        loss = -F.logsigmoid(self.beta * logits).mean()

        return loss
