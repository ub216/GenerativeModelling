import torch.nn.functional as F


class DiffusionLoss:
    def __init__(self, reduction="mean"):
        assert reduction in [
            "sum",
            "mean",
            "none",
        ], "Reduction must be 'sum', 'mean', or 'none'"
        self.reduction = reduction

    def __call__(self, outputs, *args, **kwargs):
        assert (
            len(outputs) == 2 and outputs[0].shape == outputs[1].shape
        ), "Outputs and inputs must have the same shape"
        predicted, input = outputs
        loss = F.mse_loss(predicted, input, reduction=self.reduction)
        return loss
