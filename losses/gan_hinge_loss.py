from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import helpers.custom_types as custom_types


class GANHingeLoss(nn.Module):
    def __init__(
        self,
        generative_weight: float = 1.0,
        discriminative_weight: float = 1.0,
    ):
        self.generative_weight = generative_weight
        self.discriminative_weight = discriminative_weight

    def forward(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the GAN higne loss
        returns: generator loss, discriminator loss
        """
        assert (
            len(outputs) == 2 and outputs[0].shape == outputs[1].shape
        ), "Outputs and inputs must have the same shape"
        generator_score, real_score = outputs

        discriminator_loss = torch.mean(F.relu(1.0 - real_score)) + torch.mean(
            F.relu(1.0 - generator_score)
        )
        generator_loss = -torch.mean(generator_score)

        return generator_loss, discriminator_loss
