from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANHingeLoss(nn.Module):
    def __init__(
        self,
        generative_weight: float = 1.0,
        discriminative_weight: float = 1.0,
    ):
        super().__init__()
        self.generative_weight = generative_weight
        self.discriminative_weight = discriminative_weight

    def forward(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the GAN higne loss
        returns: generator loss, discriminator loss
        """
        if len(outputs) != 3 or not (outputs[0].shape == outputs[1].shape == outputs[2].shape):
            raise ValueError(
                f"outputs must be 3 tensors of equal shape, got lengths={len(outputs)}"
                + (f" shapes={outputs[0].shape}, {outputs[1].shape}, {outputs[2].shape}" if len(outputs) == 3 else "")
            )
        generator_score_gen, generator_score_dis, real_score = outputs

        discriminator_loss = torch.mean(F.relu(1.0 - real_score)) + torch.mean(F.relu(1.0 - generator_score_dis))
        generator_loss = -torch.mean(generator_score_gen)

        return {"generator": generator_loss, "discriminator": discriminator_loss}
