from typing import Tuple

import torch
from torch.utils.data import Dataset

import helpers.custom_types as custom_types


class GeneratedDataset(Dataset):
    def __init__(
        self,
        model: custom_types.GenBaseModel,
        num_samples: int,
        device: custom_types.DeviceType,
        image_size: int | Tuple[int, int],
        batch_size: int,
    ):
        self.model = model
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    @torch.no_grad()
    def __getitem__(self, *args, **kwargs) -> Tuple[torch.Tensor, str]:
        sample = self.model.sample(1, self.device, self.image_size, batch_size=1).squeeze(
            0
        )  # (C,H,W)
        return sample, str(0)  # label not needed

