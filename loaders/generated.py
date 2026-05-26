from typing import Tuple

import torch
from torch.utils.data import IterableDataset

import helpers.custom_types as custom_types


class GeneratedDataset(IterableDataset):
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
    def __iter__(self):
        generated = 0
        while generated < self.num_samples:
            bs = min(self.batch_size, self.num_samples - generated)
            samples = self.model.sample(bs, self.device, self.image_size, batch_size=bs)
            for sample in samples:
                yield sample, str(0)
            generated += bs
