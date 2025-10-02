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
        img_size: int,
        batch_size: int,
    ):
        self.model = model
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.img_size = img_size

    def __len__(self) -> int:
        return self.num_samples

    @torch.no_grad()
    def __getitem__(self, *args, **kwargs) -> Tuple[torch.Tensor, str]:
        sample = self.model.sample(1, self.device, self.img_size, batch_size=1).squeeze(
            0
        )  # (C,H,W)
        return sample, str(0)  # label not needed


"""
def get_generated_dataloader(model, num_samples, device, img_size, batch_size=64):
    dataset = GeneratedDataset(model, num_samples, device, img_size, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader
"""
