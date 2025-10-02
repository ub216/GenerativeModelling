from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataset(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = False,
    ):
        super(MNISTDataset, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        img, target = super(MNISTDataset, self).__getitem__(index)
        return img, str(target)


def get_mnist_dataloader(
    root,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    transform=transforms.ToTensor(),
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = MNISTDataset(root=root, train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    dataloader.img_size = 28
    return dataloader
