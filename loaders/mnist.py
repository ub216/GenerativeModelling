import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataset(MNIST):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super(MNISTDataset, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, index):
        img, target = super(MNISTDataset, self).__getitem__(index)
        return img, target


def get_mnist_dataloader(
    root,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    transform=transforms.ToTensor(),
    persistent_workers=False,
):
    dataset = MNISTDataset(root=root, train=True, download=True, transform=transform)
    dataloader = data.DataLoader(
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
