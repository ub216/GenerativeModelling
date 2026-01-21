import os
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CelebAHQDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform=None,
        attr_target: Union[str, List[str]] = "all",
        use_negation: bool = False,
    ):
        """
        Args:
            root: Path to 'data/datasets/CelebA/celeba_hq'
            transform: torchvision transforms
            attr_target: Attribute(s) to include in caption
            use_negation: Whether to include "no [attribute]" in caption
        """
        self.image_dir = os.path.join(root, "CelebA-HQ-img")
        self.mapping_file = os.path.join(root, "CelebA-HQ-to-CelebA-mapping.txt")
        self.attr_file = os.path.join(root, "list_attr_celeba.txt")

        self.transform = transform
        self.use_negation = use_negation

        # load the Mapping File
        # Expected cols: idx, orig_idx, orig_file
        mapping_df = pd.read_csv(self.mapping_file, sep="\s+")

        # load the Original Attributes
        # Line 1: Total count, Line 2: Attr names. Use sep='\s+' for varying whitespace.
        attr_df = pd.read_csv(self.attr_file, sep="\s+", skiprows=1)
        self.all_labels = list(attr_df.columns)
        self.all_labels = [label.strip().lower() for label in self.all_labels]

        # create the HQ-specific attribute table
        # We use 'orig_idx' from mapping to pick the correct rows from original attributes
        # Note: In most mapping files, orig_idx is the 0-based index of the original 202k set
        hq_orig_indices = mapping_df["orig_idx"].values
        self.attributes = attr_df.iloc[hq_orig_indices].reset_index(drop=True)

        # HQ filenames are usually '0.jpg', '1.jpg', etc.
        self.filenames = [f"{i}.jpg" for i in range(len(mapping_df))]

        # handle attribute selection
        if attr_target == "all":
            self.selected_indices = list(range(len(self.all_labels)))
        else:
            if isinstance(attr_target, str):
                attr_target = [attr_target]
            self.selected_indices = [self.all_labels.index(a) for a in attr_target]

        self.selected_labels = [self.all_labels[i] for i in self.selected_indices]

    def _generate_caption(self, attr_values: torch.Tensor) -> str:
        tags = []
        for i, val in enumerate(attr_values):
            label_name = self.selected_labels[i].replace("_", " ")
            if val > 0:
                tags.append(label_name)
            elif self.use_negation:
                tags.append(f"no {label_name}")
        return ", ".join(tags) if tags else ""

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        # Load Image
        img_path = os.path.join(self.image_dir, self.filenames[index])
        img = Image.open(img_path).convert("RGB")

        # Get Attributes for this specific HQ index
        attr_row = self.attributes.iloc[index].values
        # CelebA uses -1 and 1; we convert to float tensor
        selected_attr_values = torch.tensor(
            attr_row[self.selected_indices].astype(float)
        )

        if self.transform:
            img = self.transform(img)

        caption = self._generate_caption(selected_attr_values)
        return img, caption


def get_celeb_hq_dataloader(
    root: str,
    batch_size: int = 32,
    image_size: int = 256,
    attr_target: Union[str, List[str]] = "all",
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    # Preprocessing for Latent Diffusion
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # Maps to [0, 1]
        ]
    )

    dataset = CelebAHQDataset(root=root, transform=transform, attr_target=attr_target)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    dataloader.image_size = image_size
    return dataloader
