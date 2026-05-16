from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA

from helpers.image_utils import get_transforms

class CelebDataset(CelebA):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        download: bool = False,
        attr_target: Union[str, List[str]] = "all",
        use_negation: bool = False,  # Toggle "no [attribute]" vs omitting it
    ):
        super(CelebDataset, self).__init__(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # Standard 40 CelebA attributes
        self.all_labels = [
            "5_o_clock_shadow",
            "arched_eyebrows",
            "attractive",
            "bags_under_eyes",
            "bald",
            "bangs",
            "big_lips",
            "big_nose",
            "black_hair",
            "blond_hair",
            "blurry",
            "brown_hair",
            "bushy_eyebrows",
            "chubby",
            "double_chin",
            "eyeglasses",
            "goatee",
            "gray_hair",
            "heavy_makeup",
            "high_cheekbones",
            "male",
            "mouth_slightly_open",
            "mustache",
            "narrow_eyes",
            "no_beard",
            "oval_face",
            "pale_skin",
            "pointy_nose",
            "receding_hairline",
            "rosy_cheeks",
            "sideburns",
            "smiling",
            "straight_hair",
            "wavy_hair",
            "wearing_earrings",
            "wearing_hat",
            "wearing_lipstick",
            "wearing_necklace",
            "wearing_necktie",
            "young",
        ]

        self.use_negation = use_negation

        # Handle attribute selection
        if attr_target == "all":
            self.selected_indices = torch.arange(len(self.all_labels))
        elif isinstance(attr_target, str):
            self.selected_indices = torch.tensor([self.all_labels.index(attr_target)])
        elif isinstance(attr_target, list):
            self.selected_indices = torch.tensor(
                [self.all_labels.index(a) for a in attr_target]
            )

        # Store selected label names for string formatting
        self.selected_labels = [self.all_labels[i] for i in self.selected_indices]

    def _generate_caption(self, attr_values: torch.Tensor) -> str:
        """
        Formats:
        - No attributes: "photo of a face"
        - With attributes: "photo of a smiling, young face"
        """
        present_tags = []

        for i, val in enumerate(attr_values):
            # CelebA labels are usually -1 (False) or 1 (True)
            if val > 0:
                label_name = self.selected_labels[i].replace("_", " ")
                present_tags.append(label_name)
            elif self.use_negation:
                # Optional: If you still want negation, though for your
                # current experiment "a photo of a face" is cleaner.
                label_name = self.selected_labels[i].replace("_", " ")
                present_tags.append(f"non-{label_name}")

        if not present_tags:
            return "photo of a face"

        # Join attributes with commas: "smiling, young"
        attr_string = ", ".join(present_tags)

        # Final prompt: "a photo of a smiling, young face"
        return f"photo of a {attr_string} face"

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        img, all_target = super(CelebDataset, self).__getitem__(index)

        # Filter target to only include selected attributes
        selected_attr_values = all_target[self.selected_indices]

        # Convert to string
        caption = self._generate_caption(selected_attr_values)

        return img, caption


def get_celeb_dataloader(
    root,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int | Tuple[int, int] = (64, 64),
    persistent_workers: bool = False,
    split: str = "all",
    attr_target: str = "all",
) -> DataLoader:
    """
    transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),  # data augmentation
            transforms.ToTensor(),  # -> [0,1]
        ]
    )
    """
    transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            get_transforms(image_size),
        ]
    )
    dataset = CelebDataset(
        root=root,
        split=split,
        download=True,
        transform=transform,
        attr_target=attr_target,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    dataloader.image_size = image_size
    return dataloader
